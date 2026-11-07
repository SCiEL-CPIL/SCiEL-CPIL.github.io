import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5

def _build_matrices():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    I3 = np.eye(3, dtype=complex)
    
    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))
    
    return L_total

def _simulate_pulse(pulse, detuning, scale, a, a_dag, n):
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    states = np.column_stack([np.outer(psi, psi.conj()).flatten(order='F') for psi in inputs])
    
    for i in range(N_STEPS):
        I_val, Q_val = pulse[i, 0], pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        states = U_super @ states
    
    fidelities = []
    for j, psi_in in enumerate(inputs):
        psi_target = X_mat @ psi_in
        rho_out = states[:, j].reshape((3, 3), order='F')
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelities.append(np.clip(fidelity, 0.0, 1.0))
    
    return np.mean(fidelities)

def _check_validity(pulse):
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False
    return True

def _compute_score(pulse, a, a_dag, n):
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse(pulse, detuning, scale, a, a_dag, n)
            infidelities.append(1.0 - fid)
    return np.mean(infidelities) + np.max(infidelities)

def _generate_two_gaussian_drag_pulse(params):
    """Two Gaussian basis with DRAG term.
    params: [A1, sigma1, t1, A2, sigma2, t2, beta]
    I = A1*G1 + A2*G2 (offset to 0 at edges, area normalized)
    Q = beta * dI/dt
    """
    A1, sigma1, t1, A2, sigma2, t2, beta = params
    
    t = np.arange(N_STEPS) * DT
    
    def gaussian(A, sigma, center):
        g = A * np.exp(-0.5 * ((t - center) / sigma)**2)
        edge = A * np.exp(-0.5 * ((0 - center) / sigma)**2)
        g = g - edge
        return g
    
    G1 = gaussian(A1, sigma1, t1)
    G2 = gaussian(A2, sigma2, t2)
    
    I = G1 + G2
    
    # Normalize I to have area pi
    area = np.sum(I) * DT
    if abs(area) > 1e-15:
        I = I * (np.pi / area)
    else:
        I = np.zeros(N_STEPS)
    
    # DRAG term: Q = beta * dI/dt
    dI = np.zeros(N_STEPS)
    dI[1:] = (I[1:] - I[:-1]) / DT
    dI[0] = dI[1]
    dI[-1] = dI[-2]
    Q = beta * dI
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    # Enforce slew bound by scaling if needed
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND:
            scale_factor = SLEW_BOUND / diff
            pulse *= scale_factor
            break
    
    return pulse

def _objective(params, a, a_dag, n):
    pulse = _generate_two_gaussian_drag_pulse(params)
    if not _check_validity(pulse):
        return 1.0
    return _compute_score(pulse, a, a_dag, n)

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    a, a_dag, n = _build_matrices()
    
    # Initial DRAG seed from task contract
    t = np.arange(N_STEPS) * DT
    sigma = 40 / 6
    gauss = np.exp(-0.5 * ((t - 20.0) / sigma)**2)
    edge_val = np.exp(-0.5 * ((0 - 20.0) / sigma)**2)
    gauss = gauss - edge_val
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = np.zeros(N_STEPS)
    
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    seed_pulse = np.column_stack([gauss_norm, 0.2 * dgauss])
    amps = np.sqrt(seed_pulse[:, 0]**2 + seed_pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        seed_pulse *= AMPLITUDE_BOUND / max_amp
    
    best_pulse = seed_pulse.copy()
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Try several starting points for optimization
    starts = [
        [0.3, 5.0, 20.0, 0.0, 5.0, 20.0, 0.2],   # Single Gaussian with DRAG
        [0.25, 4.0, 18.0, 0.15, 6.0, 22.0, 0.15], # Two Gaussians
        [0.2, 4.5, 19.0, 0.2, 5.5, 21.0, 0.25],   # Two Gaussians
        [0.3, 5.0, 20.0, 0.0, 5.0, 20.0, 0.1],    # Single Gaussian, lower DRAG
        [0.3, 5.0, 20.0, 0.0, 5.0, 20.0, 0.3],    # Single Gaussian, higher DRAG
        [0.2, 4.0, 17.0, 0.2, 6.0, 23.0, 0.2],    # Spread two Gaussians
    ]
    
    for x_start in starts:
        if time.time() > deadline - 20.0:
            break
        
        try:
            res = minimize(_objective, x_start, args=(a, a_dag, n), 
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_two_gaussian_drag_pulse(res.x)
        except Exception:
            pass
    
    # If we still have time, try differential evolution for global search
    if time.time() < deadline - 40.0:
        try:
            bounds = [
                (0.05, 0.4),   # A1
                (2.0, 8.0),    # sigma1
                (15.0, 25.0),  # t1
                (-0.2, 0.3),   # A2
                (2.0, 8.0),    # sigma2
                (15.0, 25.0),  # t2
                (-0.5, 0.5),   # beta
            ]
            de_res = differential_evolution(_objective, bounds, args=(a, a_dag, n),
                                            maxiter=50, tol=1e-5, seed=42,
                                            polish=True, init='sobol')
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_pulse = _generate_two_gaussian_drag_pulse(de_res.x)
        except Exception:
            pass
    
    return best_pulse
