import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import differential_evolution

# Constants from the task contract
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
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    """Build the 9x9 Liouvillian superoperator."""
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

def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n):
    """
    Simulate the pulse for 6 input states simultaneously.
    Returns mean fidelity.
    """
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    # Initial density matrices vectorized
    initial_states = []
    for psi in inputs:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    # Stack into a 9x6 matrix
    states = np.column_stack(initial_states)
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        
        states = U_super @ states
    
    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
        
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
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
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _compute_score_smooth(pulse, a, a_dag, n, beta=50.0):
    """Compute a smooth surrogate of mean + max using soft-max via log-sum-exp."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    mean_infid = np.mean(infidelities)
    
    # Soft-max: beta * log(sum(exp(infids / beta))) approximates max as beta -> inf
    max_infid = beta * np.log(np.sum(np.exp(infidelities / beta)))
    
    return mean_infid + max_infid

def _generate_pulse_drag(params):
    """Generate a DRAG-style pulse from parameters [A, sigma, beta]."""
    A, sigma, beta = params
    
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    # Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to 0 at edges
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Normalize area to 1
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = gauss
    
    # Derivative
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = A * gauss_norm
    Q = beta * A * dgauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        
    return pulse

def _generate_pulse_gaussians(params):
    """
    Generate a flexible pulse from 4+4 independent Gaussian components.
    params: [A1, t1, s1, B1, A2, t2, s2, B2, A3, t3, s3, B3, A4, t4, s4, B4]
    where (A_i, t_i, s_i, B_i) are amplitude, center, sigma, and Q-weight for Gaussian i.
    I = sum A_i * exp(-0.5*((t-t_i)/s_i)^2)
    Q = sum B_i * exp(-0.5*((t-t_i)/s_i)^2)
    """
    t = np.arange(N_STEPS) * DT
    
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    
    for k in range(4):
        A_k = params[4*k]
        t_k = params[4*k + 1]
        s_k = params[4*k + 2]
        B_k = params[4*k + 3]
        
        if s_k < 0.5:
            s_k = 0.5
        if s_k > 10.0:
            s_k = 10.0
            
        gauss = np.exp(-0.5 * ((t - t_k) / s_k)**2)
        I += A_k * gauss
        Q += B_k * gauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit, use 300s with safety margin
    
    a, a_dag, n = _build_matrices()
    
    # Initial guess: DRAG pulse
    x0_drag = np.array([0.3, 5.0, 0.2])
    best_pulse = _generate_pulse_drag(x0_drag)
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Phase 1: Differential Evolution with flexible 4+4 Gaussian basis
    # Parameter bounds for 4+4 Gaussians:
    # A_i in [-0.4, 0.4], t_i in [0, 40], s_i in [1, 10], B_i in [-0.4, 0.4]
    bounds_de = []
    for k in range(4):
        bounds_de.append((-0.4, 0.4))   # A_k
        bounds_de.append((0, 40))       # t_k
        bounds_de.append((1, 10))       # s_k
        bounds_de.append((-0.4, 0.4))   # B_k
    
    def objective_de(params):
        p = _generate_pulse_gaussians(params)
        if not _check_validity(p):
            return 1.0  # Penalty for invalid
        return _compute_score_smooth(p, a, a_dag, n)
    
    try:
        if time.time() < deadline - 120.0:
            result_de = differential_evolution(
                objective_de, bounds_de,
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, atol=1e-8,
                init='sobol', mutation=(0.3, 1.0), recombination=0.9,
                polish=False, updating='immediate'
            )
            
            if result_de.success or result_de.fun < best_score:
                p_candidate = _generate_pulse_gaussians(result_de.x)
                if _check_validity(p_candidate):
                    score_candidate = _compute_score(p_candidate, a, a_dag, n)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_pulse = p_candidate
    except Exception:
        pass
    
    # Phase 2: Local refinement with Nelder-Mead on the DRAG basis
    if time.time() < deadline - 60.0:
        def objective_smooth_drag(params):
            p = _generate_pulse_drag(params)
            if not _check_validity(p):
                return 1.0
            return _compute_score_smooth(p, a, a_dag, n)
        
        from scipy.optimize import minimize
        
        starts = [
            np.array([0.3, 5.0, 0.2]),
            np.array([0.35, 4.0, 0.1]),
            np.array([0.25, 6.0, 0.3]),
            np.array([0.4, 5.0, 0.0]),
            np.array([0.3, 5.0, -0.1]),
            np.array([0.32, 4.5, 0.15]),
            np.array([0.28, 5.5, 0.25]),
            np.array([0.38, 4.2, 0.22]),
            np.array([0.22, 7.0, 0.15]),
        ]
        
        for x_start in starts:
            if time.time() > deadline - 60.0:
                break
            try:
                res = minimize(objective_smooth_drag, x_start, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-8})
                p_candidate = _generate_pulse_drag(res.x)
                if _check_validity(p_candidate):
                    score_candidate = _compute_score(p_candidate, a, a_dag, n)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_pulse = p_candidate
            except Exception:
                pass
    
    # Phase 3: Final polish with exact objective on DRAG basis
    if time.time() < deadline - 30.0:
        def objective_exact_drag(params):
            p = _generate_pulse_drag(params)
            if not _check_validity(p):
                return 1.0
            return _compute_score(p, a, a_dag, n)
        
        from scipy.optimize import minimize
        
        # Start from a few good initial points
        for x_start in [np.array([0.3, 5.0, 0.2]), np.array([0.35, 4.5, 0.15]), np.array([0.28, 5.5, 0.2])]:
            if time.time() > deadline - 30.0:
                break
            try:
                res = minimize(objective_exact_drag, x_start, method='Nelder-Mead',
                               options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-10})
                p_candidate = _generate_pulse_drag(res.x)
                if _check_validity(p_candidate):
                    score_candidate = _compute_score(p_candidate, a, a_dag, n)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_pulse = p_candidate
            except Exception:
                pass
    
    return best_pulse
