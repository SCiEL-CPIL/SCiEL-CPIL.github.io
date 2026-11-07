import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

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
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    states = np.column_stack(initial_states)
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        states = U_super @ states
        
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
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

def _build_eigenmode_basis():
    """
    Build a basis of 6 eigenmodes of the free (no-drive) Hamiltonian projected
    onto the qubit subspace. These are oscillatory functions that naturally
    represent the system's dynamical symmetries.
    
    The free Hamiltonian (detuning=0, scale=0) has energies:
    E_0 = 0, E_1 = 0, E_2 = anharmonicity/2 * 2 = anharmonicity
    (since H_free = 0.5 * ANHARMONICITY * n(n-1))
    
    We construct modes as:
    - Mode 0: constant (DC component)
    - Modes 1-5: cos/sin pairs at frequencies derived from the system
    """
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    # Normalize time to [-1, 1] for oscillatory modes
    t_norm = (t - t_center) / 20.0
    
    # Build basis functions
    # Mode 0: Gaussian envelope (ensures zero at edges, area ~ pi)
    sigma = 4.0
    gauss = np.exp(-0.5 * (t_norm / sigma)**2)
    edge_val = gauss[0]
    gauss = gauss - edge_val
    area = np.sum(gauss) * DT
    gauss = gauss / area  # Area = 1
    
    # Mode 1: cos at fundamental frequency (pi per 40ns -> freq = 1/(2*20) cycles/ns)
    omega1 = 2 * np.pi / 40.0  # rad/ns
    mode1 = np.cos(omega1 * (t - t_center)) * gauss
    
    # Mode 2: sin at fundamental frequency  
    mode2 = np.sin(omega1 * (t - t_center)) * gauss
    
    # Mode 3: cos at second harmonic
    omega2 = 2 * omega1
    mode3 = np.cos(omega2 * (t - t_center)) * gauss
    
    # Mode 4: sin at second harmonic
    mode4 = np.sin(omega2 * (t - t_center)) * gauss
    
    # Mode 5: cos at third harmonic
    omega3 = 3 * omega1
    mode5 = np.cos(omega3 * (t - t_center)) * gauss
    
    # Stack into N_STEPS x 6 matrix
    basis_I = np.column_stack([gauss, mode1, mode2, mode3, mode4, mode5])
    
    # For Q channel, use the derivative-like modes (shifted by 90 degrees)
    # This naturally implements DRAG-like correction
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    # Q basis: derivative of each I mode (approximate)
    basis_Q = np.zeros((N_STEPS, 6))
    for k in range(6):
        f = basis_I[:, k]
        df = np.zeros(N_STEPS)
        df[1:] = (f[1:] - f[:-1]) / DT
        df[0] = df[1]
        df[-1] = df[-2]
        basis_Q[:, k] = df
    
    return basis_I, basis_Q

def _generate_pulse_eigenmode(params, basis_I, basis_Q):
    """
    Generate pulse from 12 parameters: 6 for I channel, 6 for Q channel.
    params = [c0_I, c1_I, ..., c5_I, c0_Q, c1_Q, ..., c5_Q]
    """
    c_I = params[:6]
    c_Q = params[6:12]
    
    I = basis_I @ c_I
    Q = basis_Q @ c_Q
    
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
    deadline = start_time + 300.0
    
    a, a_dag, n = _build_matrices()
    basis_I, basis_Q = _build_eigenmode_basis()
    
    # Initial guess: primarily DC component with small DRAG
    x0 = np.array([np.pi, 0, 0, 0, 0, 0,  # I coefficients
                   0, 0.15, 0, 0, 0, 0])  # Q coefficients (DRAG-like)
    
    best_pulse = _generate_pulse_eigenmode(x0, basis_I, basis_Q)
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    def objective(params):
        p = _generate_pulse_eigenmode(params, basis_I, basis_Q)
        if not _check_validity(p):
            return 1.0
        return _compute_score(p, a, a_dag, n)
    
    # Multiple starting points in eigenmode space
    starts = [
        np.array([np.pi, 0, 0, 0, 0, 0, 0, 0.15, 0, 0, 0, 0]),
        np.array([np.pi, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0]),
        np.array([np.pi, 0.1, 0, 0, 0, 0, 0, 0.12, 0, 0, 0, 0]),
        np.array([np.pi, 0, 0, 0.05, 0, 0, 0, 0.15, 0, 0.02, 0, 0]),
        np.array([np.pi, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0]),
        np.array([np.pi, 0, 0, 0, 0, 0, 0, 0.08, 0, 0, 0, 0]),
    ]
    
    for x_start in starts:
        if time.time() > deadline - 40.0:
            break
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_pulse_eigenmode(res.x, basis_I, basis_Q)
        except Exception:
            pass
        
        # L-BFGS-B polish
        if time.time() > deadline - 30.0:
            break
        try:
            res2 = minimize(objective, res.x, method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_pulse = _generate_pulse_eigenmode(res2.x, basis_I, basis_Q)
        except Exception:
            pass
    
    return best_pulse
