import numpy as np
import time
from scipy.linalg import expm

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

def _simulate_pulse(pulse, detuning, scale, a, a_dag, n):
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
    
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    # Initial density matrices vectorized
    initial_states = []
    for psi in inputs:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    states = np.column_stack(initial_states)
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        
        states = U_super @ states
    
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
            fid = _simulate_pulse(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _drag_pulse(A, sigma, beta):
    """Generate a DRAG pulse from parameters."""
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = gauss
    
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

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0  # Leave 30s safety margin
    
    a, a_dag, n = _build_matrices()
    
    # Try a grid of DRAG parameters
    best_pulse = None
    best_score = np.inf
    
    # Coarse grid
    A_vals = [0.25, 0.3, 0.35, 0.4]
    sigma_vals = [4.0, 5.0, 6.0, 7.0]
    beta_vals = [-0.1, 0.0, 0.1, 0.2, 0.3]
    
    for A in A_vals:
        for sigma in sigma_vals:
            for beta in beta_vals:
                if time.time() > deadline - 60.0:
                    break
                
                p = _drag_pulse(A, sigma, beta)
                if not _check_validity(p):
                    continue
                
                score = _compute_score(p, a, a_dag, n)
                if score < best_score:
                    best_score = score
                    best_pulse = p.copy()
    
    # Fine-tune around the best
    if best_pulse is not None and time.time() < deadline - 30.0:
        # Find the parameters that gave the best score
        # For simplicity, do a local grid search around a few promising points
        fine_A = [0.28, 0.3, 0.32, 0.35, 0.38]
        fine_sigma = [4.5, 5.0, 5.5, 6.0, 6.5]
        fine_beta = [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
        
        for A in fine_A:
            for sigma in fine_sigma:
                for beta in fine_beta:
                    if time.time() > deadline - 30.0:
                        break
                    
                    p = _drag_pulse(A, sigma, beta)
                    if not _check_validity(p):
                        continue
                    
                    score = _compute_score(p, a, a_dag, n)
                    if score < best_score:
                        best_score = score
                        best_pulse = p.copy()
    
    # If we still have time, try a few more points
    if best_pulse is not None and time.time() < deadline - 10.0:
        # Try some extreme values
        extra_configs = [
            (0.4, 4.0, 0.3),
            (0.4, 5.0, 0.2),
            (0.35, 4.5, 0.25),
            (0.4, 6.0, 0.15),
        ]
        
        for A, sigma, beta in extra_configs:
            if time.time() > deadline - 10.0:
                break
            
            p = _drag_pulse(A, sigma, beta)
            if not _check_validity(p):
                continue
            
            score = _compute_score(p, a, a_dag, n)
            if score < best_score:
                best_score = score
                best_pulse = p.copy()
    
    # Return the best pulse found
    if best_pulse is None:
        # Fallback: return a simple DRAG pulse
        best_pulse = _drag_pulse(0.3, 5.0, 0.1)
    
    return best_pulse
