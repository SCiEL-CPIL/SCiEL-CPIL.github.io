# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT

# Matrices
a_mat = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat
n_nm1 = np.diag([0, 0, 2])
anharmonic_term = 0.5 * ANHARMONICITY * n_nm1

L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2 / TPHI)

L1_dag_L1 = L1.conj().T @ L1
L2_dag_L2 = L2.conj().T @ L2

a_plus_adag = a_mat + a_dag
a_minus_adag = a_mat - a_dag

# States and Targets
states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1, 1j, 0],
    [1, -1j, 0]
], dtype=complex) / np.sqrt(np.array([1, 1, 2, 2, 2, 2]))

targets = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1, -1j, 0],
    [1, 1j, 0]
], dtype=complex) / np.sqrt(np.array([1, 1, 2, 2, 2, 2]))

def _generate_pulse(params):
    sigma, alpha, area_scale = params
    sigma = np.clip(sigma, 1.0, 20.0)
    
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT/2
    A = np.sqrt(np.pi / 2) * area_scale / sigma
    I = A * np.exp(-t**2 / (2 * sigma**2))
    dI_dt = -t / (sigma**2) * I
    Q = alpha * dI_dt
    
    pulse = np.column_stack((I, Q))
    
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.any(amplitudes > AMPLITUDE_BOUND):
        scale_factor = AMPLITUDE_BOUND / np.max(amplitudes)
        pulse *= scale_factor
        
    diffs = np.diff(pulse, axis=0)
    diffs = np.vstack([pulse[0], diffs, -pulse[-1]])
    norms = np.linalg.norm(diffs, axis=1)
    max_slew = np.max(norms) / DT
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        pulse *= scale_factor
        
    return pulse

def _simulate_pulse_euler(pulse, detuning, scale):
    # Precompute H components
    I9 = np.eye(9, dtype=complex)
    I3 = np.eye(3, dtype=complex)
    
    # Precompute Lindblad superoperator parts (constant)
    # L_mat_lind = sum (L_k^* \otimes L_k) - 0.5 (L_k^dag L_k \otimes I + I \otimes (L_k^dag L_k)^T)
    L_mat_lind = np.zeros((9, 9), dtype=complex)
    L_mat_lind += np.kron(L1.conj(), L1) - 0.5 * (np.kron(L1_dag_L1, I3) + np.kron(I3, L1_dag_L1.T))
    L_mat_lind += np.kron(L2.conj(), L2) - 0.5 * (np.kron(L2_dag_L2, I3) + np.kron(I3, L2_dag_L2.T))
    
    fidelities = []
    
    for i in range(6):
        psi = states[i]
        rho = np.outer(psi, psi.conj())
        vec_rho = rho.flatten()
        
        for step in range(N_STEPS):
            I_val = pulse[step, 0]
            Q_val = pulse[step, 1]
            
            drive_term = 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
            H = detuning * n_mat + anharmonic_term + drive_term
            
            # Hamiltonian superoperator
            L_mat_ham = -1j * (np.kron(I3, H.T)) + 1j * (np.kron(H.T, I3))
            
            # Total Liouvillian
            L_mat = L_mat_ham + L_mat_lind
            
            # Euler step
            vec_rho = vec_rho + DT * (L_mat @ vec_rho)
            
        rho = vec_rho.reshape((3, 3))
        w, v = np.linalg.eigh(rho)
        psi_final = v[:, -1]
        F = abs(np.vdot(targets[i], psi_final))**2
        fidelities.append(F)
        
    return np.mean(fidelities)

def _compute_score(params):
    pulse = _generate_pulse(params)
    infidelities = []
    for det in DE_TUNINGS:
        for scale in SCALES:
            fid = _simulate_pulse_euler(pulse, det, scale)
            infidelities.append(1 - fid)
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Seed parameters
    x0 = np.array([40.0/6.0, 0.1, 1.0])
    best_params = x0
    best_score = _compute_score(x0)
    
    # Optimization
    if time.time() < deadline - 60:
        try:
            res = minimize(_compute_score, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 100, 'adaptive': True})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception as e:
            pass
            
    final_pulse = _generate_pulse(best_params)
    
    # Final validity check
    amplitudes = np.sqrt(final_pulse[:, 0]**2 + final_pulse[:, 1]**2)
    if np.any(amplitudes > AMPLITUDE_BOUND):
        scale_factor = AMPLITUDE_BOUND / np.max(amplitudes)
        final_pulse *= scale_factor
        
    diffs = np.diff(final_pulse, axis=0)
    diffs = np.vstack([final_pulse[0], diffs, -final_pulse[-1]])
    norms = np.linalg.norm(diffs, axis=1)
    max_slew = np.max(norms) / DT
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        final_pulse *= scale_factor
        
    return final_pulse

# EVOLVE-BLOCK-END
