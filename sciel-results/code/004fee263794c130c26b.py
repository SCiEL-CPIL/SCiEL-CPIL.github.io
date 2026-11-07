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

# Precompute matrices
# Basis: |0>, |1>, |2>
a_mat = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat

# Lindblad operators
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2 / TPHI)

# Precompute constant parts of Liouvillian
I3 = np.eye(3, dtype=complex)
L1_dag_L1 = L1.conj().T @ L1
L2_dag_L2 = L2.conj().T @ L2

L_const = np.kron(L1.conj(), L1) - 0.5 * (np.kron(L1_dag_L1, I3) + np.kron(I3, L1_dag_L1.T))
L_const += np.kron(L2.conj(), L2) - 0.5 * (np.kron(L2_dag_L2, I3) + np.kron(I3, L2_dag_L2.T))

# Precompute Hamiltonian components
a_plus_adag = a_mat + a_dag
a_minus_adag = a_mat - a_dag
n_nm1 = np.diag([0, 0, 2])
anharmonic_term = 0.5 * ANHARMONICITY * n_nm1

# Test states and targets
states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1, 1j, 0],
    [1, -1j, 0]
], dtype=complex)
norms = np.linalg.norm(states, axis=1)
states = states / norms[:, np.newaxis]

targets = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1, -1j, 0],
    [1, 1j, 0]
], dtype=complex)
norms = np.linalg.norm(targets, axis=1)
targets = targets / norms[:, np.newaxis]

# Precompute initial density matrices
rho_initial = [np.outer(s, s.conj()).flatten() for s in states]

def _generate_pulse(params):
    """
    params: [beta, area_scale, sigma]
    Returns (80, 2) array of [I, Q]
    """
    beta, area_scale, sigma = params
    
    sigma = np.clip(sigma, 2.0, 15.0)
    area_scale = np.clip(area_scale, 0.8, 1.2)
    beta = np.clip(beta, -2.0, 2.0)
    
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT/2
    
    # Gaussian envelope with area = pi * area_scale
    A = np.sqrt(np.pi / 2) * area_scale / sigma
    
    I = A * np.exp(-t**2 / (2 * sigma**2))
    
    # DRAG Q component proportional to derivative of I
    dI_dt = -t / (sigma**2) * I
    Q = beta * dI_dt
    
    pulse = np.column_stack((I, Q))
    
    # Enforce amplitude bound
    amplitudes = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amplitudes)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        
    # Enforce slew rate
    diffs = np.diff(pulse, axis=0)
    diffs = np.vstack([pulse[0], diffs, -pulse[-1]])
    norms_diff = np.linalg.norm(diffs, axis=1)
    max_slew = np.max(norms_diff) / DT
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        pulse *= scale_factor
        
    return pulse

def _simulate_pulse_fast(pulse, detuning, scale):
    """
    Fast simulation using piecewise-constant Liouvillian with expm per step.
    Simulates all 6 states in parallel.
    """
    rho_all = np.array(rho_initial, dtype=complex)  # shape (6, 9)
    
    for step in range(N_STEPS):
        I_val = pulse[step, 0]
        Q_val = pulse[step, 1]
        
        drive_term = 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        H = detuning * n_mat + anharmonic_term + drive_term
        
        L_H = -1j * np.kron(I3, H.T) + 1j * np.kron(H.T, I3)
        L_step = L_H + L_const
        
        exp_L = expm(L_step * DT)
        rho_all = (exp_L @ rho_all.T).T
    
    # Calculate fidelities
    fidelities = np.zeros(6)
    for i in range(6):
        rho = rho_all[i].reshape(3, 3)
        psi_t = targets[i]
        F = np.real(np.vdot(psi_t, rho @ psi_t))
        fidelities[i] = np.clip(F, 0.0, 1.0)
        
    return np.mean(fidelities)

def _compute_score(params):
    """
    Compute the score for a given set of pulse parameters.
    Score = mean(infidelities) + max(infidelities)
    """
    pulse = _generate_pulse(params)
    
    infidelities = []
    for det in DE_TUNINGS:
        for scale in SCALES:
            fid = _simulate_pulse_fast(pulse, det, scale)
            infidelities.append(1 - fid)
            
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 280s margin for 330s limit
    
    # Initial guess - DRAG coefficient for transmon
    x0 = np.array([-0.5, 1.0, 40.0/6.0])
    
    best_params = x0.copy()
    best_score = _compute_score(x0)
    
    # Coarse grid search
    if time.time() < deadline - 60:
        sigma_vals = np.array([5.0, 6.0, 6.667, 7.0, 8.0])
        beta_vals = np.array([-1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
        
        for sigma in sigma_vals:
            for beta in beta_vals:
                if time.time() > deadline - 60:
                    break
                params = np.array([beta, 1.0, sigma])
                score = _compute_score(params)
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
    
    # Fine local optimization around best
    if time.time() < deadline - 30:
        try:
            res = minimize(_compute_score, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Second round of fine tuning
    if time.time() < deadline - 20:
        try:
            res2 = minimize(_compute_score, best_params, method='Nelder-Mead',
                            options={'xatol': 1e-5, 'fatol': 1e-9, 'maxiter': 100, 'adaptive': True})
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass
    
    # Generate final pulse
    final_pulse = _generate_pulse(best_params)
    
    # Final validity check
    amplitudes = np.sqrt(final_pulse[:, 0]**2 + final_pulse[:, 1]**2)
    if np.any(amplitudes > AMPLITUDE_BOUND):
        scale_factor = AMPLITUDE_BOUND / np.max(amplitudes)
        final_pulse *= scale_factor
        
    diffs = np.diff(final_pulse, axis=0)
    diffs = np.vstack([final_pulse[0], diffs, -final_pulse[-1]])
    norms_diff = np.linalg.norm(diffs, axis=1)
    max_slew = np.max(norms_diff) / DT
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        final_pulse *= scale_factor
        
    return final_pulse

# EVOLVE-BLOCK-END
