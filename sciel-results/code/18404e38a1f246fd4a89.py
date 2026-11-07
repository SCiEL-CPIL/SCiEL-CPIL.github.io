"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T2 = 40000.0

N_STEPS = 80
DT = 0.5
DIM = 3

# Build ladder operators for 3-level system
def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

# Precompute static parts of Liouvillian
def build_static_liouvillian_parts():
    """Precompute static (non-drive) parts of the Liouvillian superoperator."""
    dim = DIM
    L_static = np.zeros((dim * dim, dim * dim), dtype=complex)
    
    # Static Hamiltonian: detuning * n + 0.5 * anharmonicity * n*(n-1)
    # This is diagonal, so it contributes to the commutator part
    # We'll handle detuning separately since it varies
    
    # Dissipative terms (same for all cases)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    eye = np.eye(dim)
    
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L_dag_L = L_dag @ L_op
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for m in range(dim):
                        L_static[i * dim + j, k * dim + m] += L_op[i, k] * L_dag[m, j]
                        L_static[i * dim + j, k * dim + m] -= 0.5 * L_dag_L[i, k] * (eye[j, m] + eye[m, j])
    
    return L_static

L_STATIC = build_static_liouvillian_parts()

def build_liouvillian(detuning, scale, I, Q):
    """Build full Liouvillian superoperator for given drive parameters."""
    dim = DIM
    L = L_STATIC.copy()
    
    # Hamiltonian: detuning*n + 0.5*anharm*n*(n-1) + 0.5*scale*(I*(a+a†) + iQ*(a-a†))
    # The n and n*(n-1) terms are diagonal
    # n = diag(0, 1, 2)
    # n*(n-1) = diag(0, 0, 2)
    
    # Coherent part: -i[H, rho]
    # H = H_diag + H_drive
    # H_diag = diag(detuning*0 + 0.5*anharm*0, detuning*1 + 0.5*anharm*0, detuning*2 + 0.5*anharm*2)
    #        = diag(0, detuning, 2*detuning + anharm)
    # H_drive = 0.5*scale*(I*(a+a†) + iQ*(a-a†))
    
    h0 = 0.0
    h1 = detuning
    h2 = 2 * detuning + ANHARMONICITY
    
    # Drive terms:
    # a+a† has elements: <0|a†|1> = 1, <1|a†|0> = 1, <1|a†|2> = sqrt(2), <2|a†|1> = sqrt(2)
    # a-a† has elements: <0|a|1> = 1, <1|a|0> = -1, <1|a|2> = sqrt(2), <2|a|1> = -sqrt(2)
    # Wait, a-a†: a has <1|a|0>=1, <2|a|1>=sqrt(2); a† has <0|a†|1>=1, <1|a†|2>=sqrt(2)
    # So (a-a†)[0,1] = -1, (a-a†)[1,0] = 1, (a-a†)[1,2] = -sqrt(2), (a-a†)[2,1] = sqrt(2)
    
    drive_coeff = 0.5 * scale
    # H_drive elements:
    # H_drive[0,1] = drive_coeff * (I * 1 + iQ * (-1)) = drive_coeff * (I - iQ)
    # H_drive[1,0] = drive_coeff * (I * 1 + iQ * 1) = drive_coeff * (I + iQ)
    # H_drive[1,2] = drive_coeff * (I * sqrt(2) + iQ * (-sqrt(2))) = drive_coeff * sqrt(2) * (I - iQ)
    # H_drive[2,1] = drive_coeff * (I * sqrt(2) + iQ * sqrt(2)) = drive_coeff * sqrt(2) * (I + iQ)
    
    H_drive = np.zeros((dim, dim), dtype=complex)
    H_drive[0, 1] = drive_coeff * (I - 1j * Q)
    H_drive[1, 0] = drive_coeff * (I + 1j * Q)
    H_drive[1, 2] = drive_coeff * np.sqrt(2) * (I - 1j * Q)
    H_drive[2, 1] = drive_coeff * np.sqrt(2) * (I + 1j * Q)
    
    H_full = np.diag([h0, h1, h2]) + H_drive
    
    # Add coherent part to Liouvillian: -i[H, rho]
    # (L_coh rho)_ij = -i(H rho - rho H)_ij = -i sum_k (H_ik rho_kj - rho_ik H_kj)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                L[i * dim + j, k * dim + i] -= 1j * H_full[i, k]
                L[i * dim + j, k * dim + j] += 1j * H_full[k, j]
    
    return L

def check_validity(pulse):
    """Check if the pulse satisfies all constraints."""
    # Amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    
    # Slew bound
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-10:
            return False
        prev = curr
    # From last step to zero
    if np.linalg.norm(prev - np.array([0.0, 0.0])) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def seed_pulse():
    """DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, N_STEPS * DT, N_STEPS)
    sigma = 40.0 / 6.0
    center = N_STEPS * DT / 2.0
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset to make edges zero
    edge_offset = env[0]
    env = env - edge_offset
    env = np.maximum(env, 0)
    
    # Scale to area pi
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    
    # DRAG: add derivative term
    denv = np.gradient(env, DT)
    drag_coeff = 0.1
    
    I = env
    Q = drag_coeff * denv
    
    pulse = np.column_stack([I, Q])
    
    # Ensure validity
    if not check_validity(pulse):
        max_amp = np.max(np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            pulse *= AMPLITUDE_BOUND / max_amp
        
        for i in range(1, N_STEPS):
            delta = np.linalg.norm(pulse[i] - pulse[i-1]) / DT
            if delta > SLEW_BOUND:
                scale = SLEW_BOUND / delta
                pulse[i:] *= scale
    
    return pulse

def compute_score(pulse):
    """Compute the 9-case infidelity score with vectorized simulation."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # 6 initial states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    # Target states after X
    target_states = []
    for state in states:
        ts = np.zeros(3, dtype=complex)
        ts[0] = state[1]
        ts[1] = state[0]
        ts[2] = state[2]
        target_states.append(ts)
    
    target_rhos = [np.outer(ts, ts.conj()) for ts in target_states]
    
    n_det = len(detunings)
    n_scale = len(scales)
    n_states = len(states)
    
    # For each (det, scale) pair, evolve all 6 states
    case_infidelities = []
    
    for di, det in enumerate(detunings):
        for si, scale in enumerate(scales):
            # Evolve all 6 states for this (det, scale)
            # Store density matrices for all states at each step
            rhos = [np.outer(states[s], states[s].conj()) for s in range(n_states)]
            
            for i in range(N_STEPS):
                I_val = pulse[i, 0]
                Q_val = pulse[i, 1]
                L = build_liouvillian(det, scale, I_val, Q_val)
                U = expm(L * DT)
                
                for s in range(n_states):
                    rho_vec = rhos[s].flatten()
                    rho_new_vec = U @ rho_vec
                    rhos[s] = rho_new_vec.reshape(DIM, DIM)
            
            # Compute fidelities
            case_fids = []
            for s in range(n_states):
                fid = np.real(np.trace(rhos[s] @ target_rhos[s]))
                case_fids.append(fid)
            
            mean_fid = np.mean(case_fids)
            infid = 1.0 - mean_fid
            case_infidelities.append(infid)
    
    score = np.mean(case_infidelities) + np.max(case_infidelities)
    return score

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Start with seed
    best_pulse = seed_pulse()
    best_score = compute_score(best_pulse)
    
    if time.time() > deadline - 60:
        return best_pulse
    
    # Optimization using Nelder-Mead with tighter tolerance
    def objective(params):
        pulse_arr = params.reshape(N_STEPS, 2)
        if not check_validity(pulse_arr):
            return 1e10
        return compute_score(pulse_arr)
    
    x0 = best_pulse.flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
        
        if result.fun < best_score:
            candidate = result.x.reshape(N_STEPS, 2)
            if check_validity(candidate):
                best_pulse = candidate
                best_score = result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_pulse
    
    # Random local search with adaptive step size
    np.random.seed(42)
    step_size = 0.02
    for iteration in range(200):
        if time.time() > deadline - 30:
            break
        
        # Adaptive step size
        if iteration % 20 == 0 and iteration > 0:
            step_size *= 0.8
        
        perturbation = np.random.randn(N_STEPS, 2) * step_size
        candidate = best_pulse + perturbation
        
        # Clip to amplitude bound
        for i in range(N_STEPS):
            amp = np.sqrt(candidate[i, 0]**2 + candidate[i, 1]**2)
            if amp > AMPLITUDE_BOUND:
                candidate[i] *= AMPLITUDE_BOUND / amp
        
        if check_validity(candidate):
            score = compute_score(candidate)
            if score < best_score:
                best_pulse = candidate
                best_score = score
    
    return best_pulse
# EVOLVE-BLOCK-END
