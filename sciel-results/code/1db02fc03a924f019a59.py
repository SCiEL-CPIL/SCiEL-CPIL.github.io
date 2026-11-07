"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    n = a_dag @ a  # diag(0, 1, 2)
    return a, a_dag, n

def _liouvillian(I, Q, detuning, scale, a, a_dag, n):
    """Build the Liouvillian superoperator for one step."""
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

def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n, input_states, target_states):
    """Simulate the pulse for all input states in parallel."""
    # Stack initial density matrices
    # input_states: (6, 3)
    # rho_k = |psi_k><psi_k|
    # vec(rho_k) is 9x1
    # Stack into (9, 6)
    rhos_vec = np.zeros((9, 6), dtype=complex)
    for k in range(6):
        rho = np.outer(input_states[k], input_states[k].conj())
        rhos_vec[:, k] = rho.flatten(order='F')
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        L_super = _liouvillian(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        
        # Apply U_super to all states: (9,9) @ (9,6) -> (9,6)
        rhos_vec = U_super @ rhos_vec
    
    # Compute fidelities
    fidelities = []
    for k in range(6):
        rho = rhos_vec[:, k].reshape((3, 3), order='F')
        psi_target = target_states[k]
        fidelity = np.real(np.conj(psi_target) @ rho @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    # Slew rate check
    # Add zero padding to check boundaries
    p_ext = np.vstack([np.zeros((1, 2)), pulse, np.zeros((1, 2))])
    diffs = np.linalg.norm(np.diff(p_ext, axis=0), axis=1)
    slews = diffs / DT
    if np.any(slews > SLEW_BOUND + 1e-12):
        return False
    
    return True

def _compute_score(pulse, a, a_dag, n, input_states, target_states):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n, input_states, target_states)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _generate_pulse(params):
    """Generate a DRAG-style pulse from parameters [A, beta, sigma]."""
    A, beta, sigma = params
    
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    # Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Normalize area to 1, then scale by A to control amplitude
    # Actually, we want to optimize A.
    # Let's normalize the shape to have max 1, then scale by A?
    # Or normalize area to 1?
    # If we normalize area to 1, then A is the "area" parameter.
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = np.zeros_like(gauss)
    
    # Scale by A (A is the area of the pulse)
    I = A * gauss_norm
    
    # Derivative for DRAG
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    # Normalize derivative similarly?
    # DRAG term is usually proportional to derivative of the envelope.
    # Let's use the derivative of the normalized shape.
    Q = A * beta * dgauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound by scaling down if necessary
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        # Note: This changes the effective A. 
        # For optimization, it's better to let the optimizer handle bounds, 
        # but hard clipping ensures validity.
        
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0
    
    a, a_dag, n = _build_matrices()
    
    # Define input and target states
    input_states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)
    
    target_states = np.array([
        [0, 1, 0], # X|0> = |1>
        [1, 0, 0], # X|1> = |0>
        [1, 1, 0] / np.sqrt(2), # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
        [1, -1, 0] / np.sqrt(2), # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2)
        [1j, 1, 0] / np.sqrt(2), # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
        [-1j, 1, 0] / np.sqrt(2) # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    ], dtype=complex)
    # Note: Fidelity is |<psi_t|psi_out>|^2, so global phase in target doesn't matter.
    # The target states above are just the X-gate applied to inputs.
    
    # Initial guess: Seed-like
    # Seed: sigma = 40/6 ~ 6.67, Area = pi ~ 3.14
    # A is the area.
    init_params = [np.pi, 0.0, 40.0/6.0]
    
    best_pulse = _generate_pulse(init_params)
    best_score = _compute_score(best_pulse, a, a_dag, n, input_states, target_states)
    
    # Optimization function
    def objective(params):
        p = _generate_pulse(params)
        if not _check_validity(p):
            return 1e6 # Penalty
        return _compute_score(p, a, a_dag, n, input_states, target_states)
    
    # Nelder-Mead
    bounds = [(0.1, 1.0), (-1.0, 1.0), (2.0, 10.0)] # A, beta, sigma
    
    result = minimize(objective, init_params, method='Nelder-Mead', bounds=bounds, 
                      options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if result.fun < best_score:
        best_pulse = _generate_pulse(result.x)
        best_score = result.fun
    
    # Fine-tuning with smaller steps if time permits
    if time.time() < deadline - 60:
        # Try a few more points around the best
        rng = np.random.RandomState(42)
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            perturb = rng.normal(0, 0.05, size=3)
            new_params = result.x + perturb
            # Clip to bounds
            new_params[0] = np.clip(new_params[0], 0.1, 1.0)
            new_params[1] = np.clip(new_params[1], -1.0, 1.0)
            new_params[2] = np.clip(new_params[2], 2.0, 10.0)
            
            p = _generate_pulse(new_params)
            if not _check_validity(p):
                continue
            score = _compute_score(p, a, a_dag, n, input_states, target_states)
            if score < best_score:
                best_score = score
                best_pulse = p.copy()
    
    return best_pulse
# EVOLVE-BLOCK-END
