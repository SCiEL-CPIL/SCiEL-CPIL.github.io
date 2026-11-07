"""Direct numerical optimization of 160 I/Q amplitudes for robust X-pi pulse."""
import numpy as np
import time
from scipy.linalg import expm

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT


def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n


def _precompute_superparts(a, adag, n):
    """Precompute static parts of the Liouvillian superoperator."""
    I3 = np.eye(3, dtype=complex)
    
    # Static Hamiltonian part: anharmonicity
    H_anh = 0.5 * ANHARMONICITY * (n @ (n - I3))
    H_anh_super = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    
    # Drive Hamiltonian parts: I*(a+adag) and Q*(a-adag)
    H_I_part = 0.5 * (a + adag)
    H_Q_part = 0.5 * (1j * (a - adag))
    H_I_super = -1j * (np.kron(H_I_part, I3) - np.kron(I3, H_I_part.T))
    H_Q_super = -1j * (np.kron(H_Q_part, I3) - np.kron(I3, H_Q_part.T))
    
    # Detuning part
    H_det_part = n
    H_det_super = -1j * (np.kron(H_det_part, I3) - np.kron(I3, H_det_part.T))
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    Ldag1 = L1.conj().T
    Ldag2 = L2.conj().T
    Ldag1_L1 = Ldag1 @ L1
    Ldag2_L2 = Ldag2 @ L2
    
    L1_super = np.kron(L1, L1.conj())
    L1_diss_super = -0.5 * (np.kron(Ldag1_L1, I3) + np.kron(I3, Ldag1_L1.T))
    
    L2_super = np.kron(L2, L2.conj())
    L2_diss_super = -0.5 * (np.kron(Ldag2_L2, I3) + np.kron(I3, Ldag2_L2.T))
    
    diss_super = L1_super + L1_diss_super + L2_super + L2_diss_super
    
    return H_anh_super, H_I_super, H_Q_super, H_det_super, diss_super


def _make_superoperator(I_val, Q_val, detuning, scale, parts):
    H_anh_super, H_I_super, H_Q_super, H_det_super, diss_super = parts
    return H_anh_super + scale * (I_val * H_I_super + Q_val * H_Q_super) + detuning * H_det_super + diss_super


def _simulate_pulse_fast(IQ, detuning, scale, parts, states_vecs, X_gate):
    """Simulate and return mean fidelity."""
    # Precompute superoperators for all steps
    supers = []
    for step in range(N_STEPS):
        I_val = IQ[step, 0]
        Q_val = IQ[step, 1]
        L_super = _make_superoperator(I_val, Q_val, detuning, scale, parts)
        supers.append(expm(L_super * DT))
    
    fidelities = []
    for rho_vec in states_vecs:
        v = rho_vec.copy()
        for step in range(N_STEPS):
            v = supers[step] @ v
        rho_final = v.reshape(3, 3)
        psi_target = X_gate @ states_vecs[0]  # placeholder, need per-state target
        # Actually need per-state target
        fidelities.append(0.0)  # placeholder
    
    # Redo properly
    fidelities = []
    for idx, psi0 in enumerate(states_vecs):
        rho = np.outer(psi0, psi0.conj())
        rho_vec = rho.flatten().copy()
        for step in range(N_STEPS):
            rho_vec = supers[step] @ rho_vec
        rho_final = rho_vec.reshape(3, 3)
        psi_target = X_gate @ psi0
        rho_target = np.outer(psi_target, psi_target.conj())
        fidelity = np.real(np.trace(rho_target @ rho_final))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)


def _compute_score(IQ, parts, states_list, X_gate):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    all_infidelities = []
    for det in detunings:
        for sc in scales:
            fid = _simulate_pulse_fast(IQ, det, sc, parts, states_list, X_gate)
            all_infidelities.append(1.0 - fid)
    
    infidelities = np.array(all_infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _check_validity(IQ):
    """Check if the pulse satisfies the constraints."""
    amps = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        diff = np.linalg.norm(IQ[i] - prev) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
        prev = IQ[i]
    
    diff = np.linalg.norm(np.array([0.0, 0.0]) - IQ[-1]) / DT
    if diff > SLEW_BOUND + 1e-12:
        return False
    
    return True


def _project_valid(IQ):
    """Project IQ to satisfy constraints approximately."""
    IQ = IQ.copy()
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(IQ[i, 0]**2 + IQ[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            IQ[i] *= scale
    
    # Enforce slew rate from zero
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        diff = np.linalg.norm(IQ[i] - prev)
        max_diff = SLEW_BOUND * DT
        if diff > max_diff:
            scale = max_diff / diff
            IQ[i] = prev + scale * (IQ[i] - prev)
        prev = IQ[i]
    
    # Enforce final slew to zero
    diff = np.linalg.norm(IQ[-1])
    max_diff = SLEW_BOUND * DT
    if diff > max_diff:
        IQ[-1] *= max_diff / diff
    
    return IQ


def _make_seed_pulse():
    """Generate a DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    
    env = np.exp(-(t - TOTAL_TIME / 2)**2 / (2 * sigma**2))
    env = env - env[0] - env[-1]
    env = np.maximum(env, 0)
    
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    
    I = env
    Q = -0.2 * np.gradient(I, DT)
    
    IQ = np.column_stack([I, Q])
    return _project_valid(IQ)


def pulse(target_gate, hardware_model):
    """
    Generate a pulse for the given target gate and hardware model.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Build matrices
    a, adag, n = _build_matrices()
    parts = _precompute_superparts(a, adag, n)
    
    # Target gate
    X_gate = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
    
    # Initial states
    states_list = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    # Create seed
    IQ_seed = _make_seed_pulse()
    
    # Evaluate seed
    best_IQ = IQ_seed.copy()
    best_score = _compute_score(IQ_seed, parts, states_list, X_gate)
    
    # Objective function for optimization
    def objective(x):
        IQ = x.reshape(N_STEPS, 2)
        if not _check_validity(IQ):
            # Penalize invalid pulses
            amps = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
            amp_violation = np.max(np.maximum(0, amps - AMPLITUDE_BOUND))
            
            slew_violation = 0.0
            prev = np.array([0.0, 0.0])
            for i in range(N_STEPS):
                diff = np.linalg.norm(IQ[i] - prev) / DT
                slew_violation = max(slew_violation, diff - SLEW_BOUND)
                prev = IQ[i]
            slew_violation = max(slew_violation, np.linalg.norm(IQ[-1]) / DT - SLEW_BOUND)
            
            penalty = 10.0 * (amp_violation + slew_violation)
            return penalty
        
        return _compute_score(IQ, parts, states_list, X_gate)
    
    # Nelder-Mead optimization on raw amplitudes
    x0 = IQ_seed.flatten()
    
    # Use a moderate initial step size
    initial_simplex = [x0.copy()]
    for i in range(len(x0)):
        xi = x0.copy()
        step = 0.01 if i % 2 == 0 else 0.01  # Small perturbation
        xi[i] += step
        initial_simplex.append(xi)
    
    try:
        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'initial_simplex': initial_simplex,
                'maxiter': 500,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
        
        if time.time() < deadline:
            IQ_candidate = result.x.reshape(N_STEPS, 2)
            if _check_validity(IQ_candidate):
                score_candidate = _compute_score(IQ_candidate, parts, states_list, X_gate)
                if score_candidate < best_score:
                    best_IQ = IQ_candidate.copy()
                    best_score = score_candidate
    except Exception:
        pass
    
    # If we have time, try a few random perturbations around best
    if time.time() < deadline - 60:
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            noise = np.random.normal(0, 0.005, (N_STEPS, 2))
            IQ_trial = best_IQ + noise
            IQ_trial = _project_valid(IQ_trial)
            if _check_validity(IQ_trial):
                score_trial = _compute_score(IQ_trial, parts, states_list, X_gate)
                if score_trial < best_score:
                    best_IQ = IQ_trial.copy()
                    best_score = score_trial
    
    # Final validity check
    if not _check_validity(best_IQ):
        best_IQ = _project_valid(best_IQ)
        if not _check_validity(best_IQ):
            best_IQ = IQ_seed
    
    return best_IQ
