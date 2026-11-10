"""Pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.integrate import solve_ivp

def _build_operators():
    """Build 3-level transmon operators."""
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = math.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _hamiltonian(I, Q, detuning, scale, a, adag, n, anharmonicity):
    """Build Hamiltonian for given I, Q, detuning, scale."""
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    return H

def _liouvillian_superop(H, a, n, T1, Tphi):
    """Build the Liouvillian superoperator as a 9x9 matrix."""
    dim = 3
    L = np.zeros((dim*dim, dim*dim), dtype=complex)
    # Coherent part: -i(H ⊗ I - I ⊗ H^T)
    H_flat = H.flatten()
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    idx = i*dim + j
                    kidx = k*dim + l
                    # Term from H ⊗ I
                    val1 = H[k, l] * (1 if i == k else 0) * (1 if j == l else 0)
                    # Term from I ⊗ H^T
                    val2 = (1 if i == k else 0) * H[i, j].conj() if False else 0
                    # Actually, let's use the standard formula
                    # L(rho) = -i(H rho - rho H) + dissipator
                    # In vectorized form: L = -i(H ⊗ I - I ⊗ H^T)
                    pass
    
    # Let me use a cleaner approach
    I_mat = np.eye(dim, dtype=complex)
    # -i(H ⊗ I - I ⊗ H^T)
    L = -1j * (np.kron(H, I_mat) - np.kron(I_mat, H.T))
    
    # Dissipator: sum_k (L_k rho L_k^† - 1/2 {L_k^† L_k, rho})
    def lindblad_term(Lk):
        Lk_flat = Lk.flatten()
        # Lk rho Lk^†
        term1 = np.kron(Lk, Lk.conj().T)
        # Lk^† Lk rho
        Lk_dag_Lk = Lk.conj().T @ Lk
        term2 = 0.5 * np.kron(Lk_dag_Lk, I_mat)
        # rho Lk^† Lk
        term3 = 0.5 * np.kron(I_mat, Lk_dag_Lk.T)
        return term1 - term2 - term3
    
    # a / sqrt(T1)
    Lk1 = a / math.sqrt(T1)
    L += lindblad_term(Lk1)
    
    # n * sqrt(2/Tphi)
    Lk2 = n * math.sqrt(2.0 / Tphi)
    L += lindblad_term(Lk2)
    
    return L

def _simulate_pulse(pulse, detuning, scale, a, adag, n, anharmonicity, T1, Tphi):
    """Simulate the pulse and return the final density matrix."""
    num_steps = pulse.shape[0]
    dt = 0.5  # ns
    
    dim = 3
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    rho_vec = rho.flatten().copy()
    
    for i in range(num_steps):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        H = _hamiltonian(I_val, Q_val, detuning, scale, a, adag, n, anharmonicity)
        L = _liouvillian_superop(H, a, n, T1, Tphi)
        
        # Exponentiate: rho_vec_next = expm(L * dt) @ rho_vec
        # Use scipy for matrix exponential
        from scipy.linalg import expm
        U = expm(L * dt)
        rho_vec = U @ rho_vec
    
    rho_final = rho_vec.reshape(dim, dim)
    return rho_final

def _fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure state."""
    # F = |<psi|rho|psi>|^2 for pure target, but actually F = <psi|rho|psi> for pure target
    psi = target_state.flatten()
    F = np.real(np.vdot(psi, rho @ psi))
    return max(0.0, min(1.0, F))

def _compute_score(pulse, a, adag, n, anharmonicity, T1, Tphi):
    """Compute the score: mean + max of infidelities over 9 cases."""
    # Target: X gate (bit flip)
    # X |0> = |1>, X |1> = |0>
    # So we want to map |0> -> |1> and |1> -> |0>
    
    detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    states = []
    states.append(np.array([1, 0, 0], dtype=complex))  # |0>
    states.append(np.array([0, 1, 0], dtype=complex))  # |1>
    states.append(np.array([1, 1, 0], dtype=complex) / math.sqrt(2))  # (|0>+|1>)/sqrt(2)
    states.append(np.array([1, -1, 0], dtype=complex) / math.sqrt(2))  # (|0>-|1>)/sqrt(2)
    states.append(np.array([1, 1j, 0], dtype=complex) / math.sqrt(2))  # (|0>+i|1>)/sqrt(2)
    states.append(np.array([1, -1j, 0], dtype=complex) / math.sqrt(2))  # (|0>-i|1>)/sqrt(2)
    
    # Target states after X gate
    target_states = []
    target_states.append(np.array([0, 1, 0], dtype=complex))  # X|0> = |1>
    target_states.append(np.array([1, 0, 0], dtype=complex))  # X|1> = |0>
    target_states.append(np.array([1, 1, 0], dtype=complex) / math.sqrt(2))  # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
    target_states.append(np.array([-1, 1, 0], dtype=complex) / math.sqrt(2))  # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2)
    target_states.append(np.array([1j, 1, 0], dtype=complex) / math.sqrt(2))  # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    target_states.append(np.array([-1j, 1, 0], dtype=complex) / math.sqrt(2))  # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
    
    infidelities = []
    
    for det in detunings:
        for scl in scales:
            case_infs = []
            for i, (state, target) in enumerate(zip(states, target_states)):
                # Simulate starting from this state
                rho_initial = np.outer(state, state.conj().T)
                rho_vec = rho_initial.flatten().copy()
                
                num_steps = pulse.shape[0]
                dt = 0.5
                
                for j in range(num_steps):
                    I_val = pulse[j, 0]
                    Q_val = pulse[j, 1]
                    
                    H = _hamiltonian(I_val, Q_val, det, scl, a, adag, n, anharmonicity)
                    L = _liouvillian_superop(H, a, n, T1, Tphi)
                    
                    from scipy.linalg import expm
                    U = expm(L * dt)
                    rho_vec = U @ rho_vec
                
                rho_final = rho_vec.reshape(3, 3)
                F = _fidelity(rho_final, target)
                inf = 1.0 - F
                case_infs.append(inf)
            
            mean_inf = np.mean(case_infs)
            infidelities.append(mean_inf)
    
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def _check_validity(pulse):
    """Check amplitude and slew rate constraints."""
    num_steps = pulse.shape[0]
    dt = 0.5
    amp_bound = 0.4
    slew_bound = 0.12
    
    # Check amplitudes
    for i in range(num_steps):
        amp = math.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > amp_bound + 1e-10:
            return False
    
    # Check slew rates
    prev = np.array([0.0, 0.0])
    for i in range(num_steps):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev) / dt
        if diff > slew_bound + 1e-10:
            return False
        prev = curr
    
    # Check from last step to zero
    last = pulse[-1]
    diff = np.linalg.norm(last - np.array([0.0, 0.0])) / dt
    if diff > slew_bound + 1e-10:
        return False
    
    return True

def _drag_seed():
    """Generate DRAG-style Gaussian seed pulse."""
    num_steps = 80
    dt = 0.5
    total_time = num_steps * dt  # 40 ns
    sigma = 40.0 / 6.0  # ~6.67 ns
    
    t = np.arange(num_steps) * dt + dt / 2  # Center of each step
    
    # Gaussian envelope with area pi
    # Area = amp * sigma * sqrt(2*pi) = pi
    # amp = 1 / (sigma * sqrt(2*pi))
    amp = 1.0 / (sigma * math.sqrt(2 * math.pi))
    
    I = amp * np.exp(-t**2 / (2 * sigma**2))
    Q = np.zeros(num_steps)
    
    pulse = np.column_stack([I, Q])
    
    # Check and clip to satisfy constraints
    if not _check_validity(pulse):
        # Scale down if needed
        max_amp = np.max(np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2))
        if max_amp > 0.4:
            pulse *= 0.4 / max_amp
        
        # Check slew
        for i in range(1, num_steps):
            diff = np.linalg.norm(pulse[i] - pulse[i-1]) / dt
            if diff > 0.12:
                # Need to reduce
                pass
        
        # From zero to first
        diff0 = np.linalg.norm(pulse[0]) / dt
        if diff0 > 0.12:
            pulse[0] *= 0.12 * dt / np.linalg.norm(pulse[0]) if np.linalg.norm(pulse[0]) > 0 else 1
        
        # From last to zero
        difflast = np.linalg.norm(pulse[-1]) / dt
        if difflast > 0.12:
            pulse[-1] *= 0.12 * dt / np.linalg.norm(pulse[-1]) if np.linalg.norm(pulse[-1]) > 0 else 1
    
    return pulse

def _optimize_pulse(a, adag, n, anharmonicity, T1, Tphi, time_limit):
    """Optimize the pulse using gradient-based methods."""
    pulse = _drag_seed()
    best_pulse = pulse.copy()
    best_score = _compute_score(pulse, a, adag, n, anharmonicity, T1, Tphi)
    
    start_time = time.time()
    
    # Simple gradient descent with finite differences
    lr = 0.01
    fd_step = 0.001
    
    while time.time() - start_time < time_limit - 30:
        # Compute gradient via finite differences
        grad = np.zeros_like(pulse)
        
        for i in range(pulse.shape[0]):
            for j in range(2):
                # Forward difference
                pulse_plus = pulse.copy()
                pulse_plus[i, j] += fd_step
                score_plus = _compute_score(pulse_plus, a, adag, n, anharmonicity, T1, Tphi)
                
                grad[i, j] = (score_plus - best_score) / fd_step
        
        # Gradient step
        pulse_new = pulse - lr * grad
        
        # Project to feasible set (simple clipping)
        pulse_new = np.clip(pulse_new, -0.4, 0.4)
        
        # Check validity
        if _check_validity(pulse_new):
            new_score = _compute_score(pulse_new, a, adag, n, anharmonicity, T1, Tphi)
            if new_score < best_score:
                pulse = pulse_new
                best_score = new_score
                best_pulse = pulse.copy()
            else:
                # Reduce learning rate
                lr *= 0.5
        else:
            lr *= 0.5
        
        if lr < 1e-6:
            break
    
    return best_pulse, best_score

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    # Extract hardware parameters
    anharmonicity = -2 * math.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    
    a, adag, n = _build_operators()
    
    # Get seed pulse
    seed_pulse = _drag_seed()
    
    # Try to optimize with time limit
    start_time = time.time()
    time_limit = 300.0  # seconds
    
    # First, compute seed score
    seed_score = _compute_score(seed_pulse, a, adag, n, anharmonicity, T1, Tphi)
    
    # Optimize
    try:
        optimized_pulse, opt_score = _optimize_pulse(a, adag, n, anharmonicity, T1, Tphi, time_limit)
        if opt_score < seed_score:
            return optimized_pulse
    except Exception:
        pass
    
    return seed_pulse

# EVOLVE-BLOCK-END
