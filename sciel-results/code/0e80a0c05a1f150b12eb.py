"""Rewritten program: DRAG pulse optimization for three-level transmon X-pi gate."""
import numpy as np
import time

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Precompute basis operators for 3-level system
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op

# Liouvillian components
# Lindblad operators: L1 = a/sqrt(T1), L2 = n*sqrt(2/Tphi)
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)

# Precompute commutator and dissipative superoperators for speed
def _build_hamiltonian(detuning, scale, I, Q):
    """Build Hamiltonian for given parameters."""
    h_det = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    h_drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
    return h_det + h_drive

def _liouvillian_step(h, dt):
    """Compute the Liouvillian superoperator for one step."""
    d = 3
    # Coherent part: -i*[H, rho] -> -i*(H@rho - rho@H)
    # Dissipative part: sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # We'll build the full superoperator matrix
    
    # Coherent: S_c = -1j * (H kron I - I kron H^T) acting on vec(rho)
    # Actually, let's use the standard form:
    # d(rho)/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 L_k^dag L_k rho - 0.5 rho L_k^dag L_k)
    
    # For numerical stability and speed, let's compute the evolution operator
    # by exponentiating the Liouvillian matrix.
    
    # Build Liouvillian matrix L such that d(vec(rho))/dt = L @ vec(rho)
    # vec(rho) stacks columns of rho
    
    L_mat = np.zeros((d*d, d*d), dtype=complex)
    
    # Coherent part: -i(H ⊗ I - I ⊗ H^T)
    L_mat += -1j * (np.kron(h, np.eye(d)) - np.kron(np.eye(d), h.T))
    
    # Dissipative part for each Lindblad operator
    for L_k in [L1, L2]:
        Lk_dag = L_k.conj().T
        Lk_dag_Lk = Lk_dag @ L_k
        # L_k rho L_k^dag -> kron(L_k, L_k^T) vec(rho)
        L_mat += np.kron(L_k, L_k.T)
        # -0.5 L_k^dag L_k rho -> -0.5 kron(I, L_k^dag L_k) vec(rho)
        L_mat += -0.5 * np.kron(np.eye(d), Lk_dag_Lk)
        # -0.5 rho L_k^dag L_k -> -0.5 kron((L_k^dag L_k)^T, I) vec(rho)
        L_mat += -0.5 * np.kron(Lk_dag_Lk.T, np.eye(d))
    
    # Exponentiate: expm(L * dt)
    from scipy.linalg import expm
    return expm(L_mat * dt)

def simulate_pulse(I, Q, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = rho.flatten()
    
    for i in range(N_STEPS):
        h = _build_hamiltonian(detuning, scale, I[i], Q[i])
        evo = _liouvillian_step(h, DT)
        rho_vec = evo @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho

def compute_fidelity(rho, target_state):
    """Compute fidelity between density matrix and target state."""
    # Fidelity = <psi|rho|psi> for pure state target
    psi = target_state.flatten()
    fid = np.real(np.conj(psi) @ rho @ psi)
    return max(0, min(1, fid))

def compute_score(I, Q):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    target_states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # X gate: |0> -> |1>, |1> -> |0>
    # So target for |0> is |1>, for |1> is |0>, etc.
    target_X = {
        0: np.array([0, 1, 0], dtype=complex),  # |0> -> |1>
        1: np.array([1, 0, 0], dtype=complex),  # |1> -> |0>
        2: (np.array([0, 1, 0], dtype=complex) + np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
        3: (np.array([0, 1, 0], dtype=complex) - np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
        4: (np.array([0, 1, 0], dtype=complex) + 1j * np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
        5: (np.array([0, 1, 0], dtype=complex) - 1j * np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
    }
    
    infidelities = []
    for det in detunings:
        for scl in scales:
            rho = simulate_pulse(I, Q, det, scl)
            fids = []
            for i, ts in enumerate(target_states):
                tgt = target_X[i]
                fid = compute_fidelity(rho, tgt)
                fids.append(fid)
            mean_fid = np.mean(fids)
            infidelities.append(1 - mean_fid)
    
    inf_arr = np.array(infidelities)
    return np.mean(inf_arr) + np.max(inf_arr)

def check_validity(I, Q):
    """Check if the pulse satisfies amplitude and slew constraints."""
    # Amplitude bound
    amps = np.sqrt(I**2 + Q**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero into step 1
    p0 = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        p_i = np.array([I[i], Q[i]])
        if i == 0:
            diff = np.linalg.norm(p_i - p0)
        else:
            p_prev = np.array([I[i-1], Q[i-1]])
            diff = np.linalg.norm(p_i - p_prev)
        if diff / DT > SLEW_BOUND + 1e-12:
            return False
    
    # From last step to zero
    p_last = np.array([I[-1], Q[-1]])
    if np.linalg.norm(p_last) / DT > SLEW_BOUND + 1e-12:
        return False
    
    return True

def make_drag_pulse(sigma, amplitude, drag_coeff):
    """Create a DRAG pulse with given parameters."""
    t = np.arange(N_STEPS) * DT
    # Gaussian envelope
    envelope = np.exp(-t**2 / (2 * sigma**2))
    # Normalize to have area pi
    area = np.sum(envelope) * DT
    envelope = envelope * (np.pi / area)
    
    I = amplitude * envelope
    # DRAG: Q is proportional to derivative of I
    dI_dt = np.gradient(I, DT)
    Q = drag_coeff * dI_dt
    
    return I, Q

def optimize():
    """Optimize the DRAG pulse parameters."""
    start_time = time.time()
    deadline = start_time + 320  # 10s safety margin
    
    # Initial guess
    best_I, best_Q = make_drag_pulse(40/6, 0.35, 0.1)
    best_score = compute_score(best_I, best_Q)
    
    # Try a few parameter combinations
    for sigma in [40/6, 5, 6, 7]:
        for amp in [0.30, 0.35, 0.40]:
            for drag in [-0.2, -0.1, 0, 0.1, 0.2]:
                if time.time() > deadline:
                    break
                I, Q = make_drag_pulse(sigma, amp, drag)
                if check_validity(I, Q):
                    score = compute_score(I, Q)
                    if score < best_score:
                        best_score = score
                        best_I, best_Q = I, Q
    
    return best_I, best_Q

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 320
    
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate supported")
    
    # Optimize the pulse
    I, Q = optimize()
    
    # Ensure validity
    if not check_validity(I, Q):
        # Fallback to a simple valid pulse
        t = np.arange(N_STEPS) * DT
        envelope = np.exp(-t**2 / (2 * (40/6)**2))
        area = np.sum(envelope) * DT
        envelope = envelope * (np.pi / area)
        I = 0.35 * envelope
        Q = np.zeros(N_STEPS)
        # Adjust to satisfy slew constraint
        for i in range(1, N_STEPS):
            if abs(I[i] - I[i-1]) / DT > SLEW_BOUND:
                I[i] = I[i-1] + SLEW_BOUND * DT
        if abs(I[0]) / DT > SLEW_BOUND:
            I[0] = SLEW_BOUND * DT
        if abs(I[-1]) / DT > SLEW_BOUND:
            I[-1] = SLEW_BOUND * DT
    
    return np.column_stack([I, Q])
