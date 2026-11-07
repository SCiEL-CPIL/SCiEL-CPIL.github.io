"""Robust X-pi pulse for a three-level transmon."""
# EVOLVE-BLOCK-START
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
TOTAL_TIME = N_STEPS * DT  # 40 ns


def _build_matrices():
    """Build 3x3 matrices for the three-level transmon."""
    # a = lowering operator for 3 levels
    # |0> = [1,0,0], |1> = [0,1,0], |2> = [0,0,1]
    # a|1> = |0>, a|2> = sqrt(2)|1>
    a = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    adag = a.conj().T
    n = adag @ a  # n|0>=0, n|1>=1, n|2>=2
    return a, adag, n


def _simulate_pulse(I, Q, detuning, scale, a, adag, n):
    """
    Simulate the pulse using Liouvillian exponentiation.
    Returns the final state (as a vector) for each initial basis state.
    """
    # Build Lindblad operators
    # L1 = a / sqrt(T1)
    # L2 = n * sqrt(2/TPHI)
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    # Precompute H for each step
    # H = detuning * n + 0.5 * anharmonicity * n * (n-1) + 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    
    # State evolution: |psi> -> U |psi>
    # We use the Lindblad master equation. For a single trajectory (pure state evolution with decoherence),
    # we can use the superoperator approach or just track the density matrix.
    # The task says "exponentiates the Liouvillian", so we work with density matrices.
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, 0, 0], dtype=complex) - np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 0, 0], dtype=complex) + 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, 0, 0], dtype=complex) - 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2)  # (|0>-i|1>)/sqrt(2)
    ]
    
    # Target: X gate, which maps |0> -> |1>, |1> -> |0>
    # X = [[0,1],[1,0]] in the 2-level subspace
    # For the 3-level system, X should map |0> <-> |1> and leave |2> unchanged (or we only care about the 2-level subspace)
    # Fidelity to X: |<X|psi_final>|^2 for each initial state
    
    # X operator for 3 levels (only acting on 0-1 subspace)
    X_op = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]  # |2> stays |2>
    ], dtype=complex)
    
    # For each initial state, compute the final density matrix
    fidelities = []
    
    for psi0 in states:
        rho = np.outer(psi0, psi0.conj())
        
        for i in range(N_STEPS):
            # Build Hamiltonian for this step
            H = (detuning * n + 
                 0.5 * ANHARMONICITY * n @ (n - np.eye(3)) + 
                 0.5 * scale * (I[i] * (a + adag) + 1j * Q[i] * (a - adag)))
            
            # Build Liouvillian
            # L(rho) = -i[H, rho] + sum_k (Lk rho Lk^dag - 0.5 {Lk^dag Lk, rho})
            # We exponentiate the superoperator
            # For efficiency, we can build the superoperator matrix
            # Or we can use the fact that for small dt, we can approximate
            # But the task says "exponentiates the Liouvillian", so let's build the superoperator
            
            # Superoperator approach:
            # vec(L(rho)) = S * vec(rho) where S is the Liouvillian superoperator
            # -i[H, rho] -> -i (H kron I - I kron H^T)
            # Lk rho Lk^dag -> (Lk^dag^T kron Lk)
            # 0.5 {Lk^dag Lk, rho} -> 0.5 (Lk^dag Lk kron I + I kron (Lk^dag Lk)^T)
            
            # This is expensive to build each step. Let's precompute what we can.
            # Actually, H depends on I[i], Q[i], so we need to build it each step.
            
            # Alternative: use the fact that dt is small and use a first-order approximation
            # rho_{i+1} = rho_i + dt * L(rho_i)
            # But the task says "exponentiates", so let's do it properly.
            
            # For a 3x3 density matrix, the superoperator is 9x9
            # Let's build it
            
            # Term 1: -i[H, rho]
            # In vectorized form: -i (H kron I - I kron H^T) vec(rho)
            H_vec = np.kron(H, np.eye(3)) - np.kron(np.eye(3), H.T)
            
            # Term 2: Dissipation
            # L1 = a/sqrt(T1), L2 = n*sqrt(2/TPHI)
            L1_dag_L1 = L1.conj().T @ L1
            L2_dag_L2 = L2.conj().T @ L2
            
            # L1 rho L1^dag: (L1^dag^T kron L1) vec(rho)
            # L2 rho L2^dag: (L2^dag^T kron L2) vec(rho)
            # 0.5 {L1^dag L1, rho}: 0.5 (L1^dag L1 kron I + I kron (L1^dag L1)^T)
            # 0.5 {L2^dag L2, rho}: 0.5 (L2^dag L2 kron I + I kron (L2^dag L2)^T)
            
            L1_term = np.kron(L1.conj().T, L1)
            L2_term = np.kron(L2.conj().T, L2)
            anti1 = 0.5 * (np.kron(L1_dag_L1, np.eye(3)) + np.kron(np.eye(3), L1_dag_L1.T))
            anti2 = 0.5 * (np.kron(L2_dag_L2, np.eye(3)) + np.kron(np.eye(3), L2_dag_L2.T))
            
            # Total Liouvillian superoperator
            S = -1j * H_vec + L1_term + L2_term - anti1 - anti2
            
            # Exponentiate
            S_exp = expm(S * DT)
            
            # Apply to vec(rho)
            rho_vec = rho.reshape(9)
            rho_vec = S_exp @ rho_vec
            rho = rho_vec.reshape(3, 3)
        
        # Compute fidelity to X
        # Fidelity = |<psi_target|psi_final>|^2 where |psi_target> = X |psi0>
        psi_target = X_op @ psi0
        # For mixed state, fidelity is Tr(rho_final * |psi_target><psi_target|)
        # Or more generally, for pure target: F = <psi_target| rho_final |psi_target>
        F = np.real(np.vdot(psi_target, rho @ psi_target))
        F = max(0, min(1, F))  # Clamp to [0,1]
        fidelities.append(F)
    
    return fidelities


def _compute_score(I, Q, a, adag, n):
    """Compute the score for a given pulse."""
    detunings = np.linspace(DE_TUNING_BOUNDS[0], DE_TUNING_BOUNDS[1], 3)
    scales = np.linspace(DRIVE_SCALE_BOUNDS[0], DRIVE_SCALE_BOUNDS[1], 3)
    
    all_fidelities = []
    
    for det in detunings:
        for scale in scales:
            fids = _simulate_pulse(I, Q, det, scale, a, adag, n)
            mean_fid = np.mean(fids)
            all_fidelities.append(1 - mean_fid)  # infidelity
    
    mean_inf = np.mean(all_fidelities)
    max_inf = np.max(all_fidelities)
    score = mean_inf + max_inf
    
    return score


def _check_validity(I, Q):
    """Check if the pulse satisfies all constraints."""
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-10:
            return False
    
    # Slew bound
    # From zero to step 1
    p0 = np.array([0, 0])
    p1 = np.array([I[0], Q[0]])
    if np.linalg.norm(p1 - p0) / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        pi = np.array([I[i], Q[i]])
        pi1 = np.array([I[i+1], Q[i+1]])
        if np.linalg.norm(pi1 - pi) / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    plast = np.array([I[-1], Q[-1]])
    if np.linalg.norm(p0 - plast) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True


def pulse(target_gate, hardware_model):
    """
    Generate a robust X-pi pulse for the three-level transmon.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    start_time = time.time()
    deadline = start_time + 300  # 300 second limit
    
    a, adag, n = _build_matrices()
    
    # Start with a DRAG-style Gaussian seed
    # sigma = 40/6 ns
    sigma = TOTAL_TIME / 6.0
    t = np.linspace(0, TOTAL_TIME, N_STEPS)
    
    # Gaussian pulse with area pi
    # I(t) = A * exp(-t^2 / (2*sigma^2))
    # Area = A * sigma * sqrt(2*pi) = pi => A = 1 / (sigma * sqrt(2))
    A = 1.0 / (sigma * np.sqrt(2))
    
    I_seed = A * np.exp(-t**2 / (2 * sigma**2))
    
    # Offset to 0 at the edges - the Gaussian is already ~0 at edges
    # But let's make sure
    I_seed[0] = 0
    I_seed[-1] = 0
    
    # DRAG term: Q(t) proportional to dI/dt
    # Q(t) = alpha * dI/dt
    dI_dt = np.gradient(I_seed, DT)
    # Choose alpha to minimize leakage
    # For a transmon, alpha ~ 1/(2*chi) where chi is anharmonicity
    # chi = ANHARMONICITY / (2*pi) = -0.25 GHz = -0.25 * 2*pi rad/ns
    # So alpha ~ 1/(2*0.25*2*pi) = 1/(pi)
    alpha = 1.0 / np.pi
    Q_seed = alpha * dI_dt
    
    # Ensure validity
    if not _check_validity(I_seed, Q_seed):
        # Scale down to satisfy constraints
        scale_factor = 1.0
        while not _check_validity(I_seed * scale_factor, Q_seed * scale_factor):
            scale_factor *= 0.9
        I_seed *= scale_factor
        Q_seed *= scale_factor
    
    I_best = I_seed.copy()
    Q_best = Q_seed.copy()
    
    # Try to improve with a simple optimization
    # Use a gradient-free method: try small perturbations
    
    # Precompute for speed
    score_best = _compute_score(I_best, Q_best, a, adag, n)
    
    # Simple local search
    step_size = 0.01
    improvement_threshold = 1e-6
    max_iterations = 500
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        # Try random perturbations
        improved = False
        for _ in range(10):
            I_trial = I_best.copy()
            Q_trial = Q_best.copy()
            
            # Random perturbation
            mask = np.random.random(N_STEPS) < 0.3
            I_trial[mask] += np.random.normal(0, step_size, mask.sum())
            Q_trial[mask] += np.random.normal(0, step_size, mask.sum())
            
            # Check validity
            if not _check_validity(I_trial, Q_trial):
                continue
            
            score_trial = _compute_score(I_trial, Q_trial, a, adag, n)
            
            if score_trial < score_best - improvement_threshold:
                I_best = I_trial
                Q_best = Q_trial
                score_best = score_trial
                improved = True
                break
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-4:
                break
    
    # Final check
    if not _check_validity(I_best, Q_best):
        # Return the seed if best is invalid
        return np.column_stack([I_seed, Q_seed])
    
    return np.column_stack([I_best, Q_best])
# EVOLVE-BLOCK-END
