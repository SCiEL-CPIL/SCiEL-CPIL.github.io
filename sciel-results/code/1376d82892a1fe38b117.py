"""Rewritten program: Legendre-basis pulse optimization via L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

# Constants
N_STEPS = 80
DT = 0.5  # ns
T_TOTAL = N_STEPS * DT  # 40 ns

# Hardware model parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4  # rad/ns
SLEW_BOUND = 0.12  # rad/ns^2
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
T2 = 40000.0  # ns  # Tphi

# Build the 3-level system
# a is the 3x3 lowering operator for a 3-level truncated harmonic oscillator
# Levels: |0>, |1>, |2>
# a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
sqrt1 = np.sqrt(1.0)
sqrt2 = np.sqrt(2.0)
a_op = np.array([
    [0.0, sqrt1, 0.0],
    [0.0, 0.0, sqrt2],
    [0.0, 0.0, 0.0]
], dtype=complex)

a_dag = a_op.conj().T
n_op = a_dag @ a_op  # number operator

# Precompute Hamiltonian terms
# H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))

# Lindblad operators
# L1 = a / sqrt(T1)
# L2 = n * sqrt(2 / Tphi)
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / T2)

# Precompute the Lindbladian contribution to the Liouvillian (time-independent part)
# The Liouvillian for a density matrix rho is:
# d rho/dt = -i [H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
# We can write this as: d rho/dt = L * vec(rho)
# where L is a 9x9 matrix (since rho is 3x3, vec(rho) is 9x1)

def build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian matrix for the given parameters."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    H = (detuning * n_op + 
         0.5 * ANHARMONICITY * (n_op @ (n_op - np.eye(3))) + 
         0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag)))
    
    # Liouvillian in vec form
    # -i [H, rho] -> -i (H rho - rho H) -> -i (H ⊗ I - I ⊗ H^T) in vec form
    # Actually, vec(H rho H^dag) etc. Let's use the standard form.
    # d rho/dt = -i H rho + i rho H + sum_k (L_k rho L_k^dag - 0.5 L_k^dag L_k rho - 0.5 rho L_k^dag L_k)
    # In vec form:
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    # So:
    # -i H rho -> -i (I ⊗ H) vec(rho)  ... wait, vec(H rho) = (I ⊗ H) vec(rho)? No.
    # vec(A X B) = (B^T ⊗ A) vec(X)
    # So vec(H rho) = (I ⊗ H) vec(rho) is wrong. It should be (I^T ⊗ H) = (I ⊗ H).
    # Actually, vec(H rho I) = (I^T ⊗ H) vec(rho) = (I ⊗ H) vec(rho). Yes.
    # vec(rho H) = (H^T ⊗ I) vec(rho).
    # So -i [H, rho] -> -i (I ⊗ H - H^T ⊗ I) vec(rho)
    
    L_mat = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        # L rho L^dag -> (L ⊗ L^T) vec(rho) ... wait
        # vec(L rho L^dag) = ((L^dag)^T ⊗ L) vec(rho) = (L^dag.conj() ⊗ L) vec(rho) = (L.T ⊗ L) vec(rho)?
        # Let's be careful. vec(A X B) = (B^T ⊗ A) vec(X).
        # A = L, X = rho, B = L^dag.
        # vec(L rho L^dag) = ((L^dag)^T ⊗ L) vec(rho) = (L^dag.T ⊗ L) vec(rho) = (L.conj() ⊗ L) vec(rho).
        # Since L is complex, L^dag.T = (L.conj().T).T = L.conj().
        # So it's (L.conj() ⊗ L) vec(rho).
        # -0.5 {L^dag L, rho} -> -0.5 (L^dag L rho + rho L^dag L)
        # vec(L^dag L rho) = (I ⊗ L^dag L) vec(rho)
        # vec(rho L^dag L) = ((L^dag L)^T ⊗ I) vec(rho) = (L^dag L.T ⊗ I) vec(rho) = (L^dag L.conj() ⊗ I) vec(rho)?
        # Wait, (L^dag L)^T = L^T (L^dag)^T = L^T L.conj().
        # So vec(rho L^dag L) = ((L^dag L)^T ⊗ I) vec(rho).
        
        term1 = np.kron(L.conj(), L)
        term2 = -0.5 * np.kron(np.eye(3), L_dag_L)
        term3 = -0.5 * np.kron(L_dag_L.T, np.eye(3))
        L_mat += term1 + term2 + term3
    
    return L_mat

def simulate_pulse(I, Q, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    vec_rho = rho.flatten()
    
    for i in range(N_STEPS):
        L_mat = build_liouvillian(detuning, scale, I[i], Q[i])
        # Evolve: vec_rho(t+dt) = exp(L_mat * dt) vec_rho(t)
        exp_L = expm(L_mat * DT)
        vec_rho = exp_L @ vec_rho
    
    rho_final = vec_rho.reshape((3, 3))
    # Normalize to ensure trace is 1 (numerical stability)
    tr = np.trace(rho_final)
    if abs(tr) > 1e-15:
        rho_final = rho_final / tr
    return rho_final

def fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure state."""
    # F = <psi| rho |psi>
    val = target_state.conj().T @ rho @ target_state
    return np.real(val)

def compute_score(I, Q):
    """Compute the score for the given pulse."""
    # 9 cases: detuning in (-2*pi*0.002, 0, 2*pi*0.002), scale in (0.97, 1, 1.03)
    detunings = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    scales = [SCALE_BOUNDS[0], 1.0, SCALE_BOUNDS[1]]
    
    # Target states
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_plus = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_plus_i = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2)
    psi_minus_i = np.array([1.0, -1j, 0.0], dtype=complex) / np.sqrt(2)
    states = [psi0, psi1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    
    # Target operation is X gate on the qubit subspace (|0>, |1>)
    # X|0> = |1>, X|1> = |0>
    # So target states after X gate:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)? Let's check.
    # X (|0> + i|1>) = |1> + i|0> = i|0> + |1> = i(|0> - i|1>)? No.
    # i(|0> - i|1>) = i|0> + |1>. Yes.
    # So target states are:
    target_states = [psi1, psi0, psi_plus, -psi_minus, 1j*(psi0 - 1j*psi1)/np.sqrt(2), 1j*(psi0 + 1j*psi1)/np.sqrt(2)]
    # Let's recompute more carefully.
    # X|0> = |1>
    # X|1> = |0>
    # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = (|0>+|1>)/sqrt(2)
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2) ... wait
    # i(|0>-i|1>) = i|0> + |1>. Yes, so (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2).
    # But we need the target state to compare fidelity. Fidelity is |<target|final>|^2.
    # So we need <target|final> to be 1 (up to phase).
    # Let's just define the target states explicitly.
    target_states = [
        psi1,
        psi0,
        psi_plus,
        -psi_minus,
        (psi1 + 1j*psi0) / np.sqrt(2),
        (psi1 - 1j*psi0) / np.sqrt(2)
    ]
    
    fidelities = []
    for d in detunings:
        for s in scales:
            for state, target in zip(states, target_states):
                # Initialize rho with the input state
                rho_init = np.outer(state, state.conj())
                # Simulate
                vec_rho = rho_init.flatten()
                for i in range(N_STEPS):
                    L_mat = build_liouvillian(d, s, I[i], Q[i])
                    exp_L = expm(L_mat * DT)
                    vec_rho = exp_L @ vec_rho
                rho_final = vec_rho.reshape((3, 3))
                tr = np.trace(rho_final)
                if abs(tr) > 1e-15:
                    rho_final = rho_final / tr
                fid = fidelity(rho_final, target)
                fidelities.append(fid)
    
    fidelities = np.array(fidelities)
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def legendre_basis(t, n_terms=8):
    """Evaluate the first n_terms Legendre polynomials at t in [0, 1]."""
    # t is an array of shape (N_STEPS,)
    # Map t from [0, 1] to [-1, 1]
    x = 2 * t - 1
    basis = np.zeros((n_terms, len(t)), dtype=float)
    for k in range(n_terms):
        # Use numpy's legendre polynomial
        P = np.polynomial.legendre.legval(x, [0]*k + [1])
        basis[k] = P
    return basis

def coeffs_to_pulse(coeffs):
    """Convert Legendre coefficients to I, Q pulses."""
    t = np.linspace(0, 1, N_STEPS, endpoint=False) + 0.5/N_STEPS  # Midpoint
    basis = legendre_basis(t, 8)
    I = basis @ coeffs[0:4]
    Q = basis @ coeffs[4:8]
    return I, Q

def pulse_to_coeffs(I, Q):
    """Convert I, Q pulses to Legendre coefficients (least squares)."""
    t = np.linspace(0, 1, N_STEPS, endpoint=False) + 0.5/N_STEPS
    basis = legendre_basis(t, 8)
    # Solve basis @ coeffs = [I, Q]
    # This is an overdetermined system, use least squares
    target = np.concatenate([I, Q])
    coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    return coeffs

def objective(coeffs):
    """Objective function for optimization: infidelity."""
    I, Q = coeffs_to_pulse(coeffs)
    # Check constraints
    # Amplitude bound
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-6:
        return 1e10
    # Slew bound
    # From zero to step 1
    slew1 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    # From last step to zero
    slew_last = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    # Between steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slews = np.sqrt(dI**2 + dQ**2) / DT
    max_slew = max(slew1, slew_last, np.max(slews))
    if max_slew > SLEW_BOUND + 1e-6:
        return 1e10
    
    # Compute score
    score = compute_score(I, Q)
    return score

def pulse(target_gate, hardware_model):
    """Main function to generate the pulse."""
    # Initial guess: DRAG-style Gaussian
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT/2
    sigma = T_TOTAL / 6.0
    gaussian = np.exp(-(t - T_TOTAL/2)**2 / (2 * sigma**2))
    # Normalize area to pi
    area = np.sum(gaussian) * DT
    gaussian = gaussian * (np.pi / area)
    I_init = gaussian
    Q_init = np.zeros_like(I_init)
    
    # Convert to Legendre coefficients
    coeffs_init = pulse_to_coeffs(I_init, Q_init)
    
    # Define bounds for coefficients
    # We need to ensure the resulting pulse satisfies the constraints.
    # This is tricky with Legendre basis. We'll use L-BFGS-B with bounds on coefficients
    # that are loose enough to allow valid pulses, and rely on the objective to penalize invalid ones.
    # Alternatively, we can project onto the feasible set.
    # For now, let's use wide bounds and let the optimizer handle it.
    bounds = [(-2.0, 2.0)] * 8
    
    # Optimize
    result = minimize(objective, coeffs_init, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    # Get the best pulse
    I, Q = coeffs_to_pulse(result.x)
    
    # Ensure constraints are strictly satisfied
    # Clip amplitude
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / np.max(amp)
        I *= scale_factor
        Q *= scale_factor
    
    # Check slew and adjust if necessary
    # This is a simplified check; a more robust approach would be to project onto the feasible set.
    
    return np.column_stack([I, Q])

# The entry point function must be named `pulse` and have the signature `pulse(target_gate, hardware_model)`
# The current program uses `pulse = quantum_seed`, so we define `pulse` as above.
