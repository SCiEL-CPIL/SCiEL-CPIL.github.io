"""Rewritten program: Legendre basis expansion with L-BFGS-B optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Constants from the task
DT = 0.5  # ns
N_STEPS = 80
TOTAL_TIME = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4  # rad/ns
SLEW_BOUND = 0.12  # rad/ns^2
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns

# Precompute time grid
t_grid = np.arange(N_STEPS + 1) * DT  # 0 to 40 ns, 81 points
t_mid = t_grid[:-1] + DT / 2  # midpoints for step evaluation

# Legendre polynomials on [0, 1]
from numpy.polynomial import legendre as L
# First 8 Legendre polynomials evaluated at scaled time
def legendre_basis(t):
    """Return (N, 8) matrix of first 8 Legendre polynomials scaled to [0,1]."""
    n = len(t)
    basis = np.zeros((n, 8))
    # Legendre polynomials P_k(x) for k=0..7, x in [-1, 1]
    # Map t from [0, 40] to x from [-1, 1]: x = 2*t/40 - 1 = t/20 - 1
    x = t / 20.0 - 1.0
    for k in range(8):
        # Evaluate Legendre polynomial of degree k
        # Use numpy's legendre module
        Pk = L.legvander(x, k)[:, k]
        basis[:, k] = Pk
    return basis

# Precompute basis at midpoints and at edges (for slew constraints)
B_mid = legendre_basis(t_mid)  # (80, 8)
B_all = legendre_basis(t_grid)  # (81, 8)

def coeffs_to_pulse(c):
    """Convert 8 Legendre coefficients to (80, 2) pulse array.
    c = [I0, I1, ..., I7, Q0, Q1, ..., Q7]
    """
    c = np.asarray(c, dtype=float)
    Ic = c[:8]
    Qc = c[8:]
    I = B_mid @ Ic  # (80,)
    Q = B_mid @ Qc  # (80,)
    return np.column_stack([I, Q])

def pulse_from_coeffs(c):
    """Return (80, 2) array."""
    return coeffs_to_pulse(c)

def check_constraints(pulse):
    """Check amplitude and slew constraints. Returns (valid, max_violation)."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amp = np.sqrt(I**2 + Q**2)
    amp_viol = np.max(np.maximum(0, amp - AMPLITUDE_BOUND))
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    # p[0] is first step, so we need:
    # ||p[0] - 0|| / dt <= 0.12
    # ||p[i+1] - p[i]|| / dt <= 0.12 for i=0..78
    # ||0 - p[79]|| / dt <= 0.12
    slew_viol = 0.0
    # From zero to first step
    d0 = np.linalg.norm(pulse[0]) / DT
    slew_viol = max(slew_viol, max(0, d0 - SLEW_BOUND))
    # Between steps
    for i in range(N_STEPS - 1):
        d = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        slew_viol = max(slew_viol, max(0, d - SLEW_BOUND))
    # From last step to zero
    dlast = np.linalg.norm(pulse[-1]) / DT
    slew_viol = max(slew_viol, max(0, dlast - SLEW_BOUND))
    
    valid = (amp_viol < 1e-10) and (slew_viol < 1e-10)
    return valid, max(amp_viol, slew_viol)

def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian matrix.
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a + a^dag) + i*Q*(a - a^dag))
    """
    # a = diag(sqrt([1, 2]), 1) means:
    # a[0,1] = sqrt(1) = 1, a[1,2] = sqrt(2), rest 0
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    
    # n = a^dag a
    n = a_dag @ a  # diag(0, 1, 2)
    
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a + a^dag) + i*Q*(a - a^dag))
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3)))
    drive = I * (a + a_dag) + 1j * Q * (a - a_dag)
    H += 0.5 * scale * drive
    return H

def build_liouvillian(H):
    """Build the Liouvillian superoperator for the 3-level system.
    Lindblad operators: L1 = a / sqrt(T1), L2 = n * sqrt(2/Tphi)
    """
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a  # diag(0, 1, 2)
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Liouvillian: dρ/dt = -i[H, ρ] + Σ (L ρ L† - 0.5{L†L, ρ})
    # In vectorized form, we can build the superoperator
    # But for efficiency, we'll use the matrix form directly in the ODE solver
    
    return H, L1, L2

def fidelity_step(pulse, detuning, scale):
    """Compute fidelity for a single (detuning, scale) case.
    Returns mean fidelity over the 5 test states.
    """
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Initial state: |0> = [1, 0, 0]^T
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    # Evolve step by step using exponentiation of Liouvillian
    for i in range(N_STEPS):
        H = build_hamiltonian(detuning, scale, I[i], Q[i])
        a = np.zeros((3, 3), dtype=complex)
        a[0, 1] = 1.0
        a[1, 2] = np.sqrt(2.0)
        a_dag = a.conj().T
        n = a_dag @ a
        L1 = a / np.sqrt(T1)
        L2 = n * np.sqrt(2.0 / Tphi)
        
        # Liouvillian superoperator in vectorized form
        # dvec(ρ)/dt = -i(H⊗I - I⊗H^T)vec(ρ) + ... 
        # Actually, let's use the standard form:
        # L(ρ) = -i[H, ρ] + Σ_k (L_k ρ L_k† - 0.5 L_k†L_k ρ - 0.5 ρ L_k†L_k)
        
        # Build the superoperator matrix S such that vec(dρ/dt) = S vec(ρ)
        # S = -i(H⊗I - I⊗H^T) + Σ_k (L_k⊗L_k* - 0.5(L_k†L_k)⊗I - 0.5 I⊗(L_k†L_k)^T)
        
        I3 = np.eye(3, dtype=complex)
        S = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        for Lk in [L1, L2]:
            Lk_dag = Lk.conj().T
            Lk_dag_Lk = Lk_dag @ Lk
            S += np.kron(Lk, Lk.conj()) - 0.5 * np.kron(Lk_dag_Lk, I3) - 0.5 * np.kron(I3, Lk_dag_Lk.T)
        
        # Exponentiate S * dt
        # Use scipy.linalg.expm
        from scipy.linalg import expm
        Sdt = S * DT
        U = expm(Sdt)
        rho_vec = rho.flatten()
        rho_vec_new = U @ rho_vec
        rho = rho_vec_new.reshape(3, 3)
    
    # Target: X gate on the 2-level subspace {|0>, |1>}
    # X|0> = |1>, X|1> = |0>, X|0>+|1>)/sqrt(2) = (|0>+|1>)/sqrt(2), etc.
    # Target unitary on {|0>, |1>} subspace: X = [[0, 1], [1, 0]]
    # In the 3-level system, we only care about the 2x2 block
    # Fidelity = <ψ| U† X U |ψ> for each test state
    
    # Extract 2x2 block of rho (indices 0 and 1)
    rho_2d = rho[np.ix_([0, 1], [0, 1])]
    
    # Target states and their X images
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    # Fidelity for state |ψ>: F = <ψ| U† X U |ψ> = Tr(|ψ><ψ| U† X U)
    # But we need to be careful: the target is X, so we want U to implement X
    # Fidelity = |<X|ψ>† U |ψ>|^2 ... actually for pure states:
    # F = |<ψ_target| U |ψ_initial>|^2 where |ψ_target> = X|ψ_initial>
    
    test_states = [
        np.array([1, 0], dtype=complex),  # |0>
        np.array([0, 1], dtype=complex),  # |1>
        np.array([1, 1], dtype=complex) / np.sqrt(2),  # |+>
        np.array([1, -1], dtype=complex) / np.sqrt(2),  # |->
        np.array([1, 1j], dtype=complex) / np.sqrt(2),  # |+i>
        np.array([1, -1j], dtype=complex) / np.sqrt(2),  # |-i>
    ]
    
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    
    fidelities = []
    for psi in test_states:
        psi_full = np.zeros(3, dtype=complex)
        psi_full[:2] = psi
        psi_target_full = np.zeros(3, dtype=complex)
        psi_target_full[:2] = X_mat @ psi
        
        # U|ψ> is the final state. We have rho = U|0><0|U† but we started from |0>
        # We need to evolve each test state separately!
        
    # Actually, we need to evolve each test state. Let's redo this properly.
    fidelities = []
    for psi in test_states:
        psi_full = np.zeros(3, dtype=complex)
        psi_full[:2] = psi
        rho = np.outer(psi_full, psi_full.conj())
        
        for i in range(N_STEPS):
            H = build_hamiltonian(detuning, scale, I[i], Q[i])
            a = np.zeros((3, 3), dtype=complex)
            a[0, 1] = 1.0
            a[1, 2] = np.sqrt(2.0)
            a_dag = a.conj().T
            n = a_dag @ a
            L1 = a / np.sqrt(T1)
            L2 = n * np.sqrt(2.0 / Tphi)
            
            I3 = np.eye(3, dtype=complex)
            S = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
            for Lk in [L1, L2]:
                Lk_dag = Lk.conj().T
                Lk_dag_Lk = Lk_dag @ Lk
                S += np.kron(Lk, Lk.conj()) - 0.5 * np.kron(Lk_dag_Lk, I3) - 0.5 * np.kron(I3, Lk_dag_Lk.T)
            
            from scipy.linalg import expm
            Sdt = S * DT
            U = expm(Sdt)
            rho_vec = rho.flatten()
            rho_vec_new = U @ rho_vec
            rho = rho_vec_new.reshape(3, 3)
        
        psi_target_full = np.zeros(3, dtype=complex)
        psi_target_full[:2] = X_mat @ psi
        
        # Fidelity = |<ψ_target|ψ_final>|^2
        # ψ_final is the state such that rho = |ψ_final><ψ_final|
        # For pure state evolution, we can extract the state vector
        # rho[0,0] + rho[1,1] should be ~1 for the 2-level subspace
        # Extract state vector from rho
        # Since rho is pure (starting from pure state with no decoherence in the 2-level subspace approximately),
        # we can get the state vector by taking the first row/column
        
        # Better: compute <ψ_target|ρ|ψ_target>
        psi_t = psi_target_full
        fid = np.real(np.vdot(psi_t, rho @ psi_t))
        fid = max(0, min(1, fid))
        fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    return mean_fid

def score_pulse(pulse):
    """Compute the score: mean + max of infidelity over 9 cases."""
    detunings = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    scales = [SCALE_BOUNDS[0], 1.0, SCALE_BOUNDS[1]]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fid = fidelity_step(pulse, det, sc)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def objective(c):
    """Objective function for optimization."""
    pulse = coeffs_to_pulse(c)
    
    # Check constraints
    valid, violation = check_constraints(pulse)
    if not valid:
        # Penalize constraint violations
        return 1e6 * (1.0 + violation)
    
    # Compute score
    try:
        sc = score_pulse(pulse)
        return sc
    except Exception as e:
        return 1e6

# Seed: DRAG-style Gaussian
def seed_pulse():
    """Generate the seed DRAG-style Gaussian pulse."""
    # Gaussian with sigma = 40/6 ns, offset to 0 at edges, area pi
    t = np.arange(N_STEPS) * DT + DT / 2  # midpoints
    sigma = 40.0 / 6.0
    # Center at t=20 ns
    t_center = 20.0
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    # Offset to make it 0 at edges: subtract env at t=0 and t=40
    env[0] -= env[0]  # This is 0 already if we want it 0 at edges
    # Actually, "offset to 0 at the edges" means the pulse is 0 at t=0 and t=40
    # The Gaussian is naturally small at edges, but let's enforce it
    # Scale to have area pi: ∫ env dt = pi
    area = np.trapz(env, t)
    if area > 0:
        env = env * (np.pi / area)
    
    # DRAG: add a Q component proportional to the derivative of I
    # dI/dt approximated by finite difference
    dI = np.zeros_like(t)
    dI[1:] = (env[1:] - env[:-1]) / DT
    dI[:-1] = (env[1:] - env[:-1]) / DT
    dI[0] = (env[1] - env[0]) / DT
    dI[-1] = (env[-1] - env[-2]) / DT
    
    # DRAG coefficient: typically small, let's use 0.1 * anharmonicity correction
    # For a pi pulse, DRAG coefficient ≈ -anharmonicity / (2 * drive_freq) but simplified
    # Let's use a small coefficient
    drag_coeff = 0.1
    Q = drag_coeff * dI
    I = env
    
    pulse = np.column_stack([I, Q])
    return pulse

# Fit Legendre coefficients to the seed pulse
def fit_legendre_to_seed(seed_p):
    """Fit 8 Legendre coefficients to the seed pulse."""
    I = seed_p[:, 0]
    Q = seed_p[:, 1]
    # Least squares fit: B @ c = I
    Ic, _, _, _ = np.linalg.lstsq(B_mid, I, rcond=None)
    Qc, _, _, _ = np.linalg.lstsq(B_mid, Q, rcond=None)
    return np.concatenate([Ic, Qc])

# Optimize
def optimize_pulse():
    """Optimize the 8 Legendre coefficients using L-BFGS-B."""
    seed = seed_pulse()
    c0 = fit_legendre_to_seed(seed)
    
    # Bounds for coefficients: keep them reasonable
    bounds = [(-1.0, 1.0)] * 16
    
    result = minimize(objective, c0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
    
    c_opt = result.x
    pulse = coeffs_to_pulse(c_opt)
    
    # Ensure constraints are satisfied
    valid, violation = check_constraints(pulse)
    if not valid:
        # Fall back to seed if optimization violated constraints
        return seed
    
    return pulse

# Precompute the optimized pulse at module load time
# But we need to be careful about time - let's do a quick optimization
_pulse_cache = None

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for the X-pi gate."""
    global _pulse_cache
    if _pulse_cache is not None:
        return _pulse_cache
    
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Try to use optimized pulse, fall back to seed
    try:
        # Do a quick optimization with limited iterations
        seed = seed_pulse()
        c0 = fit_legendre_to_seed(seed)
        bounds = [(-1.0, 1.0)] * 16
        result = minimize(objective, c0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 20, 'ftol': 1e-5, 'gtol': 1e-3})
        c_opt = result.x
        p = coeffs_to_pulse(c_opt)
        valid, _ = check_constraints(p)
        if valid:
            _pulse_cache = p
            return p
    except Exception:
        pass
    
    # Fall back to seed
    _pulse_cache = seed_pulse()
    return _pulse_cache
# EVOLVE-BLOCK-END
