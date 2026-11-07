"""
Robust X-pi pulse design for a three-level transmon.

Strategy:
  - Start from the DRAG-style Gaussian seed.
  - Parameterize the pulse as a sum of the Gaussian seed plus a small number
    of Legendre polynomial corrections (shifted to [0,1]).
  - Optimize coefficients with L-BFGS-B to minimize the mean+max infidelity
    over the 9 detuning/scale cases.
  - Enforce amplitude and slew constraints via projection during optimization.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# --- Constants from the task ---
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
SCALE_BOUNDS = np.array([0.97, 1.0, 1.03])
T1 = 25000.0  # ns
T_PHI = 40000.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
DEADLINE = 330.0  # seconds wall-clock limit


def _a_operator():
    """Return the 3x3 lowering operator a = diag(sqrt([1,2]),1)."""
    return np.diag([np.sqrt(1.0), np.sqrt(2.0), 1.0])


def _a_dagger():
    """Return the 3x3 raising operator a^dagger."""
    return _a_operator().T


def _number_operator():
    """Return n = a^dagger a."""
    return _a_dagger() @ _a_operator()


def _free_hamiltonian(detuning, scale):
    """
    H = detuning * n + 0.5 * anharmonicity * n*(n-1)
      + 0.5 * scale * (I*(a+a^dagger) + i*Q*(a-a^dagger))
    But for the free part (no drive), we just have:
    H_free = detuning * n + 0.5 * anharmonicity * n*(n-1)
    """
    n = _number_operator()
    n_nm1 = n @ (n - np.eye(3))
    H = detuning * n + 0.5 * ANHARMONICITY * n_nm1
    return H


def _drive_hamiltonian(I, Q, scale):
    """
    H_drive = 0.5 * scale * (I*(a+a^dagger) + i*Q*(a-a^dagger))
    """
    a = _a_operator()
    ad = _a_dagger()
    H = 0.5 * scale * (I * (a + ad) + 1j * Q * (a - ad))
    return H


def _liouvillian_step(H_full, dt):
    """
    Compute the superoperator for one time step.
    L(rho) = -i*[H, rho] + sum_k (L_k rho L_k^dagger - 0.5{L_k^dagger L_k, rho})
    We exponentiate the Liouvillian: rho_{k+1} = exp(L * dt) rho_k
    """
    # Lindblad operators
    a = _a_operator()
    n = _number_operator()
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / T_PHI)

    # Build the Liouvillian as a 9x9 matrix acting on vec(rho)
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    dim = 3
    L_mat = np.zeros((9, 9), dtype=complex)

    # Coherent part: -i [H, rho] = -i H rho + i rho H
    # vec(-i H rho) = -i (I ⊗ H) vec(rho)  -- wait, need to be careful
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    # vec(H rho) = (I^T ⊗ H) vec(rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    I3 = np.eye(3)
    L_mat += -1j * (np.kron(I3, H_full)) + 1j * (np.kron(H_full.T, I3))

    # Dissipative part
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        # Lk rho Lk^dagger: vec = (Lk^T ⊗ Lk) vec(rho)  -- actually (Lk^dagger)^T ⊗ Lk
        # vec(Lk rho Lk^dagger) = ((Lk^dagger)^T ⊗ Lk) vec(rho) = (conj(Lk) ⊗ Lk) vec(rho)
        L_mat += np.kron(Lk.conj(), Lk)
        # -0.5 {Lk^dagger Lk, rho} = -0.5 Lk^dagger Lk rho - 0.5 rho Lk^dagger Lk
        Lk_dag_Lk = Lk_dag @ Lk
        L_mat += -0.5 * (np.kron(I3, Lk_dag_Lk)) - 0.5 * (np.kron(Lk_dag_Lk.T, I3))

    # Exponentiate
    E = expm(L_mat * dt)
    return E


def _fidelity(rho, psi):
    """Compute fidelity |<psi|rho|psi>|^2 for a pure state psi."""
    psi_vec = psi.reshape(-1, 1)
    prob = np.abs(psi_vec.conj().T @ rho @ psi_vec)[0, 0].real
    return np.sqrt(np.clip(prob, 0.0, 1.0))


def _simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """
    Simulate the pulse and return the final density matrix.
    pulse_I, pulse_Q: arrays of length N_STEPS.
    """
    H_free = _free_hamiltonian(detuning, scale)
    a = _a_operator()
    ad = _a_dagger()
    n = _number_operator()
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / T_PHI)
    I3 = np.eye(3)

    # Precompute Lindblad superoperator parts
    Lk_parts = []
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        Lk_parts.append((np.kron(Lk.conj(), Lk),
                         -0.5 * (np.kron(I3, Lk_dag_Lk)) - 0.5 * (np.kron(Lk_dag_Lk.T, I3))))

    # Start from |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = rho.reshape(9)

    for step in range(N_STEPS):
        I_val = pulse_I[step]
        Q_val = pulse_Q[step]
        H_drive = 0.5 * scale * (I_val * (a + ad) + 1j * Q_val * (a - ad))
        H_full = H_free + H_drive

        # Build Liouvillian
        L_mat = -1j * (np.kron(I3, H_full)) + 1j * (np.kron(H_full.T, I3))
        for (diss1, diss2) in Lk_parts:
            L_mat += diss1 + diss2

        E = expm(L_mat * DT)
        rho_vec = E @ rho_vec
        rho = rho_vec.reshape(3, 3)

    return rho


def _compute_infidelity(pulse_I, pulse_Q):
    """
    Compute the scoring metric: mean + max of infidelities over 9 cases.
    """
    # Test states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    test_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]

    # X gate: |0> -> |1>, |1> -> |0>
    # Fidelity to X|psi>
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)

    infidelities = []
    for det in DETUNING_BOUNDS:
        for scl in SCALE_BOUNDS:
            rho = _simulate_pulse(pulse_I, pulse_Q, det, scl)
            case_inf = 0.0
            for psi in test_states:
                psi_target = X @ psi
                fid = _fidelity(rho, psi_target)
                case_inf += (1.0 - fid)
            case_inf /= len(test_states)
            infidelities.append(case_inf)

    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _gaussian_seed():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0  # sigma = 40/6 ns
    # Gaussian centered at T_TOTAL/2 = 20 ns
    g = np.exp(-((t - T_TOTAL / 2) ** 2) / (2 * sigma ** 2))
    # Offset to 0 at edges: subtract g[0] and g[-1]
    g = g - g[0]
    # Normalize area to pi
    area = np.trapz(g, dx=DT)
    if abs(area) > 1e-15:
        g = g * (np.pi / area)
    # Apply amplitude bound
    g = np.clip(g, -AMP_BOUND, AMP_BOUND)
    pulse_I = g
    pulse_Q = np.zeros(N_STEPS)
    return pulse_I, pulse_Q


def _project_pulse(pulse_I, pulse_Q):
    """
    Project the pulse to satisfy amplitude and slew constraints.
    Simple iterative projection.
    """
    I = pulse_I.copy()
    Q = pulse_Q.copy()

    # Enforce amplitude constraint
    for _ in range(5):
        amp = np.sqrt(I**2 + Q**2)
        mask = amp > AMP_BOUND
        if np.any(mask):
            scale = AMP_BOUND / amp[mask]
            I[mask] *= scale
            Q[mask] *= scale

    # Enforce slew constraint: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # This means |I[i+1]-I[i]| <= SLEW_BOUND*DT and same for Q, but it's the vector norm
    # ||[I[i+1]-I[i], Q[i+1]-Q[i]]|| / DT <= SLEW_BOUND
    # So ||delta|| <= SLEW_BOUND * DT
    max_delta = SLEW_BOUND * DT
    for _ in range(5):
        # From zero to step 0
        d = np.array([I[0], Q[0]])
        norm = np.linalg.norm(d)
        if norm > max_delta:
            scale = max_delta / norm
            I[0] *= scale
            Q[0] *= scale
        # Between steps
        for i in range(N_STEPS - 1):
            d = np.array([I[i+1] - I[i], Q[i+1] - Q[i]])
            norm = np.linalg.norm(d)
            if norm > max_delta:
                scale = max_delta / norm
                I[i+1] = I[i] + d[0] * scale
                Q[i+1] = Q[i] + d[1] * scale
        # From last step to zero
        d = np.array([-I[-1], -Q[-1]])
        norm = np.linalg.norm(d)
        if norm > max_delta:
            scale = max_delta / norm
            I[-1] *= scale
            Q[-1] *= scale

    # Re-enforce amplitude after slew projection
    for _ in range(3):
        amp = np.sqrt(I**2 + Q**2)
        mask = amp > AMP_BOUND
        if np.any(mask):
            scale = AMP_BOUND / amp[mask]
            I[mask] *= scale
            Q[mask] *= scale

    return I, Q


def _legendre_basis(N):
    """
    Generate shifted Legendre polynomials on [0,1] evaluated at N points.
    Returns array of shape (N, N) where column j is P_j(2t-1) for t in [0,1].
    """
    t = np.linspace(0, 1, N)
    x = 2 * t - 1  # map to [-1, 1]
    basis = np.zeros((N, N))
    P0 = np.ones_like(x)
    P1 = x.copy()
    basis[:, 0] = P0
    if N > 1:
        basis[:, 1] = P1
    for j in range(2, N):
        Pj = ((2 * j - 1) * x * P1 - (j - 1) * P0) / j
        basis[:, j] = Pj
        P0, P1 = P1, Pj
    return basis


def _objective(params, basis_I, basis_Q, seed_I, seed_Q):
    """
    Objective function for optimization.
    params: coefficients for I and Q corrections.
    """
    n_I = len(basis_I)
    I = seed_I + basis_I @ params[:n_I]
    Q = seed_Q + basis_Q @ params[n_I:]
    I, Q = _project_pulse(I, Q)
    return _compute_infidelity(I, Q)


def pulse(target_gate, hardware_model):
    """
    Generate a robust X-pi pulse.
    Returns an (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - 30.0  # 30s safety margin

    # Generate seed
    seed_I, seed_Q = _gaussian_seed()
    best_I, best_Q = seed_I.copy(), seed_Q.copy()
    best_score = _compute_infidelity(seed_I, seed_Q)

    # Set up Legendre basis for corrections
    n_terms = 6  # 6 terms for I, 6 for Q
    basis_I = _legendre_basis(N_STEPS)[:, :n_terms]
    basis_Q = _legendre_basis(N_STEPS)[:, :n_terms]

    # Initial parameters: zero
    x0 = np.zeros(2 * n_terms)

    # Bounds: keep corrections small to stay near seed
    bounds = [(-0.15, 0.15)] * (2 * n_terms)

    # Optimize with L-BFGS-B
    try:
        result = minimize(
            _objective,
            x0,
            args=(basis_I, basis_Q, seed_I, seed_Q),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxls': 50, 'maxfun': 10000}
        )
        if result.success or result.fun < best_score:
            params = result.x
            n_I = len(basis_I)
            I = seed_I + basis_I @ params[:n_I]
            Q = seed_Q + basis_Q @ params[n_I:]
            I, Q = _project_pulse(I, Q)
            score = _compute_infidelity(I, Q)
            if score < best_score:
                best_I, best_Q = I, Q
                best_score = score
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        pass  # Return what we have

    # Final projection to ensure constraints are met
    best_I, best_Q = _project_pulse(best_I, best_Q)

    # Return as (80, 2) array
    return np.column_stack([best_I, best_Q])


# Keep the original interface
def pulse_fn(target_gate, hardware_model):
    return pulse(target_gate, hardware_model)
