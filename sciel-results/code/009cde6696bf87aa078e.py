"""Graph-dependent two-layer QAOA angle design for MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute the two-qubit expectation <Z_i Z_j> as a function of (gamma, beta)
# for QAOA p=2 with unweighted MaxCut. We track the operator Z_i Z_j through
# the circuit U = exp(-i beta X_i) exp(-i beta X_j) exp(-i gamma C_ij)
#          exp(-i beta X_i) exp(-i beta X_j) exp(-i gamma C_ij)
# where C_ij = (1 - Z_i Z_j)/2 for a single edge.
#
# We represent the state as a 4x4 density matrix (pure) and evolve it exactly.

# Pauli / operator matrices (2x2)
I2 = np.eye(2)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]])
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]])

# Two-qubit operators (qubit i = first, j = second)
ZZ = np.kron(Z2, Z2)
XI = np.kron(X2, I2)
IX = np.kron(I2, X2)

def _twoqubit_expectation(gamma, beta):
    """Return <Z_i Z_j> after QAOA p=2 on a single edge, starting from |+>|+>."""
    # Initial state |++> as a vector
    plus = np.array([1.0, 1.0]) / np.sqrt(2.0)
    psi = np.kron(plus, plus)

    def apply_unitary(u, vec):
        return u @ vec

    # Layer 1: exp(-i gamma C_ij), C_ij = (1 - ZZ)/2
    C_ij = 0.5 * (np.eye(4) - ZZ)
    # exp(-i gamma C_ij) = cos(gamma/2 * (1-ZZ)) ... use matrix exponential via
    # the fact C_ij is a projector-like: C_ij = (I-ZZ)/2, eigenvalues 0 and 1.
    # exp(-i gamma C_ij) = I*cos(gamma) - ... use spectral: C_ij^2 = C_ij (projector).
    # Indeed (I-ZZ)/2 squared = (I - 2ZZ + ZZ^2)/4 = (I - 2ZZ + I)/4 = (2I-2ZZ)/4 = (I-ZZ)/2.
    # So C_ij is a projector. exp(-i gamma P) = (1-P) + P exp(-i gamma) = I - P(1 - exp(-i gamma)).
    P = C_ij
    U1 = np.eye(4) - P * (1.0 - np.exp(-1j * gamma))
    psi = U1 @ psi

    # exp(-i beta X_i)
    UB_i = np.cos(beta) * np.eye(4) - 1j * np.sin(beta) * XI
    psi = UB_i @ psi
    # exp(-i beta X_j)
    UB_j = np.cos(beta) * np.eye(4) - 1j * np.sin(beta) * IX
    psi = UB_j @ psi

    # Layer 2: same as layer 1
    psi = U1 @ psi
    psi = UB_i @ psi
    psi = UB_j @ psi

    # <Z_i Z_j>
    val = np.vdot(psi, ZZ @ psi)
    return float(np.real(val))


def _expected_cut_ratio(gamma, beta, adj):
    """Expected cut / (n/2) style; actually return total expected cut value."""
    # Each edge contributes 0.5 * (1 - <Z_i Z_j>)
    zz = _twoqubit_expectation(gamma, beta)
    return 0.5 * (1.0 - zz) * adj.sum()


def _objective(gamma, beta, adj):
    """Return expected cut value for the given angles."""
    return _expected_cut_ratio(gamma, beta, adj)


def _optimize_for_graph(n, edges, p, deadline):
    """Search for best (gamma, beta) maximizing expected cut for one graph."""
    adj = np.zeros((n, n))
    for (u, v) in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    m = adj.sum()
    if m == 0:
        return np.tile([0.0, 0.0], (p, 1))

    # Coarse grid over the fundamental domain.
    # QAOA p=2 has periodicity; search gamma in [0, pi], beta in [0, pi].
    t0 = time.time()
    best_val = -np.inf
    best_angles = (0.35, 0.25)

    # Grid search
    ng = 60
    gams = np.linspace(0.0, np.pi, ng)
    betas = np.linspace(0.0, np.pi, ng)
    for g in gams:
        for b in betas:
            if time.time() > deadline:
                break
            val = _objective(g, b, adj)
            if val > best_val:
                best_val = val
                best_angles = (g, b)
    if time.time() > deadline:
        return np.tile([best_angles[0], best_angles[1]], (p, 1))

    # Local refinement (Nelder-Mead) around the grid best
    from scipy.optimize import minimize
    def neg_obj(x):
        return -_objective(x[0], x[1], adj)
    try:
        res = minimize(neg_obj, best_angles, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.success or res.fun < -best_val:
            best_angles = (float(res.x[0]), float(res.x[1]))
            best_val = float(-res.fun)
    except Exception:
        pass

    return np.tile([best_angles[0], best_angles[1]], (p, 1))


def angles(n, edges, p):
    """Return (p, 2) array of (gamma, beta) angles for the given graph."""
    deadline = time.time() + 100.0  # safety margin well under 120 s
    try:
        res = _optimize_for_graph(n, edges, p, deadline)
        return res
    except Exception:
        # Fallback to a reasonable default if anything fails.
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
