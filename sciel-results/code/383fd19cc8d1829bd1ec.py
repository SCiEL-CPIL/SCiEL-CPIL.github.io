"""QAOA parameter design with tensor-network-style exact simulation and optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import product
from functools import lru_cache


def _build_edge_list(n, edges):
    """Build edge list as list of (u, v) tuples."""
    if edges is None:
        return []
    if isinstance(edges, np.ndarray):
        if edges.size == 0:
            return []
        edges = edges.tolist()
    edge_list = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            u, v = int(e[0]), int(e[1])
            if u < v:
                edge_list.append((u, v))
    return edge_list


def _exact_optimum(n, edges):
    """Compute exact MaxCut optimum by exhaustive search."""
    if n > 20:
        # Fallback for large n (shouldn't happen in this task)
        return 0
    edge_list = _build_edge_list(n, edges)
    m = len(edge_list)
    if m == 0:
        return 0
    best = 0
    for bits in range(1 << n):
        cut = 0
        for u, v in edge_list:
            if ((bits >> u) ^ (bits >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _expected_cut_exact(n, edges, gammas, betas):
    """
    Compute expected cut value using exact state vector simulation.
    State evolves as: start in |+>^n, apply exp(-i*gamma*C) then exp(-i*beta*sum X) for each layer.
    C = sum_{(u,v) in E} (1 - Z_u Z_v)/2  (unweighted MaxCut cost)
    Expected cut = sum_{(u,v) in E} P(spin_u != spin_v)
    """
    edge_list = _build_edge_list(n, edges)
    m = len(edge_list)
    if m == 0:
        return 0.0

    dim = 1 << n
    # Initial state: |+>^n = 1/sqrt(2^n) * ones
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    # Precompute Z_u Z_v diagonal for each edge
    # Z_u Z_v for basis state |s> is (+1) if s_u == s_v, (-1) if s_u != s_v
    # So (1 - Z_u Z_v)/2 is 1 if different, 0 if same
    # C = sum_e (1 - Z_u Z_v)/2, so exp(-i*gamma*C) = prod_e exp(-i*gamma*(1-Z_u Z_v)/2)
    # = prod_e [exp(-i*gamma/2) if different, 1 if same]
    # So the phase factor for basis state |s> is exp(-i*gamma * (number of cut edges in s)/2)

    # For efficiency, precompute for each basis state the number of cut edges
    cut_counts = np.zeros(dim, dtype=np.int32)
    for u, v in edge_list:
        # For each basis state, check if bits u and v differ
        mask_u = 1 << u
        mask_v = 1 << v
        for s in range(dim):
            if ((s >> u) ^ (s >> v)) & 1:
                cut_counts[s] += 1

    # Precompute X mixer: sum of X_i
    # X_i flips bit i. sum X_i is a matrix that flips exactly one bit.
    # For state |s>, (sum X_i)|s> = sum_i |s with bit i flipped>
    # This is a tridiagonal-like structure in the computational basis.

    p = len(gammas)
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Apply cost operator: multiply state by exp(-i*gamma*cut_counts/2)
        phase = np.exp(-1j * gamma * cut_counts / 2.0)
        state = state * phase

        # Apply mixer: exp(-i*beta*sum_i X_i)
        # This is a 2D rotation in the subspace spanned by |s> and |s^1_i> for each i
        # Actually, sum_i X_i couples |s> to all |s with one bit flipped>
        # exp(-i*beta * sum_i X_i) is not simply a product of 2x2 rotations
        # because the X_i operators don't commute with each other in the sense that
        # applying them simultaneously is different from sequential.
        #
        # However, sum_i X_i is a Hermitian operator. We can compute exp(-i*beta*A)
        # where A = sum_i X_i.
        #
        # A is a sparse matrix. For small n (up to 10, dim=1024), we can construct it.
        # But for efficiency, let's use the fact that A has a simple structure.
        #
        # Actually, let's just construct the mixer matrix for small n.
        # For n up to 10, dim=1024, a dense 1024x1024 complex matrix is ~16MB, fine.

        # Build A = sum_i X_i
        # A[s, s'] = 1 if s' differs from s in exactly one bit
        A = np.zeros((dim, dim), dtype=np.complex128)
        for s in range(dim):
            for i in range(n):
                s_prime = s ^ (1 << i)
                A[s_prime, s] = 1.0

        # Compute exp(-i*beta*A) using eigendecomposition or scipy
        # A is real symmetric, so eigendecomposition is efficient
        # For dim up to 1024, this is fine
        eigvals, eigvecs = np.linalg.eigh(A)
        # exp(-i*beta*A) = V diag(exp(-i*beta*eigvals)) V^H
        exp_eigvals = np.exp(-1j * beta * eigvals)
        mixer = (eigvecs * exp_eigvals) @ eigvecs.conj().T
        state = mixer @ state

    # Compute expected cut: sum over edges of P(spin_u != spin_v)
    # P(spin_u != spin_v) = sum_{s: s_u != s_v} |<s|state>|^2
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for u, v in edge_list:
        # Sum probabilities where bits u and v differ
        for s in range(dim):
            if ((s >> u) ^ (s >> v)) & 1:
                expected_cut += probs[s]
    return expected_cut


def _objective(params, n, edges, opt_cut):
    """Objective function to minimize (negative of expected cut ratio)."""
    gammas = params[:2]
    betas = params[2:4]
    try:
        ec = _expected_cut_exact(n, edges, gammas, betas)
        if opt_cut > 0:
            return -ec / opt_cut
        return -ec
    except Exception:
        return 1e6


def angles(n, edges, p):
    """
    Design QAOA angles for given graph.
    Returns array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit

    edge_list = _build_edge_list(n, edges)
    opt_cut = _exact_optimum(n, edges)

    # If no edges or trivial case
    if opt_cut == 0:
        return np.zeros((p, 2))

    dim = 1 << n

    # Multi-start optimization
    best_params = np.array([0.35, 0.35, 0.25, 0.25])
    best_obj = 1e6

    # Initial guesses
    initial_guesses = [
        [0.35, 0.35, 0.25, 0.25],
        [0.5, 0.5, 0.3, 0.3],
        [0.2, 0.2, 0.1, 0.1],
        [0.7, 0.7, 0.4, 0.4],
        [0.1, 0.1, 0.05, 0.05],
        [0.4, 0.4, 0.2, 0.2],
    ]

    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(8):
        g = rng.uniform(0.05, 1.0, size=2)
        b = rng.uniform(0.05, 0.6, size=2)
        initial_guesses.append(np.concatenate([g, b]))

    bounds = [(0.01, 2.0), (0.01, 2.0), (0.01, 1.0), (0.01, 1.0)]

    for x0 in initial_guesses:
        if time.time() > deadline:
            break

        # Nelder-Mead
        try:
            res = minimize(
                _objective, x0, args=(n, edges, opt_cut),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass

        # L-BFGS-B polish
        if time.time() > deadline:
            break
        try:
            res2 = minimize(
                _objective, best_params, args=(n, edges, opt_cut),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-12}
            )
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Extract gammas and betas
    gammas = best_params[:2]
    betas = best_params[2:4]

    result = np.column_stack([gammas, betas])
    return result
# EVOLVE-BLOCK-END
