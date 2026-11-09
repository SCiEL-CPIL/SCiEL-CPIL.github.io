"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import time

# Time budget for the whole evaluation
_TIME_LIMIT = 110.0  # seconds, with safety margin before 120s kill
_start_time = None


def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()


def _time_remaining():
    if _start_time is None:
        return _TIME_LIMIT
    return max(0.0, _TIME_LIMIT - (time.time() - _start_time))


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(adj, gammas, betas):
    """
    Compute the expected cut value for QAOA with given angles.
    
    For MaxCut with Hamiltonian C = sum_{(i,j) in E} (1 - Z_i Z_j)/2,
    the state after QAOA layers is |psi> = prod_l exp(-i beta_l X_sum) exp(-i gamma_l C) |+>^n
    
    The expected cut = sum_{(i,j) in E} <psi| (1 - Z_i Z_j)/2 |psi>
                     = |E|/2 - (1/2) sum_{(i,j) in E} <Z_i Z_j>
    
    We need to compute <Z_i Z_j> for each edge.
    
    For the two-layer QAOA starting from |+>^n:
    - Start: |+>^n, all Z expectations are 0, all Z_i Z_j correlations are 0 initially.
    - After exp(-i gamma C): this creates entanglement.
    - After exp(-i beta X_sum): this rotates.
    
    The exact computation of <Z_i Z_j> for QAOA is complex. Let me use a different approach.
    
    Actually, for small n (6,8,10), we can compute the full state vector.
    """
    n = adj.shape[0]
    p = len(gammas)
    
    # Build the cut Hamiltonian matrix C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    # In the computational basis, Z_i Z_j has eigenvalue +1 if bits are same, -1 if different
    # C |x> = (number of cut edges in x) |x>
    
    # For n up to 10, 2^10 = 1024 states, which is manageable
    
    dim = 1 << n
    
    # Precompute the cut values for all bitstrings
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                edge_list.append((i, j))
    
    # C is diagonal in computational basis: C[x] = number of edges cut by assignment x
    c_vals = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        cut_count = 0
        for i, j in edge_list:
            bit_i = (x >> i) & 1
            bit_j = (x >> j) & 1
            if bit_i != bit_j:
                cut_count += 1
        c_vals[x] = cut_count
    
    # X_sum = sum_i X_i, which is the bit-flip-all operator
    # In computational basis, X_sum |x> = |~x & (2^n - 1)>
    
    # Start state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for l in range(p):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i gamma C): phase each basis state
        state = state * np.exp(-1j * gamma * c_vals)
        
        # Apply exp(-i beta X_sum): this is exp(-i beta sum_i X_i)
        # sum_i X_i in computational basis: |x> -> |x ^ all_ones>
        # So exp(-i beta X_sum) = cos(beta) I - i sin(beta) X_sum
        # X_sum |x> = |x ^ mask> where mask = 2^n - 1
        mask = dim - 1
        flipped = np.arange(dim) ^ mask
        state = np.cos(beta) * state - 1j * np.sin(beta) * state[flipped]
    
    # Compute <Z_i Z_j> for each edge
    # Z_i |x> = (-1)^{bit_i} |x>
    # Z_i Z_j |x> = (-1)^{bit_i + bit_j} |x>
    # <Z_i Z_j> = sum_x |psi_x|^2 * (-1)^{bit_i + bit_j}
    
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for i, j in edge_list:
        # (-1)^{bit_i + bit_j} = +1 if bits same, -1 if different
        # This is 1 - 2*(bit_i XOR bit_j)
        corr = 0.0
        for x in range(dim):
            bit_i = (x >> i) & 1
            bit_j = (x >> j) & 1
            if bit_i == bit_j:
                corr += probs[x]
            else:
                corr -= probs[x]
        # <Z_i Z_j> = corr
        # Expected cut contribution from edge (i,j) = (1 - <Z_i Z_j>)/2
        expected_cut += (1.0 - corr) / 2.0
    
    return expected_cut


def _exact_optimum(n, edges):
    """Brute force exact MaxCut for small graphs."""
    dim = 1 << n
    edge_list = [(u, v) for u, v in edges]
    
    best = 0
    for x in range(dim):
        cut = 0
        for i, j in edge_list:
            bit_i = (x >> i) & 1
            bit_j = (x >> j) & 1
            if bit_i != bit_j:
                cut += 1
        if cut > best:
            best = cut
    
    return float(best)


def _objective(gammas, betas, adj, n, edges, optimum):
    """Compute the ratio expected_cut / optimum."""
    ec = _expected_cut(adj, gammas, betas)
    if optimum <= 0:
        return 1.0
    return ec / optimum


def _optimize_angles(n, edges, p, adj, optimum, time_budget):
    """
    Optimize QAOA angles for a given graph.
    Uses gradient-free optimization with multiple restarts.
    """
    if time_budget < 1.0:
        # Not enough time, return default
        return np.array([[0.5, 0.5]] * p)
    
    # Initial guesses based on heuristics
    # Common good starting points for QAOA
    init_points = [
        [0.5, 0.5] * p,
        [0.3, 0.3] * p,
        [0.7, 0.2] * p,
        [0.2, 0.7] * p,
        [0.4, 0.4] * p,
        [0.6, 0.3] * p,
        [0.1, 0.1] * p,
        [0.8, 0.1] * p,
    ]
    
    # Add some random initializations
    rng = np.random.RandomState(42)
    for _ in range(4):
        init_points.append(rng.uniform(0, np.pi, size=2 * p).tolist())
    
    best_angles = None
    best_score = -np.inf
    
    # Evaluate initial points
    for init in init_points:
        gammas = init[0::2]
        betas = init[1::2]
        score = _objective(gammas, betas, adj, n, edges, optimum)
        if score > best_score:
            best_score = score
            best_angles = init.copy()
    
    if best_angles is None:
        return np.array([[0.5, 0.5]] * p)
    
    # Nelder-Mead style local refinement
    # Use scipy if available for more robust optimization
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy and time_budget > 5.0:
        # Try scipy optimization from the best initial point
        def neg_obj(params):
            gammas = params[0::2]
            betas = params[1::2]
            return -_objective(gammas, betas, adj, n, edges, optimum)
        
        x0 = np.array(best_angles)
        
        # Try a few different optimizers
        for method in ['Nelder-Mead', 'Powell']:
            if time_budget < 2.0:
                break
            try:
                result = minimize(neg_obj, x0, method=method,
                                  options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_angles = result.x.tolist()
            except Exception:
                pass
    
    # Additional local search with smaller steps
    if time_budget > 3.0:
        # Simple coordinate descent / hill climbing
        current = np.array(best_angles, dtype=np.float64)
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for ss in step_sizes:
            if time_budget < 1.0:
                break
            improved = True
            while improved and time_budget > 1.0:
                improved = False
                for d in range(2 * p):
                    for direction in [1, -1]:
                        trial = current.copy()
                        trial[d] += direction * ss
                        # Clip to reasonable range
                        trial = np.clip(trial, 0, np.pi)
                        gammas = trial[0::2]
                        betas = trial[1::2]
                        score = _objective(gammas, betas, adj, n, edges, optimum)
                        if score > best_score + 1e-10:
                            best_score = score
                            best_angles = trial.tolist()
                            current = trial
                            improved = True
    
    return np.array(best_angles).reshape(p, 2)


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of edges (u, v)
        p: number of QAOA layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    _init_time()
    
    # Build adjacency
    adj = _build_adjacency(n, edges)
    
    # Compute exact optimum
    optimum = _exact_optimum(n, edges)
    
    # Check time
    remaining = _time_remaining()
    
    if remaining < 2.0:
        # Not enough time for optimization, return heuristic
        return np.array([[0.5, 0.5]] * p)
    
    # Optimize
    result = _optimize_angles(n, edges, p, adj, optimum, remaining)
    
    # Ensure shape is correct
    if result.shape != (p, 2):
        result = np.resize(result, (p, 2))
    
    # Clip to [0, pi]
    result = np.clip(result, 0, np.pi)
    
    return result


def graph_cases(seed=113):
    """Generate development test graphs."""
    rng = np.random.RandomState(seed)
    cases = []
    for n in [6, 8, 10]:
        for density in [0.25, 0.55]:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            cases.append((n, edges))
    return cases
# EVOLVE-BLOCK-END
