"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _precompute_graph(n, edges):
    """Precompute adjacency, edge list, and cut values for all masks."""
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edge_list.append((i, j))
    
    dim = 1 << n
    # Precompute cut value for each mask
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for i, j in edge_list:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cv += 1
        cut_vals[mask] = cv
    
    # Precompute for each mask, which edges are cut (for phase computation)
    # We need to know, for each mask and each edge, whether the edge is cut
    # This is a (dim, len(edge_list)) boolean matrix
    m = len(edge_list)
    edge_cut_matrix = np.zeros((dim, m), dtype=np.int8)
    for mask in range(dim):
        for ei, (i, j) in enumerate(edge_list):
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                edge_cut_matrix[mask, ei] = 1
    
    return adj, edge_list, cut_vals, edge_cut_matrix, dim

def _compute_expected_cut_fast(params, n, edge_list, cut_vals, edge_cut_matrix, dim):
    """
    Compute expected cut using direct probability computation.
    For p=2 QAOA on MaxCut, the probability of each mask can be computed
    via interference of amplitudes from the initial |+>^n state.
    
    The key insight: for small n (<=10), we can compute the full state vector
    efficiently using tensor operations, but we can also use the fact that
    the cost operator is diagonal in the computational basis to simplify.
    
    Actually, the most efficient approach for n<=10 is to use the fact that
    exp(-i gamma C) is diagonal in the computational basis (C is diagonal),
    so it just applies phases. The mixer exp(-i beta sum X) is a product of
    single-qubit rotations.
    
    Let's use a vectorized approach:
    1. Start with uniform amplitudes 1/sqrt(2^n)
    2. Apply cost phase: multiply each amplitude by exp(-i gamma * cut_val[mask])
       Wait, C = sum (1-Z_iZ_j)/2, so the eigenvalue for mask is cut_val[mask].
       So exp(-i gamma C) applies phase exp(-i gamma * cut_val[mask]) to each basis state.
    3. Apply mixer: this is where it gets tricky because it's not diagonal.
    
    For n<=10, let's just do the state vector simulation but vectorized with numpy.
    """
    gamma1, beta1, gamma2, beta2 = params
    m = len(edge_list)
    
    # Initial state: uniform
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Layer 1: Cost operator (diagonal)
    # Phase for each mask is exp(-i gamma1 * cut_val[mask])
    phase1 = np.exp(-1j * gamma1 * cut_vals)
    state = state * phase1
    
    # Layer 1: Mixer operator
    # exp(-i beta1 * sum_k X_k) = prod_k exp(-i beta1 X_k)
    # Each exp(-i beta X) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    # We need to apply this to each qubit.
    
    def apply_mixer(state, beta, n):
        """Apply exp(-i beta sum X) to state vector, vectorized."""
        cb = np.cos(beta)
        sb = np.sin(beta)
        new_state = state.copy()
        for k in range(n):
            # For each pair of states differing only in bit k
            # mask with bit k = 0 and mask with bit k = 1
            bit_val = 1 << k
            # Create masks for all states
            # States where bit k is 0: (mask & bit_val) == 0
            # States where bit k is 1: (mask & bit_val) != 0
            # We can use reshape and broadcasting
            shape = [2] * n
            tensor = state.reshape(shape)
            # Swap axis k with last axis for easier manipulation
            # Actually, let's just use direct indexing
            # For efficiency, iterate over all states where bit k is 0
            # This is still O(2^n) per qubit, but with numpy it's fast
            # Let's use a different approach: create index arrays
            masks = np.arange(dim)
            # Masks where bit k is 0
            mask0 = masks[(masks & bit_val) == 0]
            mask1 = mask0 | bit_val
            a = state[mask0]
            b = state[mask1]
            state[mask0] = cb * a - 1j * sb * b
            state[mask1] = -1j * sb * a + cb * b
        return state
    
    state = apply_mixer(state, beta1, n)
    
    # Layer 2: Cost operator (diagonal)
    phase2 = np.exp(-1j * gamma2 * cut_vals)
    state = state * phase2
    
    # Layer 2: Mixer operator
    state = apply_mixer(state, beta2, n)
    
    # Expected cut = sum over masks of |state[mask]|^2 * cut_val[mask]
    probs = np.abs(state) ** 2
    expected_cut = np.dot(probs, cut_vals)
    
    return expected_cut

def _exact_maxcut(cut_vals):
    """Compute exact MaxCut value from precomputed cut values."""
    return float(np.max(cut_vals))

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Precompute graph properties
    adj, edge_list, cut_vals, edge_cut_matrix, dim = _precompute_graph(n, edges)
    
    # Calculate density for heuristic initialization
    max_edges = n * (n - 1) / 2
    density = len(edges) / max_edges if max_edges > 0 else 0.0
    
    # Exact MaxCut
    opt_cut = _exact_maxcut(cut_vals)
    if opt_cut == 0:
        return np.tile([0.35, 0.25], (p, 1))
        
    # Heuristic initial guess based on density
    if density < 0.3:
        init = np.array([0.35, 0.25, 0.25, 0.15])
    elif density < 0.6:
        init = np.array([0.40, 0.20, 0.30, 0.10])
    else:
        init = np.array([0.45, 0.15, 0.35, 0.05])
        
    best_params = init.copy()
    best_val = _compute_expected_cut_fast(init, n, edge_list, cut_vals, edge_cut_matrix, dim)
    
    # Objective function to maximize
    def objective(params):
        if time.time() > deadline:
            return 1e10
        val = _compute_expected_cut_fast(params, n, edge_list, cut_vals, edge_cut_matrix, dim)
        return -val  # Minimize negative
        
    # Try multiple optimization strategies
    # Strategy 1: L-BFGS-B from heuristic init
    try:
        res = minimize(objective, init, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False, 'maxls': 50})
        if time.time() < deadline:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Strategy 2: Random restarts with L-BFGS-B
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(12345)
        n_restarts = 0
        while time.time() < deadline - 5.0 and n_restarts < 20:
            # Generate random initial point
            rand_init = rng.uniform(0.05, 1.5, size=4)
            # Ensure reasonable ranges
            rand_init = np.clip(rand_init, 0.01, 1.5)
            try:
                res = minimize(objective, rand_init, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'maxls': 30})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
            except Exception:
                pass
            n_restarts += 1
    
    # Strategy 3: Nelder-Mead from best found so far
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if time permits
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 50, 'disp': False, 'maxls': 50})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Ensure parameters are within reasonable bounds
    best_params = np.clip(best_params, 0.0, 2.0)
    
    # Format output
    # p=2, so we need 2 rows of (gamma, beta)
    # best_params is [gamma1, beta1, gamma2, beta2]
    out = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
