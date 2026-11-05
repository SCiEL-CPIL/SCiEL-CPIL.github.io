"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _precompute_graph(n, edges):
    """Precompute adjacency, edge list, cut values, and mixer index pairs."""
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
    
    # Vectorized cut value computation
    masks = np.arange(dim, dtype=np.int64)
    cut_vals = np.zeros(dim, dtype=np.float64)
    for i, j in edge_list:
        bi = (masks >> i) & 1
        bj = (masks >> j) & 1
        cut_vals += (bi != bj).astype(np.float64)
    
    # Precompute mixer index pairs for each qubit
    mixer_indices = []
    for k in range(n):
        bit_val = 1 << k
        mask0 = masks[(masks & bit_val) == 0]
        mask1 = mask0 | bit_val
        mixer_indices.append((mask0, mask1))
    
    return adj, edge_list, cut_vals, dim, mixer_indices

def _apply_mixer(state, beta, mixer_indices):
    """Apply exp(-i beta sum X) using precomputed indices, no copies."""
    cb = np.cos(beta)
    sb = np.sin(beta)
    for mask0, mask1 in mixer_indices:
        a = state[mask0]
        b = state[mask1]
        # Use in-place operations with temp to avoid aliasing issues
        new_a = cb * a - 1j * sb * b
        new_b = -1j * sb * a + cb * b
        state[mask0] = new_a
        state[mask1] = new_b
    return state

def _compute_expected_cut(params, cut_vals, dim, mixer_indices, init_state):
    """
    Compute expected cut using direct state-vector simulation.
    For p=2 QAOA on MaxCut.
    """
    gamma1, beta1, gamma2, beta2 = params
    
    # Start with uniform state
    state = init_state.copy()
    
    # Layer 1: Cost operator (diagonal)
    state *= np.exp(-1j * gamma1 * cut_vals)
    
    # Layer 1: Mixer operator
    _apply_mixer(state, beta1, mixer_indices)
    
    # Layer 2: Cost operator (diagonal)
    state *= np.exp(-1j * gamma2 * cut_vals)
    
    # Layer 2: Mixer operator
    _apply_mixer(state, beta2, mixer_indices)
    
    # Expected cut
    probs = np.abs(state) ** 2
    return float(np.dot(probs, cut_vals))

def _exact_maxcut(cut_vals):
    """Compute exact MaxCut value from precomputed cut values."""
    return float(np.max(cut_vals))

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin before 120s limit
    
    # Precompute graph properties
    adj, edge_list, cut_vals, dim, mixer_indices = _precompute_graph(n, edges)
    
    # Precompute initial state
    init_state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Calculate density for heuristic initialization
    max_edges = n * (n - 1) / 2
    density = len(edges) / max_edges if max_edges > 0 else 0.0
    
    # Exact MaxCut
    opt_cut = _exact_maxcut(cut_vals)
    if opt_cut == 0:
        return np.tile([0.35, 0.25], (p, 1))
        
    # Heuristic initial guess based on density
    # Known good QAOA parameters for MaxCut
    if density < 0.3:
        init = np.array([0.35, 0.25, 0.25, 0.15])
    elif density < 0.6:
        init = np.array([0.40, 0.20, 0.30, 0.10])
    else:
        init = np.array([0.45, 0.15, 0.35, 0.05])
        
    best_params = init.copy()
    best_val = _compute_expected_cut(init, cut_vals, dim, mixer_indices, init_state)
    
    # Objective function to maximize
    def objective(params):
        if time.time() > deadline:
            return 1e10
        val = _compute_expected_cut(params, cut_vals, dim, mixer_indices, init_state)
        return -val  # Minimize negative
        
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
    
    # Strategy 2: Random restarts with L-BFGS-B using density-aware ranges
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(12345)
        n_restarts = 0
        while time.time() < deadline - 5.0 and n_restarts < 50:
            # Better initialization: use density-aware ranges
            if density < 0.3:
                g_range = (0.2, 0.6)
                b_range = (0.1, 0.4)
            elif density < 0.6:
                g_range = (0.3, 0.7)
                b_range = (0.05, 0.3)
            else:
                g_range = (0.4, 0.9)
                b_range = (0.02, 0.2)
            
            rand_init = np.array([
                rng.uniform(*g_range),
                rng.uniform(*b_range),
                rng.uniform(*g_range),
                rng.uniform(*b_range)
            ])
            rand_init = np.clip(rand_init, 0.01, 1.5)
            try:
                res = minimize(objective, rand_init, method='L-BFGS-B', 
                               options={'maxiter': 80, 'disp': False, 'maxls': 50})
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
                          options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Differential Evolution for global search
    if time.time() < deadline - 15.0:
        try:
            bounds = [(0.01, 1.5), (0.01, 1.5), (0.01, 1.5), (0.01, 1.5)]
            res = differential_evolution(objective, bounds, maxiter=60, popsize=20,
                                         seed=42, tol=1e-6, polish=True)
            if time.time() < deadline:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 5: High-precision polish with tight tolerance
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 300, 'disp': False, 'maxls': 150, 'gtol': 1e-14})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 6: Final local refinement with very tight tolerance
    if time.time() < deadline - 1.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'disp': False, 'maxls': 200, 'gtol': 1e-15})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Format output
    # p=2, so we need 2 rows of (gamma, beta)
    out = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
