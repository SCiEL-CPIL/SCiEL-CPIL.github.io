import numpy as np
import time
from scipy.optimize import minimize

_GLOBAL_DEADLINE = None

def _simulate_qaoa_fast(n, edges, params):
    p = len(params)
    N = 1 << n
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    # Precompute cut sizes
    cut_sizes = np.zeros(N, dtype=float)
    for mask in range(N):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        cut_sizes[mask] = cut
        
    indices = np.arange(N)
    
    for gamma, beta in params:
        state *= np.exp(-1j * gamma * cut_sizes)
        
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        for i in range(n):
            flipped_indices = indices ^ (1 << i)
            state = c * state + s * state[flipped_indices]
            
    probs = np.abs(state)**2
    return np.sum(probs * cut_sizes)

def angles(n, edges, p):
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 110.0 # 110s total for all 6 graphs
        
    start_time = time.time()
    # Per-graph deadline: assume 6 graphs, so 110/6 ~ 18s each.
    # But if previous graphs were fast, we can use more time.
    # Simple strategy: 15s per graph.
    graph_deadline = start_time + 15.0
    if graph_deadline > _GLOBAL_DEADLINE:
        graph_deadline = _GLOBAL_DEADLINE - 1.0 # 1s safety margin
        
    N = 1 << n
    max_cut = 0
    for mask in range(N):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > max_cut:
            max_cut = cut
            
    if max_cut == 0:
        return np.zeros((p, 2))
        
    def local_obj(params):
        params_reshaped = params.reshape(p, 2)
        exp_cut = _simulate_qaoa_fast(n, edges, params_reshaped)
        return -exp_cut / max_cut
        
    # Grid Search
    grid_size = 8
    grid_points = np.linspace(0, np.pi, grid_size)
    
    best_params = None
    best_score = -np.inf
    
    for g1 in grid_points:
        for b1 in grid_points:
            for g2 in grid_points:
                for b2 in grid_points:
                    params = np.array([g1, b1, g2, b2])
                    score = -local_obj(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    if time.time() > graph_deadline:
                        return best_params.reshape(p, 2)
                        
    # Local Refinement
    x0 = best_params
    if time.time() < graph_deadline:
        result = minimize(local_obj, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.success and -result.fun > best_score:
            best_params = result.x
            best_score = -result.fun
            
    # Multi-start
    np.random.seed(42)
    for _ in range(3):
        if time.time() > graph_deadline:
            break
        x_rand = np.random.uniform(0, np.pi, 4)
        result = minimize(local_obj, x_rand, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.success and -result.fun > best_score:
            best_params = result.x
            best_score = -result.fun
            
    return best_params.reshape(p, 2)
