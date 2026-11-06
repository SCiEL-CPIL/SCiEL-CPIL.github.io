import numpy as np
import time
from scipy.optimize import differential_evolution

# Global cache for graphs and optimal cuts
_graph_cache = {}

def _generate_graphs():
    if _graph_cache:
        return _graph_cache
    
    np.random.seed(713)
    configs = [
        (6, 0.25),
        (6, 0.55),
        (8, 0.25),
        (8, 0.55),
        (10, 0.25),
        (10, 0.55)
    ]
    
    graphs = []
    opt_cuts = []
    
    for n, density in configs:
        # Generate Erdos-Renyi graph
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if np.random.random() < density:
                    edges.append((i, j))
        
        # Compute optimal cut
        max_cut = 0
        for mask in range(1 << n):
            cut_size = 0
            for u, v in edges:
                if ((mask >> u) & 1) != ((mask >> v) & 1):
                    cut_size += 1
            if cut_size > max_cut:
                max_cut = cut_size
        
        graphs.append((n, edges))
        opt_cuts.append(max_cut)
        
    _graph_cache['graphs'] = graphs
    _graph_cache['opt_cuts'] = opt_cuts
    return _graph_cache

def _simulate_qaoa_fast(n, edges, params, cut_sizes, flip_indices):
    """
    Simulate QAOA for given graph and parameters.
    params: array of shape (p, 2) with (gamma, beta).
    Returns expected cut size.
    """
    p = len(params)
    N = 1 << n
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    for gamma, beta in params:
        # Apply Cost
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Apply Mixer
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        for i in range(n):
            # Use precomputed flip indices
            state = c * state + s * state[flip_indices[i]]
            
    probs = np.abs(state)**2
    return np.sum(probs * cut_sizes)

def angles(n, edges, p):
    start_time = time.time()
    deadline = 15.0 # 15 seconds per graph, total 90s for 6 graphs, safe margin
    
    # 1. Compute optimal cut for this graph
    N = 1 << n
    max_cut = 0
    # Precompute cut sizes
    cut_sizes = np.zeros(N, dtype=float)
    for mask in range(N):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        cut_sizes[mask] = cut
        if cut > max_cut:
            max_cut = cut
            
    if max_cut == 0:
        return np.zeros((p, 2))
        
    # Precompute flip indices for mixer
    flip_indices = []
    for i in range(n):
        flip_indices.append(np.arange(N) ^ (1 << i))

    # 2. Define local objective
    def local_obj(params):
        params_reshaped = params.reshape(p, 2)
        exp_cut = _simulate_qaoa_fast(n, edges, params_reshaped, cut_sizes, flip_indices)
        return -exp_cut / max_cut
        
    # 3. Global Optimization using Differential Evolution
    # Bounds: [0, pi] for all 4 parameters (gamma1, beta1, gamma2, beta2)
    bounds = [(0, np.pi)] * (2 * p)
    
    # Check time
    if time.time() - start_time > deadline:
        # Fallback to simple heuristic if we ran out of time (unlikely to happen before start)
        return np.array([[0.5, 0.5], [0.5, 0.5]])
        
    try:
        # Use differential_evolution
        # maxiter controls the number of generations
        # popsize controls the size of the population
        # tol is the tolerance for convergence
        # polish=True uses Nelder-Mead at the end for fine-tuning
        # We limit maxiter to ensure speed.
        result = differential_evolution(
            local_obj, 
            bounds, 
            maxiter=30, 
            popsize=15,
            seed=42,
            polish=True,
            tol=1e-4
        )
        
        best_params = result.x
    except Exception:
        # Fallback if optimization fails
        best_params = np.array([0.5, 0.5, 0.5, 0.5])
        
    # Check time again
    if time.time() - start_time > deadline:
        return best_params.reshape(p, 2)
        
    # Optional: If we have time left, do a quick local refinement?
    # DE with polish=True already does this.
    # Let's just return the result.
    
    return best_params.reshape(p, 2)
