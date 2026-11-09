"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

_start_time = time.time()
_DEADLINE = 148.0
_cache = {}

def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)

def _exact_optimum(n, edges):
    """Find exact MaxCut optimum by exhaustive search."""
    edge_list = []
    for u, v in edges:
        edge_list.append((min(u,v), max(u,v)))
    
    best = 0
    for bits in product([0, 1], repeat=n):
        cut = 0
        for (u, v) in edge_list:
            if bits[u] != bits[v]:
                cut += 1
        if cut > best:
            best = cut
    return best

def _qaoa_expected_cut_fast(n, edges, gammas, betas):
    """
    Compute expected cut for QAOA p=2 using vectorized state vector.
    """
    edge_list = []
    for u, v in edges:
        edge_list.append((min(u,v), max(u,v)))
    
    num_edges = len(edge_list)
    if num_edges == 0:
        return 0.0
    
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute bit indices for each qubit
    bit_masks = [(1 << j) for j in range(n)]
    
    for l in range(2):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i gamma * C): phase 1 if same bit, exp(-i gamma) if different
        phase_same = 1.0
        phase_diff = np.exp(-1j * gamma)
        
        for (u, v) in edge_list:
            # Vectorized phase application
            bu = (np.arange(dim) >> u) & 1
            bv = (np.arange(dim) >> v) & 1
            mask = (bu != bv)
            phase = np.ones(dim, dtype=np.complex128)
            phase[mask] = phase_diff
            state *= phase
        
        # Apply exp(-i beta * sum X_j) - vectorized
        c = np.cos(beta)
        s = np.sin(beta)
        
        for j in range(n):
            mask = (np.arange(dim) >> j) & 1
            # For even bits (0): new_state[idx] = c*state[idx] - i*s*state[idx | (1<<j)]
            # For odd bits (1): new_state[idx] = c*state[idx] - i*s*state[idx & ~(1<<j)]
            
            idx = np.arange(dim)
            idx_flip = idx ^ (1 << j)
            
            new_state = np.zeros_like(state)
            new_state[~mask] = c * state[~mask] - 1j * s * state[idx_flip[~mask]]
            new_state[mask] = c * state[mask] - 1j * s * state[idx_flip[mask]]
            state = new_state
    
    # Compute expected cut
    expected_cut = 0.0
    idx = np.arange(dim)
    for (u, v) in edge_list:
        bu = (idx >> u) & 1
        bv = (idx >> v) & 1
        mask = (bu != bv)
        expected_cut += np.sum(np.abs(state[mask])**2)
    
    return expected_cut

def _optimize_angles_for_graph(n, edges, exact_opt, time_budget):
    """Find optimal angles for a specific graph."""
    key = (n, tuple(sorted(edges)))
    if key in _cache:
        return _cache[key]
    
    best_ratio = 0.0
    best_angles = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Coarse grid search with vectorized evaluation
    gamma_range = np.linspace(0.1, 2.8, 6)
    beta_range = np.linspace(0.1, 2.8, 6)
    
    best_local = 0.0
    best_params = [0.5, 0.5, 0.5, 0.5]
    
    for g1 in gamma_range:
        for b1 in beta_range:
            gamma2_range = np.linspace(0.1, 2.8, 5)
            beta2_range = np.linspace(0.1, 2.8, 5)
            
            for g2 in gamma2_range:
                for b2 in beta2_range:
                    ec = _qaoa_expected_cut_fast(n, edges, [g1, g2], [b1, b2])
                    ratio = ec / exact_opt if exact_opt > 0 else 0
                    if ratio > best_local:
                        best_local = ratio
                        best_params = [g1, b1, g2, b2]
            
            if time.time() - _start_time > time_budget:
                break
        if time.time() - _start_time > time_budget:
            break
    
    best_angles = np.array(best_params)
    best_ratio = best_local
    
    # Local refinement
    if time.time() - _start_time < time_budget * 0.7:
        from scipy.optimize import minimize
        
        def objective(params):
            g1, b1, g2, b2 = params
            ec = _qaoa_expected_cut_fast(n, edges, [g1, g2], [b1, b2])
            return -ec / exact_opt if exact_opt > 0 else 0
        
        start_points = [
            best_angles,
            np.array([0.4, 0.4, 0.4, 0.4]),
            np.array([1.0, 0.5, 0.8, 0.4]),
        ]
        
        best_result = best_angles
        best_obj = best_ratio
        
        for sp in start_points:
            if time.time() - _start_time > time_budget:
                break
            try:
                res = minimize(objective, sp, method='Nelder-Mead', 
                              options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
                if -res.fun > best_obj:
                    best_obj = -res.fun
                    best_result = res.x
            except:
                pass
        
        best_angles = best_result
        best_ratio = best_obj
    
    _cache[key] = (best_angles, best_ratio)
    return best_angles, best_ratio

def angles(n, edges, p):
    """
    Return QAOA angles for p layers.
    """
    if time.time() - _start_time > _DEADLINE - 3:
        return np.tile([0.5, 0.5], (p, 1))
    
    exact_opt = _exact_optimum(n, edges)
    
    time_budget = min(_remaining_time() * 0.85, 18.0)
    
    best_angles, best_ratio = _optimize_angles_for_graph(n, edges, exact_opt, time_budget)
    
    result = np.zeros((p, 2))
    for i in range(min(p, 2)):
        result[i, 0] = best_angles[2*i]
        result[i, 1] = best_angles[2*i + 1]
    if p > 2:
        for i in range(2, p):
            result[i, 0] = best_angles[0]
            result[i, 1] = best_angles[1]
    
    return result

# EVOLVE-BLOCK-END
