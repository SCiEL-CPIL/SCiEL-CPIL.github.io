"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh
import time


def _spectral_init(a, b):
    """
    Spectral initialization: align high-flow facilities with high-distance locations.
    We use the principal eigenvectors of the symmetrized matrices.
    """
    n = a.shape[0]
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Principal eigenvector (largest eigenvalue)
        _, eigvecs_a = eigh(a_sym)
        v_a = eigvecs_a[:, -1]
        
        _, eigvecs_b = eigh(b_sym)
        v_b = eigvecs_b[:, -1]
        
        # Align high values in v_a with high values in v_b
        facility_rank = np.argsort(-v_a)
        location_rank = np.argsort(-v_b)
        
        p = np.empty(n, dtype=np.int64)
        p[facility_rank] = location_rank
    except Exception:
        p = np.arange(n)
        
    return p


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)


def _local_search(a, b, p, max_iters=1000):
    """
    Fast 2-swap local search using vectorized delta calculations.
    """
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    
    # Precompute constants for delta calculation
    # M = b[p, p]
    # E = row sums of a * M (element-wise)
    # AM = a @ M
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        M = b[np.ix_(p, p)]
        E = np.sum(a * M, axis=1)
        AM = a @ M
        
        # We search for the best swap (i, j) with i < j
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            # Calculate delta for swapping i with all j > i
            # Formula derived in thought process:
            # Delta_ij = (a_ii - a_jj)(M_jj - M_ii) + (E_i - 2(AM)_ij + E_j) 
            #            - (a_ii - a_ij)(M_ii - M_ji) - (a_ij - a_jj)(M_ij - M_jj)
            
            # Vectorized for all j
            js = np.arange(i + 1, n)
            if len(js) == 0: continue
            
            a_ii = a[i, i]
            a_jj = a[js, js] # Diagonal elements for j in js
            a_ij = a[i, js]
            
            M_ii = M[i, i]
            M_jj = M[js, js]
            M_ij = M[i, js]
            M_ji = M[js, i] # Same as M_ij if symmetric, but M is symmetric so M_ji = M_ij
            
            E_i = E[i]
            E_j = E[js]
            AM_ij = AM[i, js]
            
            term1 = (a_ii - a_jj) * (M_jj - M_ii)
            term2 = (E_i - 2 * AM_ij + E_j)
            term3 = (a_ii - a_ij) * (M_ii - M_ji)
            term4 = (a_ij - a_jj) * (M_ij - M_jj)
            
            deltas = term1 + term2 - term3 - term4
            
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_j = js[min_idx]
                best_i = i
        
        if best_i != -1 and best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
            
    return p, current_cost


def _iterated_local_search(a, b, p_init, time_budget=110):
    """
    Iterated Local Search with fast local search.
    """
    start_time = time.time()
    n = len(p_init)
    
    p_best = p_init.copy()
    cost_best = _compute_cost(a, b, p_best)
    p_current = p_best.copy()
    
    while time.time() - start_time < time_budget:
        # Local search
        p_current, cost_current = _local_search(a, b, p_current)
        
        if cost_current < cost_best:
            cost_best = cost_current
            p_best = p_current.copy()
        else:
            # If no improvement, we might have stuck in a local opt.
            # Perturb
            num_swaps = 2
            for _ in range(num_swaps):
                i, j = np.random.choice(n, size=2, replace=False)
                p_current[i], p_current[j] = p_current[j], p_current[i]
                
    return p_best, cost_best


def solve(a, b):
    n = a.shape[0]
    p_spectral = _spectral_init(a, b)
    p_best, cost_best = _iterated_local_search(a, b, p_spectral)
    return p_best.tolist()
# EVOLVE-BLOCK-END
