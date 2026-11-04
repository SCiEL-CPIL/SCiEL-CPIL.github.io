"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def compute_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute for speed
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    # --- Phase 1: WL-inspired spectral initialization ---
    # Compute eigenvectors of both matrices to get a spectral embedding
    # This captures higher-order topological structure
    
    # For the facility matrix (flow), use the dominant eigenvectors
    # For the location matrix (distance), use the dominant eigenvectors
    # Then match based on the spectral coordinates
    
    # Compute top eigenvectors of a (flow matrix)
    # Use symmetric part for stability
    a_sym = (a_float + a_float.T) / 2.0
    b_sym = (b_float + b_float.T) / 2.0
    
    # Number of eigenvectors to use for matching
    k_eig = min(20, n)
    
    try:
        # Eigendecomposition of facility matrix
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        # Sort by absolute value (take largest magnitude)
        idx_a = np.argsort(-np.abs(eigvals_a))[:k_eig]
        U_a = eigvecs_a[:, idx_a]
        
        # Eigendecomposition of location matrix
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        idx_b = np.argsort(-np.abs(eigvals_b))[:k_eig]
        U_b = eigvecs_b[:, idx_b]
        
        # Match facilities to locations based on spectral coordinates
        # Facility i should go to location j if their spectral embeddings are similar
        # But we want to minimize cost, so heavy flow facilities should go to 
        # locations that are "central" (low distance to others)
        
        # Alternative: use the eigenvector components to create a cost matrix
        # for the Hungarian algorithm
        # cost[i, j] = -dot(U_a[i], U_b[j]) for maximum alignment
        # But actually we want to minimize the QAP cost, so let's think differently
        
        # Use spectral coordinates to create an initial assignment
        # that aligns the "structure" of both graphs
        spectral_cost = -np.dot(U_a, U_b.T)
        _, p_spectral = linear_sum_assignment(spectral_cost)
        p_init = p_spectral.tolist()
        
    except Exception:
        # Fallback to rank-matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p_init = np.empty(n, dtype=np.int64)
        p_init[heavy] = central
        p_init = p_init.tolist()
    
    # --- Phase 2: Hungarian initialization with refined cost matrix ---
    # Create a cost matrix that approximates the QAP cost
    # For each facility i and location j, estimate the cost contribution
    # c[i,j] = sum_k a[i,k] * b[j, p[k]]  (but we don't know p yet)
    # Approximate: c[i,j] = sum_k a[i,k] * b[j, k]  (identity assignment as proxy)
    # Better: use the current best assignment to refine
    
    # Start with spectral initialization
    p = p_init
    current_cost = compute_cost(a, b, p)
    best_cost = current_cost
    best_p = p[:]
    
    # Refine with Hungarian: use the current assignment to build a cost matrix
    # c[i,j] = sum_k a[i,k] * b[j, p[k]]
    # This is the cost of assigning facility i to location j, given others stay
    # Solve assignment problem on this cost matrix
    
    def _hungarian_refine(p_arr):
        """Refine assignment using Hungarian algorithm with current cost estimates."""
        # Build cost matrix: c[i,j] = sum_k a[i,k] * b[j, p_arr[k]]
        # This is O(n^3) but we do it only a few times
        # c[i,j] = sum_k a[i,k] * b[j, p[k]]
        # = (a @ b[:, p])[i, j]  ... wait, let me think
        # For fixed i and j: sum over k of a[i,k] * b[j, p[k]]
        # Let bp = b[:, p]  (column j of bp is b[:, p[j]])
        # Actually: sum_k a[i,k] * b[j, p[k]] = sum_k a[i,k] * (b[:, p[k]])[j]
        # = sum_k a[i,k] * bp[j, k]  where bp = b[np.ix_(np.arange(n), p)]
        # So c[i,j] = (a @ bp.T)[i, j]
        
        bp = b_float[np.ix_(np.arange(n), p_arr)]
        cost_matrix = a_float @ bp.T
        _, p_new = linear_sum_assignment(cost_matrix)
        return p_new.tolist()
    
    # Do a few rounds of Hungarian refinement
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        p = _hungarian_refine(np.array(p, dtype=np.int64))
        cost = compute_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p[:]
    
    # --- Phase 3: Fast local search with swap moves ---
    p_arr = np.array(best_p, dtype=np.int64)
    current_cost = best_cost
    
    # Precompute for fast swap delta
    # delta_swap(i, j) = (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    #                   + sum_{k!=i,j} [(a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    #                                  + (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])]
    
    def _swap_delta_fast(p_arr, i, j):
        """Fast swap delta using numpy."""
        li = p_arr[i]
        lj = p_arr[j]
        
        # Cross terms
        delta = (a[i, j] - a[j, i]) * (b[lj, li] - b[li, lj])
        
        others_mask = np.ones(n, dtype=bool)
        others_mask[i] = False
        others_mask[j] = False
        others_idx = np.nonzero(others_mask)[0]
        
        if len(others_idx) > 0:
            pk_others = p_arr[others_idx]
            ai_jk = a[i, others_idx] - a[j, others_idx]
            aki_kj = a[others_idx, i] - a[others_idx, j]
            blj_bli = b[lj, pk_others] - b[li, pk_others]
            bpk_blj = b[pk_others, lj] - b[pk_others, li]
            delta += np.sum(ai_jk * blj_bli + aki_kj * bpk_blj)
        
        return int(delta)
    
    # Local search with simulated annealing
    temperature = max(current_cost * 0.001, 10.0)
    cool_rate = 0.999
    min_temp = 0.01
    
    steps_since_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline:
        # Pick random pair to swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        d = _swap_delta_fast(p_arr, i, j)
        
        # Accept if improvement or with probability exp(-d/T)
        if d < 0 or (temperature > min_temp and np.random.random() < np.exp(-d / temperature)):
            # Perform swap
            p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
            current_cost += d
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p_arr.tolist()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
            
            # Cool down
            temperature *= cool_rate
            if temperature < min_temp:
                temperature = min_temp
        else:
            steps_since_improve += 1
        
        if steps_since_improve > max_no_improve:
            # Reset: restart from best or reheat
            if time.time() > deadline - 10:
                break
            p_arr = np.array(best_p, dtype=np.int64)
            current_cost = best_cost
            temperature = max(best_cost * 0.002, 10.0)
            steps_since_improve = 0
    
    return best_p
# EVOLVE-BLOCK-END
