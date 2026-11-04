"""Combined QAP solver: Vectorized 2-swap local search with Tabu memory + ILS with Or-opt perturbation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j in permutation p.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1: T1[i, j] = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # Corrections for T1:
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # C1[i, j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # C2[i, j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2: T2[i, j] = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # Corrections for T2:
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3: T3[i, j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    np.fill_diagonal(D, 0)
    
    return D

def _local_search_tabu(a, b, p, max_iter=None, time_limit=None, start_time=None, tabu_size=10):
    """
    Perform steepest descent local search with a tabu list to avoid cycling.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 2000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    tabu_list = {} # (i, j) -> expiration iter
    iter_count = 0
    improved = True
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Mask tabu moves
        # We want to find the min D[i,j] where (i,j) is not tabu.
        # If all improving moves are tabu, we take the best one (aspiration).
        
        best_val = float('inf')
        best_i, best_j = -1, -1
        best_val_aspiration = float('inf')
        best_i_asp, best_j_asp = -1, -1
        
        # Iterate over upper triangle to avoid duplicates
        # Vectorized approach:
        # Create a mask for tabu moves
        # Since tabu_list is small, we can just check the candidates.
        
        # Get top K candidates from D
        # Flatten D, get indices
        flat_D = D.flatten()
        # Sort indices by value
        # We only need the top few
        # Since D is symmetric, we can just look at all.
        
        # Efficient way:
        # Find all (i,j) with D[i,j] < 0
        # Check if they are in tabu_list
        
        improving_mask = D < 0
        # Get indices of improving moves
        rows, cols = np.where(improving_mask)
        
        if len(rows) > 0:
            # Sort by delta
            deltas = D[rows, cols]
            sorted_indices = np.argsort(deltas)
            
            # Check first few for tabu status
            found = False
            for idx in sorted_indices:
                i, j = rows[idx], cols[idx]
                # Normalize (i, j) so i < j for tabu key
                key = (min(i, j), max(i, j))
                
                if key not in tabu_list or tabu_list[key] <= iter_count:
                    best_val = deltas[idx]
                    best_i, best_j = i, j
                    found = True
                    break
                else:
                    # This is a tabu move, but maybe aspiration?
                    if deltas[idx] < best_val_aspiration:
                        best_val_aspiration = deltas[idx]
                        best_i_asp, best_j_asp = i, j
            
            if found:
                i, j = best_i, best_j
            elif best_i_asp != -1:
                # Aspiration: take the best tabu move if no non-tabu improving move exists
                i, j = best_i_asp, best_j_asp
                best_val = best_val_aspiration
            else:
                improved = False
                break
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += best_val
            
            # Add to tabu list
            key = (min(i, j), max(i, j))
            tabu_list[key] = iter_count + tabu_size
            
            iter_count += 1
        else:
            improved = False
            
    return p, current_cost

def _spectral_initial(a, b):
    """
    Spectral initialization using eigenvectors.
    """
    n = len(a)
    try:
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        facility_scores = v_a
        location_scores = v_b
        
        heavy_facilities = np.argsort(-facility_scores, kind="stable")
        central_locations = np.argsort(-location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        return p
    except Exception:
        # Fallback to rank matching
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        return p

def _oropt1_perturb(p, num_moves):
    """
    Perturb the permutation by performing num_moves Or-opt(1) moves.
    Or-opt(1): Move a single element to a random position.
    """
    n = len(p)
    p = p.copy()
    for _ in range(num_moves):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        # Move p[i] to p[j]
        x = p[i]
        # Shift elements
        if i < j:
            p[i:j] = p[i+1:j+1]
            p[j] = x
        else:
            p[j+1:i+1] = p[j:i]
            p[j] = x
    return p

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start ILS
    best_p = None
    best_cost = float('inf')
    
    # Initial solution from spectral method
    p = _spectral_initial(a, b)
    
    # Also try rank matching init
    p_rank = _spectral_initial(a, b) # Fallback is rank matching, but let's force rank matching for variety
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_facilities] = central_locations
    
    # Run local search on both, keep the best
    candidates = [p, p_rank]
    
    for init_p in candidates:
        if time.time() - start_time > total_deadline:
            break
        p_ls, cost_ls = _local_search_tabu(a, b, init_p, time_limit=total_deadline, start_time=start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    p = best_p.copy()
    current_cost = best_cost
    
    # ILS Loop
    while time.time() - start_time < total_deadline:
        # Perturb
        # Scale perturbation size
        num_moves = max(2, n // 5)
        p = _oropt1_perturb(p, num_moves)
        
        # Local Search
        p, current_cost = _local_search_tabu(a, b, p, time_limit=total_deadline, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
