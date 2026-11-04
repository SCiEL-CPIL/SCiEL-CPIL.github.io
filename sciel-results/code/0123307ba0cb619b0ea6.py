"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)


def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    p[i] is the location of facility i.
    """
    li = p[i]
    lj = p[j]
    
    # Precompute difference vectors for locations li and lj
    # diff_row[k] = b[lj, k] - b[li, k]
    # diff_col[k] = b[k, lj] - b[k, li]
    # We need to evaluate these at the locations p[k]
    
    # Vectorized sums
    # S1 = sum_{k} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    # S2 = sum_{k} a[k,i] * (b[p[k], lj] - b[p[k], li])
    # S3 = sum_{k} a[j,k] * (b[li, p[k]] - b[lj, p[k]]) = - sum_{k} a[j,k] * (b[lj, p[k]] - b[li, p[k]])
    # S4 = sum_{k} a[k,j] * (b[p[k], li] - b[p[k], lj]) = - sum_{k} a[k,j] * (b[p[k], lj] - b[p[k], li])
    
    # Let D_row_p = b[lj, p] - b[li, p]
    # Let D_col_p = b[p, lj] - b[p, li]
    
    D_row_p = b[lj, p] - b[li, p]
    D_col_p = b[p, lj] - b[p, li]
    
    # Full sums
    S1_full = np.dot(a[i, :], D_row_p)
    S2_full = np.dot(a[:, i], D_col_p)
    S3_full = -np.dot(a[j, :], D_row_p)
    S4_full = -np.dot(a[:, j], D_col_p)
    
    # Subtract contributions from k=i and k=j
    # For S1 (a[i,k] * D_row_p[k]):
    # k=i: a[i,i] * (b[lj, li] - b[li, li])
    # k=j: a[i,j] * (b[lj, lj] - b[li, lj])
    s1_sub = a[i, i] * (b[lj, li] - b[li, li]) + a[i, j] * (b[lj, lj] - b[li, lj])
    
    # For S2 (a[k,i] * D_col_p[k]):
    # k=i: a[i,i] * (b[li, lj] - b[li, li])
    # k=j: a[j,i] * (b[lj, lj] - b[lj, li])
    s2_sub = a[i, i] * (b[li, lj] - b[li, li]) + a[j, i] * (b[lj, lj] - b[lj, li])
    
    # For S3 (-a[j,k] * D_row_p[k]):
    # k=i: -a[j,i] * (b[lj, li] - b[li, li])
    # k=j: -a[j,j] * (b[lj, lj] - b[li, lj])
    s3_sub = -a[j, i] * (b[lj, li] - b[li, li]) - a[j, j] * (b[lj, lj] - b[li, lj])
    
    # For S4 (-a[k,j] * D_col_p[k]):
    # k=i: -a[i,j] * (b[li, lj] - b[li, li])
    # k=j: -a[j,j] * (b[lj, lj] - b[lj, li])
    s4_sub = -a[i, j] * (b[li, lj] - b[li, li]) - a[j, j] * (b[lj, lj] - b[lj, li])
    
    s1 = S1_full - s1_sub
    s2 = S2_full - s2_sub
    s3 = S3_full - s3_sub
    s4 = S4_full - s4_sub
    
    # Diagonal terms (changes in a[i,i], a[j,j], a[i,j], a[j,i] terms)
    delta_diag = a[i, i] * (b[lj, lj] - b[li, li]) + \
                 a[j, j] * (b[li, li] - b[lj, lj]) + \
                 a[i, j] * (b[lj, li] - b[li, lj]) + \
                 a[j, i] * (b[li, lj] - b[lj, li])
    
    return delta_diag + s1 + s2 + s3 + s4


def solve(a, b):
    n = len(a)
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    max_time = 115.0
    
    # Tabu Search Parameters
    tabu_length = 50
    tabu_list = np.zeros((n, n), dtype=bool)
    iteration = 0
    
    i_list, j_list = np.triu_indices(n, k=1)
    num_pairs = len(i_list)
    
    while (time.time() - start_time) < max_time:
        # Shuffle pairs for diversity
        perm = np.random.permutation(num_pairs)
        i_list_shuffled = i_list[perm]
        j_list_shuffled = j_list[perm]
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        best_is_tabu = False
        
        # Evaluate a subset of moves to save time, or all if fast enough
        # With n=256, 32k moves * 256 ops ~ 8M ops. 
        # Let's evaluate all moves in one pass.
        for idx in range(num_pairs):
            if (time.time() - start_time) > max_time:
                break
            i = i_list_shuffled[idx]
            j = j_list_shuffled[idx]
            
            is_tabu = tabu_list[i, j] or tabu_list[j, i]
            
            delta = _swap_cost_delta(a, b, p, i, j)
            
            # Aspiration criterion: accept tabu move if it improves global best
            if not is_tabu or (current_cost + delta < best_cost):
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    best_is_tabu = is_tabu
        
        if best_i == -1:
            break
            
        # Update tabu list
        # Remove old tabu entries (simple decay or shift)
        # For simplicity, we just set the current move as tabu and let others expire by overwriting
        # A better way is a queue, but for speed, we use a counter or just random reset
        # Here we use a simple strategy: mark the move as tabu for tabu_length iterations
        # We need a 3D array or a list of (i, j, expiry)
        # To keep it fast, we'll use a 2D array of expiry times
        if not hasattr(solve, '_tabu_expiry'):
            solve._tabu_expiry = np.zeros((n, n), dtype=int)
        
        solve._tabu_expiry[best_i, best_j] = iteration + tabu_length
        solve._tabu_expiry[best_j, best_i] = iteration + tabu_length
        
        # Perform the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        iteration += 1
        
        # Update tabu status for next iteration's check? 
        # The check `tabu_list[i, j]` in the loop above uses `tabu_list` which is not updated.
        # We need to update `tabu_list` or check `solve._tabu_expiry[i, j] > iteration`.
        # Let's modify the loop to check expiry.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
