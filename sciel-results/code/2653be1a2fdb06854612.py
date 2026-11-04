import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial solution: Greedy or Random
    # Let's use a simple greedy construction to start better than random
    # Or just random permutation for SA to explore
    p = np.random.permutation(n)
    
    # Compute S: S[i, l] = sum_k a[i, k] * b[l, p[k]]
    # b[:, p] is b with columns permuted
    B_perm = b[:, p]
    S = a @ B_perm
    
    # Current cost
    # C = sum_i S[i, p[i]] - sum_i a[i, i] b[p[i], p[i]]?
    # Since a_ii=0, b_ll=0, C = sum_i S[i, p[i]]
    # Wait, S[i, p[i]] = sum_k a[i,k] b[p[i], p[k]].
    # Sum_i S[i, p[i]] = sum_i sum_k a[i,k] b[p[i], p[k]] = C.
    current_cost = np.sum(S[np.arange(n), p])
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    steps_per_check = 1000
    
    # Precompute indices
    idx = np.arange(n)
    
    for step in range(1000000):
        if time.time() > deadline:
            break
        
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        li = p[i]
        lj = p[j]
        
        # Compute delta
        # Delta = (S[i, lj] - S[i, li]) - (S[j, lj] - S[j, li])
        delta = (S[i, lj] - S[i, li]) - (S[j, lj] - S[j, li])
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            if T == 0:
                accept = False
            else:
                # exp(-delta/T)
                # If delta is large, exp is 0
                try:
                    prob = np.exp(-delta / T)
                except OverflowError:
                    prob = 0.0
                accept = np.random.rand() < prob
        
        if accept:
            # Update p
            p[i], p[j] = p[j], p[i]
            
            # Update S
            # S += (a[:, i] - a[:, j])[:, np.newaxis] * (b[:, lj] - b[:, li])[np.newaxis, :]
            diff_a = a[:, i] - a[:, j]
            diff_b = b[:, lj] - b[:, li]
            S += np.outer(diff_a, diff_b)
            
            # Update cost
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
        if T < 1e-5:
            T = 1e-5 # Floor to prevent underflow issues, or break if we want deterministic polish
            
        if step % steps_per_check == 0:
            # Optional: Deterministic 2-opt polish every so often?
            # Might be too slow if done frequently.
            pass

    # Final Polish: Deterministic 2-opt (Best Improvement)
    # Run until no improvement found
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find best improving move
        # We can vectorize the search for best move?
        # Delta_ij = (S[i, lj] - S[i, li]) - (S[j, lj] - S[j, li])
        # This is tricky to fully vectorize for all pairs at once without O(N^3) memory.
        # But N=100, so O(N^2) loop is fine.
        
        for i in range(n):
            li = p[i]
            # For all j != i
            # We want to minimize delta_ij
            # delta_ij = S[i, p[j]] - S[i, li] - S[j, p[j]] + S[j, li]
            # Let's compute for all j at once for fixed i
            # S[i, p] is a vector of S[i, p[j]] for all j
            # S[:, p] is a matrix? No.
            # S[j, p[j]] is a vector for all j.
            # S[j, li] is a vector for all j.
            
            # vector_S_i_p = S[i, p]
            # vector_S_j_pj = S[np.arange(n), p]
            # vector_S_j_li = S[:, li]
            
            # delta_vec = S[i, p] - S[i, li] - S[idx, p] + S[:, li]
            
            delta_vec = S[i, p] - S[i, li] - S[idx, p] + S[:, li]
            delta_vec[i] = np.inf # Exclude i
            
            min_idx = np.argmin(delta_vec)
            min_val = delta_vec[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_i = i
                best_j = min_idx
        
        if best_delta < 0:
            i = best_i
            j = best_j
            li = p[i]
            lj = p[j]
            
            # Update
            p[i], p[j] = p[j], p[i]
            diff_a = a[:, i] - a[:, j]
            diff_b = b[:, lj] - b[:, li]
            S += np.outer(diff_a, diff_b)
            current_cost += best_delta
            best_cost = current_cost
            best_p = p.copy()
            improved = True
        else:
            improved = False

    return best_p.tolist()
