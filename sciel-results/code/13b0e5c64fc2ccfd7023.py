import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    def get_cost(p):
        # p is permutation of locations for facilities 0..n-1
        # Cost = sum a[i,j] b[p[i], p[j]]
        # Vectorized: trace(A @ B_p)? No.
        # sum_i sum_j A[i,j] B[p[i], p[j]]
        # Let B_p be B with rows permuted by p.
        # Then sum_i sum_j A[i,j] B_p[i, j]? No.
        # B[p[i], p[j]] is element (i,j) of matrix where row i is B[p[i], :]
        # So B_perm = B[p, :]
        # Cost = sum(A * B_perm)
        return np.sum(A * B[p, :])

    def local_search(p):
        # p is list or array
        p = np.array(p, dtype=np.int64)
        best_p = p.copy()
        best_cost = get_cost(best_p)
        
        while time.time() - start_time < time_limit:
            # Compute R = A @ B_p
            # B_p[u, k] = B[u, p[k]]
            B_p = B[:, p] # Rows of B indexed by p?
            # Wait. B_p[i, j] should be B[p[i], p[j]]?
            # My derivation: R[i, u] = sum_k A[i, k] B[u, p[k]].
            # This is (A @ B_perm)[i, u] where B_perm[u, k] = B[u, p[k]].
            # So B_perm = B[:, p] (columns of B permuted by p).
            
            R = A @ B[:, p]
            
            # C[i] = R[i, p[i]]
            C = R[np.arange(n), p]
            
            # M[i, j] = R[i, p[j]] + R[j, p[i]] - R[i, p[i]] - R[j, p[j]]
            # R[i, p[j]] is element (i, j) of R[:, p]
            R_col = R[:, p]
            
            M = R_col + R_col.T - C[:, None] - C[None, :]
            
            # Set diagonal to inf
            np.fill_diagonal(M, np.inf)
            
            # Find min
            min_val = np.min(M)
            
            if min_val >= -1e-9: # No improvement
                break
            
            # Get indices
            i, j = np.unravel_index(np.argmin(M), M.shape)
            
            # Swap p[i] and p[j]
            p[i], p[j] = p[j], p[i]
            
            # Update cost
            new_cost = get_cost(p)
            # Due to float precision, use calculated delta or recompute?
            # Recompute is safe.
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
            else:
                # Should not happen if delta was correct, but float errors?
                # Revert?
                p[i], p[j] = p[j], p[i]
                break # If delta said improve but it didn't, stop to be safe?
        
        return best_p, best_cost

    # Main Loop: ILS
    best_p = None
    best_cost = np.inf
    
    # Initial random
    p = np.random.permutation(n)
    p, cost = local_search(p)
    best_p = p
    best_cost = cost
    
    # ILS
    while time.time() - start_time < time_limit:
        # Perturb
        p_cur = best_p.copy()
        # Random swaps
        for _ in range(5): # Perturbation strength
            i, j = np.random.choice(n, 2, replace=False)
            p_cur[i], p_cur[j] = p_cur[j], p_cur[i]
        
        # Local search
        p_new, cost_new = local_search(p_cur)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new
        elif cost_new == best_cost:
            # Accept equal? No, usually accept worse to escape.
            # Simulated Annealing style?
            # For now, just keep best.
            pass
    
    return best_p.tolist()
