"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """
    Solve the Quadratic Assignment Problem (QAP) for the given matrices a and b.
    Returns a permutation p of 0..n-1 that minimizes sum_{i,j} a[i,j] * b[p[i], p[j]].
    """
    n = len(a)
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin

    # Precompute some constants if needed, but a and b are passed.
    # Ensure a and b are float64 for speed in dot products
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    def compute_cost(p):
        # p is a permutation array of indices into b
        # Cost is sum(a[i,j] * b[p[i], p[j]])
        # This is trace(A * B_p) where B_p is permuted B
        # b[np.ix_(p, p)] creates the permuted matrix
        b_sub = b[np.ix_(p, p)]
        return np.sum(a * b_sub)

    def get_delta(p, i, j):
        """
        Compute delta for swapping p[i] and p[j].
        p[i] = li, p[j] = lj.
        We swap the locations of facility i and facility j.
        """
        li, lj = p[i], p[j]
        # The change in cost is:
        # Old terms involving i or j:
        # sum_k A[i,k]B[li,p[k]] + sum_k A[k,i]B[p[k],li] + sum_k A[j,k]B[lj,p[k]] + sum_k A[k,j]B[p[k],lj]
        # + A[i,j]B[li,lj] + A[j,i]B[lj,li]
        #
        # New terms:
        # sum_k A[i,k]B[lj,p[k]] + sum_k A[k,i]B[p[k],lj] + sum_k A[j,k]B[li,p[k]] + sum_k A[k,j]B[p[k],li]
        # + A[i,j]B[lj,li] + A[j,i]B[li,lj]
        #
        # Delta = New - Old
        #
        # Vectorized:
        # Let diff_a_row = a[i] - a[j]
        # Let diff_a_col = a[:, i] - a[:, j]
        # Let diff_b_row = b[li, :] - b[lj, :]  (Note: b[lj, p[k]] - b[li, p[k]] is b[lj, p] - b[li, p])
        # Let diff_b_col = b[:, lj] - b[:, li]  (Note: b[p[k], lj] - b[p[k], li] is b[p, lj] - b[p, li])
        
        # Term 1: sum_k (A[i,k] - A[j,k]) * (B[lj, p[k]] - B[li, p[k]])
        # This is dot(a[i]-a[j], b[lj,p]-b[li,p])
        # Term 2: sum_k (A[k,i] - A[k,j]) * (B[p[k], lj] - B[p[k], li])
        # This is dot(a[:,i]-a[:,j], b[p,lj]-b[p,li])
        
        # We must be careful with k=i and k=j in the sums.
        # The formula above sums over all k.
        # If k=i:
        #   Term 1 part: (A[i,i]-A[j,i]) * (B[lj, p[i]] - B[li, p[i]]) = (A[i,i]-A[j,i]) * (B[lj, li] - B[li, li])
        #   Term 2 part: (A[i,i]-A[i,j]) * (B[p[i], lj] - B[p[i], li]) = (A[i,i]-A[i,j]) * (B[li, lj] - B[li, li])
        # If k=j:
        #   Term 1 part: (A[i,j]-A[j,j]) * (B[lj, p[j]] - B[li, p[j]]) = (A[i,j]-A[j,j]) * (B[lj, lj] - B[li, lj])
        #   Term 2 part: (A[j,i]-A[j,j]) * (B[p[j], lj] - B[p[j], li]) = (A[j,i]-A[j,j]) * (B[lj, lj] - B[lj, li])
        
        # It is safer to just compute the full delta using the vectorized dot products
        # and then adjust for the diagonal/self terms if the dot product includes them incorrectly?
        # Actually, the derivation:
        # C_new - C_old
        # = sum_{k!=i,j} (A[i,k]-A[j,k])B[lj,p[k]] + ... 
        # The vectorized dot product includes k=i and k=j.
        # Let's verify if the vectorized formula holds for k=i,j.
        
        # Let's just use the robust vectorized calculation which is known to work if implemented correctly.
        # The formula:
        # delta = dot(a[i]-a[j], b[lj,p]-b[li,p]) + dot(a[:,i]-a[:,j], b[p,lj]-b[p,li])
        # This works for all k.
        
        diff_a_row = a[i] - a[j]
        diff_b_col = b[lj, p] - b[li, p]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[p, lj] - b[p, li]
        
        return np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)

    def local_search_2opt(p, current_cost, time_limit):
        """
        Steepest descent 2-opt with vectorized delta.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # To speed up, we can sample or just do full scan.
            # Full scan is O(n^2) delta calcs. Each is O(n). Total O(n^3) per pass.
            # 150^3 = 3.3M ops. Very fast.
            
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        
        return p, current_cost

    # --- Heuristic: Greedy Construction ---
    def greedy_construction():
        """
        Constructs a solution by greedily adding facilities to locations.
        """
        p = np.full(n, -1, dtype=np.int64)
        assigned_facs = np.zeros(n, dtype=bool)
        assigned_locs = np.zeros(n, dtype=bool)
        
        # marginal[i][j] = cost increase if we assign facility i to location j
        # given current assignments.
        # Initially 0.
        marginal = np.zeros((n, n), dtype=np.float64)
        
        # We need to update marginal when we make an assignment.
        # If we assign f to l:
        # For any unassigned i, j:
        # marginal[i][j] += A[i,f]B[j,l] + A[f,i]B[l,j]
        
        # Note: The term A[i,i]B[l,l] is not in marginal initially?
        # The total cost is sum A[u,v]B[p[u],p[v]].
        # When we add f at l, we add:
        # 1. A[f,f]B[l,l]
        # 2. sum_{u assigned} (A[f,u]B[l,p[u]] + A[u,f]B[p[u],l])
        
        # So marginal[i][j] should track the cross terms.
        # Let's initialize marginal with 0.
        # When assigning f to l:
        #   cost += A[f,f]*B[l,l] + marginal[f][l]
        #   Update marginal for all i,j:
        #   marginal[i][j] += A[i,f]*B[j,l] + A[f,i]*B[l,j]
        
        for _ in range(n):
            # Find best (i, j) where i not assigned, j not assigned
            # Mask assigned
            mask = np.zeros((n, n), dtype=bool)
            mask[:, assigned_locs] = True
            mask[assigned_facs, :] = True
            
            # We want to minimize marginal[i][j]
            # Set masked values to inf
            m = marginal.copy()
            m[mask] = np.inf
            
            # Find min
            idx = np.unravel_index(np.argmin(m), m.shape)
            i, j = idx[0], idx[1]
            
            # Assign i to j
            p[i] = j
            assigned_facs[i] = True
            assigned_locs[j] = True
            
            # Update marginal
            # marginal[new_i][new_j] += A[new_i, i] * B[new_j, j] + A[i, new_i] * B[j, new_j]
            # Vectorized update:
            # marginal += np.outer(a[:, i], b[j, :]) + np.outer(a[i, :], b[:, j])
            # But we only care about unassigned. We can just update the whole matrix, it's O(n^2).
            marginal += np.outer(a[:, i], b[j, :]) + np.outer(a[i, :], b[:, j])
            
            # We don't strictly need to add A[i,i]B[j,j] to marginal, we add it to cost directly.
            
        return p

    # --- Main Logic ---
    
    # 1. Generate initial solution using Greedy Construction
    p_init = greedy_construction()
    current_cost = compute_cost(p_init)
    best_p = p_init.copy()
    best_cost = current_cost
    
    # 2. Local Search
    time_left_for_ls = deadline - 2.0
    p_ls, cost_ls = local_search_2opt(p_init.copy(), current_cost, time_left_for_ls)
    
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        
    # 3. Random Restarts with Perturbation
    # If time remains, do restarts.
    while time.time() < deadline - 2.0:
        # Perturb the best solution
        p_restart = best_p.copy()
        # Perform 5-10 random swaps
        num_swaps = 5
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
        
        cost_restart = compute_cost(p_restart)
        
        # Run local search on the perturbed solution
        # Use a shorter time limit for each restart
        time_left = deadline - 2.0 - time.time()
        if time_left < 0.1:
            break
            
        # Limit local search time per restart to avoid spending too much on one
        ls_time_limit = time.time() + min(10.0, time_left * 0.8)
        
        p_r, cost_r = local_search_2opt(p_restart, cost_restart, ls_time_limit)
        
        if cost_r < best_cost:
            best_cost = cost_r
            best_p = p_r.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
