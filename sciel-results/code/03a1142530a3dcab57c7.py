import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # --- Phase 1: Constructive Heuristic (Karmarkar-Karp style) ---
    # M[i, j] is the marginal cost of assigning facility i to location j,
    # given the current partial assignment.
    
    # Initialize M with the self-term: A[i,i] * B[j,j]
    M = np.outer(np.diag(a), np.diag(b))
    
    F = set(range(n))
    L = set(range(n))
    p = np.full(n, -1, dtype=np.int64)
    
    for step in range(n):
        if time.time() > deadline - 30.0:  # Leave 30s for local search
            break
            
        # Find min marginal cost among unassigned
        min_val = float('inf')
        best_i = -1
        best_j = -1
        
        # Optimization: Use numpy to find min in the submatrix of unassigned
        # But sets are unordered. For n=150, a simple loop is fast enough.
        # We can speed this up by maintaining a list of unassigned and using numpy indexing.
        # However, removal from list is O(n). Set removal is O(1).
        # Let's stick to the loop but optimize the inner access.
        
        # Actually, let's just iterate. 150*150 = 22500 checks. 150 steps = 3.3M checks.
        # This is very fast in Python if we don't do heavy lifting.
        
        for i in F:
            # We can vectorize the j loop?
            # vals = M[i, list(L)] -> creates a list, slow.
            # Let's just loop.
            for j in L:
                val = M[i, j]
                if val < min_val:
                    min_val = val
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            # Fallback: assign remaining randomly
            break
            
        p[best_i] = best_j
        F.remove(best_i)
        L.remove(best_j)
        
        # Update M for remaining unassigned
        # M[i', j'] += A[i', best_i] * B[j', best_j] + A[best_i, i'] * B[best_j, j']
        rem_i = list(F)
        rem_j = list(L)
        
        if rem_i and rem_j:
            # Vectorized update
            # v1 = A[rem_i, best_i] (shape len(rem_i))
            # v2 = B[best_j, rem_j] (shape len(rem_j))
            # Term 1: outer(v1, v2)
            # w1 = A[best_i, rem_i]
            # w2 = B[rem_j, best_j]
            # Term 2: outer(w1, w2)
            
            v1 = a[rem_i, best_i]
            v2 = b[best_j, rem_j]
            w1 = a[best_i, rem_i]
            w2 = b[rem_j, best_j]
            
            M[np.ix_(rem_i, rem_j)] += np.outer(v1, v2) + np.outer(w1, w2)
            
    # Complete permutation if KK timed out or failed
    if -1 in p:
        assigned_locs = set(p[p != -1])
        unassigned_locs = [j for j in range(n) if j not in assigned_locs]
        unassigned_fac = [i for i in range(n) if p[i] == -1]
        for i, j in zip(unassigned_fac, unassigned_locs):
            p[i] = j

    # --- Phase 2: Local Search with O(1) Delta Calculation ---
    
    # Precompute matrices for fast delta evaluation
    # R[i, l] = sum_k A[i, k] * B[l, p[k]]
    # C[i, l] = sum_k A[k, i] * B[p[k], l]
    
    def compute_precomp(p):
        # R = A @ B[:, p].T  -> R[i, l] = sum_k A[i,k] B[l, p[k]]
        # B_col_p = B[:, p]  # (n, n), B_col_p[l, k] = B[l, p[k]]
        # R = A @ B_col_p.T  # (n, n) @ (n, n) -> R[i, l]
        
        # C = A.T @ B[p, :]  -> C[i, l] = sum_k A[k, i] B[p[k], l]
        # B_row_p = B[p, :]  # (n, n), B_row_p[k, l] = B[p[k], l]
        # C = A.T @ B_row_p  # (n, n) @ (n, n) -> C[i, l]
        
        B_col_p = b[:, p]
        B_row_p = b[p, :]
        
        R = a @ B_col_p.T
        C = a.T @ B_row_p
        
        return R, C

    R, C = compute_precomp(p)
    current_cost = np.sum(a * b[np.ix_(p, p)])
    
    b_diag = np.diag(b)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # First pass: Deterministic 2-opt (try all pairs)
        # This is O(n^3) if we recompute, but with O(1) delta it's O(n^2) per pass.
        # However, we must update R and C after each swap.
        # Updating R and C is O(n^2).
        # So one pass is O(n^4) worst case? No.
        # If we do a full pass and only update R/C when a swap is accepted,
        # and we break early or do a single pass:
        # Worst case: 1 swap per pass? No, we can do multiple.
        # But updating R/C is expensive.
        
        # Let's try a randomized local search instead, which allows us to
        # sample pairs and only update R/C when a swap is accepted.
        # This is more robust for time limits.
        
        # Actually, the deterministic 2-opt with incremental update is:
        # For each pair (i, k):
        #   Calculate delta (O(1))
        #   If delta < 0:
        #     Swap p[i], p[k]
        #     Update R and C (O(n^2))
        #     improved = True
        #     # Break? Or continue?
        #     # If we continue, R and C are updated, so subsequent deltas are correct.
        #     # But O(n^2) update per swap makes a full pass O(n^4) if many swaps.
        #     # For n=150, n^4 = 5e8. Too slow for a full pass.
        
        # So, let's use a limited number of swaps per pass or randomized search.
        
        # Strategy: Randomized Local Search
        # Try random pairs. If improvement, accept and update R, C.
        # Keep going until time limit or no improvement for a while.
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            i = np.random.randint(0, n)
            k = np.random.randint(0, n)
            
            if i == k:
                continue
            
            j = p[i]
            l = p[k]
            
            # Calculate Delta
            # Terms involving i, k directly:
            # Old: A[i,k]B[j,l] + A[k,i]B[l,j] + A[i,i]B[j,j] + A[k,k]B[l,l]
            # New: A[i,k]B[l,j] + A[k,i]B[j,l] + A[i,i]B[l,l] + A[k,k]B[j,j]
            d_self = (a[i, k] * b[l, j] + a[k, i] * b[j, l] + a[i, i] * b[l, l] + a[k, k] * b[j, j]) - \
                     (a[i, k] * b[j, l] + a[k, i] * b[l, j] + a[i, i] * b[j, j] + a[k, k] * b[l, l])
            
            # Terms involving other v:
            # Delta_v = (R[i, l] - R[i, j] - a[i,i]*(b[l,l]-b[j,j]) - a[i,k]*(b[l,j]-b[j,l])) +
            #           (R[k, j] - R[k, l] - a[k,k]*(b[j,j]-b[l,l]) - a[k,i]*(b[j,l]-b[l,j])) +
            #           (C[i, l] - C[i, j] - a[i,i]*(b[l,l]-b[j,j]) - a[k,i]*(b[l,j]-b[j,l])) +
            #           (C[k, j] - C[k, l] - a[k,k]*(b[j,j]-b[l,l]) - a[i,k]*(b[j,l]-b[l,j]))
            
            # Let's verify the R terms.
            # R[i, l] = sum_v A[i,v] B[l, p[v]]
            # We want sum_{v != i, k} A[i,v] (B[l, p[v]] - B[j, p[v]])
            # = (R[i, l] - R[i, j]) - A[i,i](B[l, p[i]] - B[j, p[i]]) - A[i,k](B[l, p[k]] - B[j, p[k]])
            # p[i] = j, p[k] = l
            # = (R[i, l] - R[i, j]) - A[i,i](B[l, j] - B[j, j]) - A[i,k](B[l, l] - B[j, l])
            
            # Similarly for C.
            # C[i, l] = sum_v A[v,i] B[p[v], l]
            # We want sum_{v != i, k} A[v,i] (B[p[v], l] - B[p[v], j])
            # = (C[i, l] - C[i, j]) - A[i,i](B[p[i], l] - B[p[i], j]) - A[k,i](B[p[k], l] - B[p[k], j])
            # = (C[i, l] - C[i, j]) - A[i,i](B[j, l] - B[j, j]) - A[k,i](B[l, l] - B[l, j])
            
            term_Ri = (R[i, l] - R[i, j]) - a[i,i]*(b[l, j] - b[j, j]) - a[i,k]*(b[l, l] - b[j, l])
            term_Rk = (R[k, j] - R[k, l]) - a[k,k]*(b[j, j] - b[l, l]) - a[k,i]*(b[j, l] - b[l, j])
            
            term_Ci = (C[i, l] - C[i, j]) - a[i,i]*(b[j, l] - b[j, j]) - a[k,i]*(b[l, l] - b[l, j])
            term_Ck = (C[k, j] - C[k, l]) - a[k,k]*(b[j, j] - b[l, l]) - a[i,k]*(b[j, l] - b[l, j])
            
            d_v = term_Ri + term_Rk + term_Ci + term_Ck
            
            delta = d_self + d_v
            
            if delta < 0:
                # Accept swap
                p[i], p[k] = p[k], p[i]
                current_cost += delta
                improved = True
                no_improve_count = 0
                
                # Update R and C
                # R[i, :] changes. R[:, i] changes? No, R is A @ B[:, p].T
                # R[k, l] = sum_v A[k,v] B[l, p[v]]
                # When p[i] and p[k] swap:
                # R[x, y] = sum_v A[x,v] B[y, p[v]]
                # Change in R[x, y]:
                # A[x,i] (B[y, l] - B[y, j]) + A[x,k] (B[y, j] - B[y, l])
                # = (A[x,i] - A[x,k]) * (B[y, l] - B[y, j])
                
                # This is an outer product update!
                # Delta_R = outer(A[:, i] - A[:, k], B[:, l] - B[:, j])
                # R += Delta_R
                
                # Similarly for C:
                # C[x, y] = sum_v A[v, x] B[p[v], y]
                # Change:
                # A[i, x] (B[l, y] - B[j, y]) + A[k, x] (B[j, y] - B[l, y])
                # = (A[i, x] - A[k, x]) * (B[l, y] - B[j, y])
                # = (A[i, :] - A[k, :])_x * (B[l, :] - B[j, :])_y
                # Delta_C = outer(A[i, :] - A[k, :], B[l, :] - B[j, :])
                # C += Delta_C
                
                # Wait, let's double check C.
                # C[x, y] = sum_v A[v, x] B[p[v], y]
                # v=i: A[i, x] B[p[i], y] -> A[i, x] B[l, y] (was B[j, y])
                # v=k: A[k, x] B[p[k], y] -> A[k, x] B[j, y] (was B[l, y])
                # Change = A[i, x] (B[l, y] - B[j, y]) + A[k, x] (B[j, y] - B[l, y])
                # = (A[i, x] - A[k, x]) * (B[l, y] - B[j, y])
                # Yes.
                
                R += np.outer(a[:, i] - a[:, k], b[:, l] - b[:, j])
                C += np.outer(a[i, :] - a[k, :], b[l, :] - b[j, :])
                
            else:
                no_improve_count += 1

    return p.tolist()
