import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 145.0
    
    # --- Phase 1: Karmarkar-Karp Constructive Heuristic ---
    F = list(range(n))
    L = list(range(n))
    p = np.full(n, -1, dtype=np.int64)
    
    # Initialize M
    M = np.outer(np.diag(a), np.diag(b))
    
    # Use indices for F and L to allow O(1) removal (swap with last)
    F_set = set(F)
    L_set = set(L)
    F_idx = {i: i for i in range(n)}
    L_idx = {j: j for j in range(n)}
    
    for step in range(n):
        if time.time() > deadline - 20:
            break
            
        min_val = float('inf')
        best_i = -1
        best_j = -1
        
        # Find min in M among unassigned
        # Optimization: iterate over current F and L lists
        for i in F:
            if i not in F_set: continue
            for j in L:
                if j not in L_set: continue
                val = M[i, j]
                if val < min_val:
                    min_val = val
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            # Fallback: assign remaining arbitrarily
            break
            
        p[best_i] = best_j
        
        # Remove best_i from F
        pos_i = F_idx[best_i]
        last_i = F[-1]
        F[pos_i] = last_i
        F_idx[last_i] = pos_i
        F.pop()
        del F_idx[best_i]
        F_set.remove(best_i)
        
        # Remove best_j from L
        pos_j = L_idx[best_j]
        last_j = L[-1]
        L[pos_j] = last_j
        L_idx[last_j] = pos_j
        L.pop()
        del L_idx[best_j]
        L_set.remove(best_j)
        
        # Update M for remaining
        if F and L:
            # We need to update M[i, j] for i in F, j in L
            # M[i, j] += A[i, best_i] * B[j, best_j] + A[best_i, i] * B[best_j, j]
            
            # Vectorized update
            # v1 = A[F, best_i] (len F)
            # v2 = B[best_j, L] (len L)
            # w1 = A[best_i, F] (len F)
            # w2 = B[L, best_j] (len L)
            
            v1 = a[F, best_i]
            v2 = b[best_j, L]
            w1 = a[best_i, F]
            w2 = b[L, best_j]
            
            # Update M[F, L]
            # M[F, L] += outer(v1, v2) + outer(w1, w2)
            # Using advanced indexing
            M[np.ix_(F, L)] += np.outer(v1, v2) + np.outer(w1, w2)
            
    # Complete if necessary
    if -1 in p:
        assigned_locs = set(p[p != -1])
        unassigned_locs = [j for j in range(n) if j not in assigned_locs]
        unassigned_fac = [i for i in range(n) if p[i] == -1]
        for i, j in zip(unassigned_fac, unassigned_locs):
            p[i] = j

    # --- Phase 2: Efficient 2-opt Local Search ---
    
    # Precompute for faster delta calculation
    # We maintain:
    # R[i, loc] = sum_v A[i, v] B[loc, p[v]]
    # C[loc, i] = sum_v B[loc, p[v]] A[v, i]
    
    # Initial computation
    # R[i, l] = sum_v A[i, v] B[l, p[v]]
    # Let P be the permutation matrix? No.
    # B_p = B[np.ix_(p, p)]
    # R = A @ B_p ?
    # (A @ B_p)[i, l] = sum_v A[i, v] B_p[v, l] = sum_v A[i, v] B[p[v], l]
    # Wait, my definition was B[l, p[v]].
    # Let's stick to:
    # S1[i, l] = sum_v A[i, v] B[l, p[v]]
    # This is (A @ B.T)[i, l] ?
    # (A @ B.T)[i, l] = sum_v A[i, v] B.T[v, l] = sum_v A[i, v] B[l, v].
    # If we replace v with p[v], it's not a simple matrix product unless we permute.
    # S1 = A @ B[np.ix_(p, range(n))]? No.
    # S1[i, l] = sum_v A[i, v] B[l, p[v]]
    # Let's compute S1 = A @ B_p.T ?
    # B_p = B[np.ix_(p, p)]
    # B_p[v, l] = B[p[v], p[l]]. Not quite.
    
    # Let's just compute S1 and S2 directly using einsum or loops if necessary,
    # or matrix multiplication with permuted matrices.
    
    # Let B_perm = B[np.ix_(p, p)]
    # Then B_perm[v, u] = B[p[v], p[u]].
    # We want sum_v A[i, v] B[l, p[v]].
    # Note B[l, p[v]] is element (l, p[v]) of B.
    # If we define B_row_l = B[l, :], then we need dot(A[i, :], B_row_l[p, :])? No.
    # sum_v A[i, v] B[l, p[v]] = dot(A[i, :], B[l, p]).
    # So S1[i, l] = dot(A[i, :], B[l, p]).
    # This is A @ B[:, p].T ?
    # Let B_cp = B[:, p] (n x n).
    # (A @ B_cp.T)[i, l] = sum_v A[i, v] B_cp.T[v, l] = sum_v A[i, v] B_cp[l, v] = sum_v A[i, v] B[l, p[v]].
    # Yes! S1 = A @ B[:, p].T
    
    # Similarly for S2[loc, i] = sum_v B[loc, p[v]] A[v, i]
    # S2 = B[:, p] @ A.T ?
    # (B[:, p] @ A.T)[loc, i] = sum_v B[:, p][loc, v] A.T[v, i] = sum_v B[loc, p[v]] A[i, v].
    # Wait, I need A[v, i]. A.T[v, i] is A[i, v]. So this gives sum_v B[loc, p[v]] A[i, v].
    # I need sum_v B[loc, p[v]] A[v, i].
    # Let's check indices.
    # S2[loc, i] = sum_v B[loc, p[v]] A[v, i].
    # Let M = B[:, p]. M[loc, v] = B[loc, p[v]].
    # We want sum_v M[loc, v] A[v, i].
    # This is (M @ A)[loc, i].
    # So S2 = B[:, p] @ A.
    
    # Let's re-verify the delta formula with these precomputations.
    # Delta for swapping i and k (locations j=p[i], l=p[k]):
    # D_v = sum_{v != i,k} ( A[i,v](B[l,p[v]] - B[j,p[v]]) + A[v,i](B[p[v],l] - B[p[v],j]) +
    #                        A[k,v](B[j,p[v]] - B[l,p[v]]) + A[v,k](B[p[v],j] - B[p[v],l]) )
    
    # Let's group terms by v.
    # Term for v:
    # T_v = A[i,v]B[l,p[v]] - A[i,v]B[j,p[v]] + A[v,i]B[p[v],l] - A[v,i]B[p[v],j] +
    #       A[k,v]B[j,p[v]] - A[k,v]B[l,p[v]] + A[v,k]B[p[v],j] - A[v,k]B[p[v],l]
    
    # Sum over all v:
    # Sum_all = sum_v (A[i,v]B[l,p[v]] - A[i,v]B[j,p[v]]) + sum_v (A[v,i]B[p[v],l] - A[v,i]B[p[v],j]) + ...
    
    # Using S1:
    # sum_v A[i,v] B[l,p[v]] = S1[i, l]
    # sum_v A[i,v] B[j,p[v]] = S1[i, j]
    # So first part: S1[i, l] - S1[i, j]
    
    # Using S2:
    # sum_v A[v,i] B[p[v],l].
    # Note B[p[v], l] is NOT B[l, p[v]]. B is not necessarily symmetric.
    # My S2 definition was S2[loc, i] = sum_v B[loc, p[v]] A[v, i].
    # So sum_v A[v,i] B[l, p[v]] = S2[l, i].
    # But I need sum_v A[v,i] B[p[v], l].
    # Let's define S3[loc, i] = sum_v A[v, i] B[p[v], loc].
    # S3[loc, i] = sum_v B[p[v], loc] A[v, i].
    # Let M = B[p, :] (n x n). M[v, loc] = B[p[v], loc].
    # (M.T @ A)[loc, i] = sum_v M.T[loc, v] A[v, i] = sum_v M[v, loc] A[v, i] = sum_v B[p[v], loc] A[v, i].
    # So S3 = B[p, :].T @ A.
    
    # Let's re-evaluate the terms:
    # 1. sum_v A[i,v] B[l,p[v]] - A[i,v]B[j,p[v]] = S1[i, l] - S1[i, j]
    # 2. sum_v A[v,i] B[p[v],l] - A[v,i]B[p[v],j] = S3[l, i] - S3[j, i]
    # 3. sum_v A[k,v] B[j,p[v]] - A[k,v]B[l,p[v]] = S1[k, j] - S1[k, l]
    # 4. sum_v A[v,k] B[p[v],j] - A[v,k]B[p[v],l] = S3[j, k] - S3[l, k]
    
    # Sum_all = (S1[i, l] - S1[i, j]) + (S3[l, i] - S3[j, i]) + (S1[k, j] - S1[k, l]) + (S3[j, k] - S3[l, k])
    
    # Now subtract terms for v=i and v=k.
    # For v=i:
    # T_i = A[i,i]B[l,j] - A[i,i]B[j,j] + A[i,i]B[j,l] - A[i,i]B[j,j] +
    #       A[k,i]B[j,j] - A[k,i]B[l,j] + A[i,k]B[j,j] - A[i,k]B[j,l]
    # (Using p[i]=j, p[k]=l)
    # Note: B[p[i], l] = B[j, l]. B[l, p[i]] = B[l, j].
    
    # For v=k:
    # T_k = A[i,k]B[l,l] - A[i,k]B[j,l] + A[k,i]B[l,l] - A[k,i]B[l,j] +
    #       A[k,k]B[j,l] - A[k,k]B[l,l] + A[k,k]B[l,j] - A[k,k]B[l,l]
    
    # D_v = Sum_all - T_i - T_k
    
    # D_self (terms involving i,k in A and B directly, not through sum_v)
    # Wait, the sum_v included v=i and v=k.
    # The "D_self" I calculated earlier was the change in the "diagonal" or "pair" terms that are explicitly handled?
    # Actually, the formula sum_{i,j} A[i,j] B[p[i], p[j]] includes all pairs.
    # When we swap i and k, the positions of i and k change.
    # The change in cost is exactly the change in the sum.
    # The sum over all v (including i and k) accounts for the interactions of i and k with ALL other facilities, including each other and themselves.
    # So D_v (corrected) IS the total delta?
    # Let's check.
    # Cost = sum_{u,v} A[u,v] B[p[u], p[v]].
    # Delta = Cost_new - Cost_old.
    # The only terms that change are those where u=i, u=k, v=i, or v=k.
    # My Sum_all - T_i - T_k approach:
    # Sum_all calculates the contribution of the new positions to the "row/col" sums for all v.
    # By subtracting T_i and T_k, we remove the contributions where the *other* facility was i or k.
    # But we still need to account for the terms where the *first* index is i or k.
    # Wait, S1[i, l] is sum_v A[i, v] B[l, p[v]]. This is the contribution of facility i to the cost if it were at l, interacting with all v.
    # So S1[i, l] - S1[i, j] is the change in contribution of facility i (as the first index) when moving from j to l.
    # S3[l, i] - S3[j, i] is the change in contribution of facility i (as the second index) when moving from j to l.
    # So (S1[i, l] - S1[i, j]) + (S3[l, i] - S3[j, i]) is the total change in cost involving facility i.
    # Similarly for k.
    # So Total Delta = Delta_i + Delta_k.
    # Delta_i = (S1[i, l] - S1[i, j]) + (S3[l, i] - S3[j, i]).
    # Delta_k = (S1[k, j] - S1[k, l]) + (S3[j, k] - S3[l, k]).
    # Sum_all = Delta_i + Delta_k.
    # BUT, this double counts the interaction between i and k?
    # In Delta_i, when v=k, we have A[i, k] B[l, p[k]] - A[i, k] B[j, p[k]] = A[i, k] B[l, l] - A[i, k] B[j, l].
    # In Delta_k, when v=i, we have A[k, i] B[j, p[i]] - A[k, i] B[l, p[i]] = A[k, i] B[j, j] - A[k, i] B[l, j].
    # Also in Delta_i, when v=i, we have A[i, i] B[l, j] - A[i, i] B[j, j] (from S1) and A[i, i] B[j, l] - A[i, i] B[j, j] (from S3).
    # This seems to handle all terms correctly without double counting, because the terms are distinct (A[i,k] vs A[k,i]).
    # So Delta = Sum_all.
    # Do we need to subtract T_i and T_k?
    # T_i and T_k were derived from the explicit expansion of the terms where v=i and v=k.
    # But S1 and S3 already include v=i and v=k in the sum.
    # So if we use S1 and S3, we DON'T need to subtract T_i and T_k. The sums are correct as they are.
    # Let's verify.
    # S1[i, l] = sum_v A[i, v] B[l, p[v]]. Includes v=i and v=k.
    # The change in cost for facility i is indeed (S1[i, l] - S1[i, j]) + (S3[l, i] - S3[j, i]).
    # This accounts for all interactions of i with all v (including i and k).
    # Similarly for k.
    # So Delta = (S1[i, l] - S1[i, j] + S3[l, i] - S3[j, i]) + (S1[k, j] - S1[k, l] + S3[j, k] - S3[l, k]).
    
    # This is O(1) per pair (i, k)!
    
    # Let's implement this.
    
    def update_precomp(p):
        B_cp = b[:, p]
        B_p_row = b[p, :]
        S1 = a @ B_cp.T
        S3 = B_p_row.T @ a
        return S1, S3

    S1, S3 = update_precomp(p)
    
    def calc_cost(p):
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    current_cost = calc_cost(p)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p.tolist()
            j = p[i]
            for k in range(i + 1, n):
                l = p[k]
                
                delta = (S1[i, l] - S1[i, j] + S3[l, i] - S3[j, i]) + \
                        (S1[k, j] - S1[k, l] + S3[j, k] - S3[l, k])
                
                if delta < 0:
                    # Swap
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    
                    # Update S1 and S3
                    # S1[i, :] changes, S1[k, :] changes
                    # S3[:, i] changes, S3[:, k] changes
                    # Actually, S1 and S3 depend on p.
                    # S1 = a @ b[:, p].T
                    # S3 = b[p, :].T @ a
                    # When we swap p[i] and p[k], the columns of b[:, p] at positions i and k swap?
                    # No, p is the assignment. b[:, p] is the matrix where column v is b[:, p[v]].
                    # So column i of b[:, p] is b[:, p[i]] = b[:, j].
                    # Column k of b[:, p] is b[:, p[k]] = b[:, l].
                    # After swap, p[i]=l, p[k]=j.
                    # New column i is b[:, l]. New column k is b[:, j].
                    # So columns i and k of b[:, p] are swapped.
                    # Thus, rows i and k of (b[:, p].T) are swapped.
                    # S1 = a @ (b[:, p].T).
                    # If we swap rows i and k of the right matrix, we swap rows i and k of the result?
                    # No. (A @ B)[i, l] = sum_v A[i, v] B[v, l].
                    # If we swap rows i and k of B, then (A @ B_new)[i, l] = sum_v A[i, v] B_new[v, l].
                    # This is not a simple row swap of S1.
                    # We need to recalculate S1 and S3.
                    # O(n^2) update.
                    # 150^2 = 22500 ops.
                    # 11k pairs * 22500 = 250M ops per pass.
                    # This might be too slow if many swaps occur.
                    # But if few swaps occur, it's fine.
                    
                    S1, S3 = update_precomp(p)
                    
                    # Break inner loop to restart i loop?
                    # Since S1, S3 changed, the delta for other k is invalid.
                    break
            
    return p.tolist()

# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 145.0
    
    # --- Phase 1: Karmarkar-Karp Constructive Heuristic ---
    F = list(range(n))
    L = list(range(n))
    p = np.full(n, -1, dtype=np.int64)
    
    # Initialize M
    M = np.outer(np.diag(a), np.diag(b))
    
    # Use indices for F and L to allow O(1) removal (swap with last)
    F_set = set(F)
    L_set = set(L)
    F_idx = {i: i for i in range(n)}
    L_idx = {j: j for j in range(n)}
    
    for step in range(n):
        if time.time() > deadline - 20:
            break
            
        min_val = float('inf')
        best_i = -1
        best_j = -1
        
        # Find min in M among unassigned
        for i in F:
            if i not in F_set: continue
            for j in L:
                if j not in L_set: continue
                val = M[i, j]
                if val < min_val:
                    min_val = val
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break
            
        p[best_i] = best_j
        
        # Remove best_i from F
        pos_i = F_idx[best_i]
        last_i = F[-1]
        F[pos_i] = last_i
        F_idx[last_i] = pos_i
        F.pop()
        del F_idx[best_i]
        F_set.remove(best_i)
        
        # Remove best_j from L
        pos_j = L_idx[best_j]
        last_j = L[-1]
        L[pos_j] = last_j
        L_idx[last_j] = pos_j
        L.pop()
        del L_idx[best_j]
        L_set.remove(best_j)
        
        # Update M for remaining
        if F and L:
            v1 = a[F, best_i]
            v2 = b[best_j, L]
            w1 = a[best_i, F]
            w2 = b[L, best_j]
            
            M[np.ix_(F, L)] += np.outer(v1, v2) + np.outer(w1, w2)
            
    # Complete if necessary
    if -1 in p:
        assigned_locs = set(p[p != -1])
        unassigned_locs = [j for j in range(n) if j not in assigned_locs]
        unassigned_fac = [i for i in range(n) if p[i] == -1]
        for i, j in zip(unassigned_fac, unassigned_locs):
            p[i] = j

    # --- Phase 2: Efficient 2-opt Local Search ---
    
    def update_precomp(p):
        B_cp = b[:, p]
        B_p_row = b[p, :]
        S1 = a @ B_cp.T
        S3 = B_p_row.T @ a
        return S1, S3

    S1, S3 = update_precomp(p)
    
    def calc_cost(p):
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    current_cost = calc_cost(p)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p.tolist()
            j = p[i]
            for k in range(i + 1, n):
                l = p[k]
                
                delta = (S1[i, l] - S1[i, j] + S3[l, i] - S3[j, i]) + \
                        (S1[k, j] - S1[k, l] + S3[j, k] - S3[l, k])
                
                if delta < 0:
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    
                    S1, S3 = update_precomp(p)
                    break
            
    return p.tolist()
# EVOLVE-BLOCK-END
