"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Hungarian-based rank structure
    # The directive suggests using Hungarian algorithm on a proxy to respect global rank structure.
    # We construct a proxy cost matrix C where C[i, j] represents the "penalty" of assigning facility i to location j.
    # A common heuristic is to use the product of row sums or similar rank-preserving metrics.
    # Let's use the outer product of row sums, normalized, as a proxy for interaction strength.
    # High flow facility i should go to low distance location j.
    # So we want to minimize sum a_row_sums[i] * b_row_sums[j] ? No, we want to match high A with low B.
    # So the cost to assign i to j should be high if a_row_sums[i] is high and b_row_sums[j] is high.
    # Thus, proxy_cost[i, j] = a_row_sums[i] * b_row_sums[j].
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # To avoid large numbers, we can normalize or just use the product.
    # Hungarian minimizes the sum. So minimizing sum a_rs[i] * b_rs[p[i]] 
    # effectively tries to match high a_rs with low b_rs.
    
    proxy_cost = np.outer(a_row_sums, b_row_sums)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Calculate initial cost
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Optimized Simulated Annealing
    start_time = time.time()
    time_limit = 118.0
    
    # Precompute necessary vectors for delta calculation
    # delta_cost for swapping i and j:
    # Let p be current permutation.
    # Let pi = p[i], pj = p[j].
    # Old terms involving i or j:
    # sum_k a[i,k] b[pi, p[k]] + sum_k a[k,i] b[p[k], pi]
    # sum_k a[j,k] b[pj, p[k]] + sum_k a[k,j] b[p[k], pj]
    # Note: terms (i,i), (j,j), (i,j), (j,i) are counted twice in the full sum?
    # The objective is sum_{i,j} a[i,j] b[p[i], p[j]].
    # When we swap i and j, only terms where at least one index is i or j change.
    
    # Let's derive the delta precisely.
    # Cost = sum_{u,v} A[u,v] B[p[u], p[v]]
    # Let S be the set of pairs (u,v) such that u==i or v==i or u==j or v==j.
    # All other terms are unchanged.
    # Old_Cost_S = sum_{u,v in S} A[u,v] B[p[u], p[v]]
    # New_Cost_S = sum_{u,v in S} A[u,v] B[p_new[u], p_new[v]]
    # Delta = New_Cost_S - Old_Cost_S
    
    # p_new[i] = p[j] = pj
    # p_new[j] = p[i] = pi
    # p_new[u] = p[u] for u != i, j
    
    # We can compute this efficiently using vectorized operations.
    
    def delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        # We need to calculate the contribution of all pairs (u,v) where u or v is i or j.
        # This includes:
        # (i,i), (i,j), (j,i), (j,j)
        # (i,k) for k != i,j
        # (k,i) for k != i,j
        # (j,k) for k != i,j
        # (k,j) for k != i,j
        
        # Let's sum these up.
        # Instead of looping over k, we can use matrix rows/cols.
        
        # Term 1: u=i. v can be anything.
        # Old: sum_v A[i,v] B[pi, p[v]]
        # New: sum_v A[i,v] B[pj, p[v]]
        
        # Term 2: v=i. u can be anything.
        # Old: sum_u A[u,i] B[p[u], pi]
        # New: sum_u A[u,i] B[p[u], pj]
        
        # Term 3: u=j. v can be anything.
        # Old: sum_v A[j,v] B[pj, p[v]]
        # New: sum_v A[j,v] B[pi, p[v]]
        
        # Term 4: v=j. u can be anything.
        # Old: sum_u A[u,j] B[p[u], pj]
        # New: sum_u A[u,j] B[p[u], pi]
        
        # However, we must be careful not to double count (i,i), (i,j), (j,i), (j,j) if we just sum rows and cols?
        # The sum over u=i includes (i,i) and (i,j) and (i,k).
        # The sum over v=i includes (i,i) and (j,i) and (k,i).
        # So (i,i) is counted in both Term 1 and Term 2.
        # (i,j) is in Term 1 (u=i, v=j) and Term 4? No, Term 4 is v=j, so (i,j) is in Term 4 as well?
        # Let's check Term 4: sum_u A[u,j] B[p[u], pj]. For u=i, this is A[i,j] B[pi, pj].
        # Term 1: sum_v A[i,v] B[pi, p[v]]. For v=j, this is A[i,j] B[pi, pj].
        # So yes, (i,j) is counted in Term 1 (as part of row i) and Term 4 (as part of col j).
        # Wait, the objective is sum_{u,v}.
        # If we sum Row_i (all v) and Col_i (all u), we count (i,i) twice, (i,j) once in Row_i, (j,i) once in Col_i.
        # If we also sum Row_j and Col_j, we count (j,j) twice, (j,i) once in Row_j, (i,j) once in Col_j.
        
        # Let's just compute the change explicitly for the 4 "boundary" indices and the "bulk" terms.
        
        # Bulk terms: k not in {i, j}
        # Old Bulk:
        # sum_{k not i,j} (A[i,k] B[pi, p[k]] + A[k,i] B[p[k], pi] + A[j,k] B[pj, p[k]] + A[k,j] B[p[k], pj])
        # New Bulk:
        # sum_{k not i,j} (A[i,k] B[pj, p[k]] + A[k,i] B[p[k], pj] + A[j,k] B[pi, p[k]] + A[k,j] B[p[k], pi])
        
        # Delta Bulk = sum_{k not i,j} (
        #    A[i,k] (B[pj, p[k]] - B[pi, p[k]]) +
        #    A[k,i] (B[p[k], pj] - B[p[k], pi]) +
        #    A[j,k] (B[pi, p[k]] - B[pj, p[k]]) +
        #    A[k,j] (B[p[k], pi] - B[p[k], pj])
        # )
        # = sum_{k not i,j} (
        #    (A[i,k] - A[j,k]) (B[pj, p[k]] - B[pi, p[k]]) +
        #    (A[k,i] - A[k,j]) (B[p[k], pj] - B[p[k], pi])
        # )
        
        # This is still O(N). We can vectorize it.
        # Let diff_A_row = A[i, :] - A[j, :]
        # Let diff_A_col = A[:, i] - A[:, j]
        # Let diff_B_row = B[pj, :] - B[pi, :]
        # Let diff_B_col = B[:, pj] - B[:, pi]
        
        # We need sum over k not i,j of:
        # diff_A_row[k] * diff_B_row[k]  (Wait, B[pj, p[k]] - B[pi, p[k]] is NOT diff_B_row[k]. diff_B_row[k] is B[pj, k] - B[pi, k]. We need B[pj, p[k]].)
        
        # So we need:
        # 1. B[pj, p] - B[pi, p]  -> Vector of length n. Let's call it dB_row_perm.
        # 2. B[p, pj] - B[p, pi]  -> Vector of length n. Let's call it dB_col_perm.
        
        dB_row_perm = b[pj, p] - b[pi, p]
        dB_col_perm = b[p, pj] - b[p, pi]
        
        dA_row = a[i, :] - a[j, :]
        dA_col = a[:, i] - a[:, j]
        
        # Sum over all k:
        # sum_k dA_row[k] * dB_row_perm[k] + sum_k dA_col[k] * dB_col_perm[k]
        
        sum1 = np.dot(dA_row, dB_row_perm)
        sum2 = np.dot(dA_col, dB_col_perm)
        
        # This sum includes k=i and k=j. We need to subtract their contributions because the "Bulk" formula assumed k not in {i,j}.
        # But actually, the formula `Delta = New_Cost_S - Old_Cost_S` can be computed directly for the whole S set if we are careful.
        
        # Let's try a different, safer approach:
        # Calculate Old_Cost_S and New_Cost_S directly.
        # S is the union of row i, col i, row j, col j.
        # We can just sum the 4 vectors/matrices and adjust for overlaps.
        
        # Old_S = 
        #   sum_v A[i,v] B[pi, p[v]]  (Row i)
        # + sum_u A[u,i] B[p[u], pi]  (Col i)
        # + sum_v A[j,v] B[pj, p[v]]  (Row j)
        # + sum_u A[u,j] B[p[u], pj]  (Col j)
        # - A[i,i] B[pi, pi]          (Double counted in Row i and Col i)
        # - A[i,j] B[pi, pj]          (Counted in Row i (v=j) and Col j (u=i)? No. Row i has (i,j). Col j has (i,j) as (u=i, v=j). Yes, double counted.)
        # - A[j,i] B[pj, pi]          (Counted in Row j (v=i) and Col i (u=j). Yes, double counted.)
        # - A[j,j] B[pj, pj]          (Double counted in Row j and Col j)
        
        # Wait, is (i,j) in Col j? Col j is sum_u A[u,j] B[p[u], pj]. For u=i, term is A[i,j] B[pi, pj].
        # Is (i,j) in Row i? Row i is sum_v A[i,v] B[pi, p[v]]. For v=j, term is A[i,j] B[pi, pj].
        # Yes, it is counted twice.
        
        # So:
        old_row_i = np.dot(a[i, :], b[pi, p])
        old_col_i = np.dot(a[:, i], b[p, pi])
        old_row_j = np.dot(a[j, :], b[pj, p])
        old_col_j = np.dot(a[:, j], b[p, pj])
        
        old_ii = a[i, i] * b[pi, pi]
        old_ij = a[i, j] * b[pi, pj]
        old_ji = a[j, i] * b[pj, pi]
        old_jj = a[j, j] * b[pj, pj]
        
        old_cost_s = old_row_i + old_col_i + old_row_j + old_col_j - old_ii - old_ij - old_ji - old_jj
        
        # New costs
        # p_new[i] = pj, p_new[j] = pi
        
        new_row_i = np.dot(a[i, :], b[pj, p])  # Note: p is still old permutation in memory, but we use b[pj, p] which is correct for new row i?
        # Wait. In New state, row i is at location pj.
        # The term is sum_v A[i,v] B[pj, p_new[v]].
        # p_new[v] is p[v] for v!=i,j, and p_new[i]=pj, p_new[j]=pi.
        # So b[pj, p_new] is NOT just b[pj, p].
        # b[pj, p_new[v]] is:
        # if v==i: b[pj, pj]
        # if v==j: b[pj, pi]
        # else: b[pj, p[v]]
        
        # This is getting complicated to vectorize perfectly without temporary arrays.
        # Given N=150, O(N) is very fast. Even O(N^2) per step might be okay if we have fewer steps?
        # But O(N) is better.
        
        # Let's construct the vectors for the "changed" parts.
        
        # For New Row i:
        # Vector V_new_row_i of length n where V_new_row_i[v] = B[pj, p_new[v]]
        # V_new_row_i = b[pj, p].copy()
        # V_new_row_i[i] = b[pj, pj]
        # V_new_row_i[j] = b[pj, pi]
        
        # For New Col i:
        # Vector V_new_col_i of length n where V_new_col_i[u] = B[p_new[u], pj]
        # V_new_col_i = b[p, pj].copy()
        # V_new_col_i[i] = b[pj, pj]
        # V_new_col_i[j] = b[pi, pj]
        
        # For New Row j:
        # Vector V_new_row_j of length n where V_new_row_j[v] = B[pi, p_new[v]]
        # V_new_row_j = b[pi, p].copy()
        # V_new_row_j[i] = b[pi, pj]
        # V_new_row_j[j] = b[pi, pi]
        
        # For New Col j:
        # Vector V_new_col_j of length n where V_new_col_j[u] = B[p_new[u], pi]
        # V_new_col_j = b[p, pi].copy()
        # V_new_col_j[i] = b[pj, pi]
        # V_new_col_j[j] = b[pi, pi]
        
        # This involves copying vectors of size 150 and modifying 2 elements. This is very fast.
        
        v_nri = b[pj, p].copy()
        v_nri[i] = b[pj, pj]
        v_nri[j] = b[pj, pi]
        
        v_nci = b[p, pj].copy()
        v_nci[i] = b[pj, pj]
        v_nci[j] = b[pi, pj]
        
        v_nrj = b[pi, p].copy()
        v_nrj[i] = b[pi, pj]
        v_nrj[j] = b[pi, pi]
        
        v_ncj = b[p, pi].copy()
        v_ncj[i] = b[pj, pi]
        v_ncj[j] = b[pi, pi]
        
        new_row_i = np.dot(a[i, :], v_nri)
        new_col_i = np.dot(a[:, i], v_nci)
        new_row_j = np.dot(a[j, :], v_nrj)
        new_col_j = np.dot(a[:, j], v_ncj)
        
        new_ii = a[i, i] * b[pj, pj]
        new_ij = a[i, j] * b[pj, pi]
        new_ji = a[j, i] * b[pi, pj]
        new_jj = a[j, j] * b[pi, pi]
        
        new_cost_s = new_row_i + new_col_i + new_row_j + new_col_j - new_ii - new_ij - new_ji - new_jj
        
        return new_cost_s - old_cost_s

    # SA Loop
    T = 1000.0
    cooling_rate = 0.9999
    no_improve_counter = 0
    max_no_improve = 20000
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        delta = delta_swap(p, i, j)
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            
            # Adaptive cooling: if stuck, cool faster? Or restart?
            # Let's just cool.
        else:
            # Revert (do nothing to p)
            pass
        
        T *= cooling_rate
        
        # If T is too low, we might be stuck in a local minimum.
        # If no improvement for a long time, we can do a random perturbation (restart from best or random).
        # For now, simple cooling.
        
    # 3. Local Search Polish
    # Try all pairs for improvement? O(N^2) checks.
    # N=150, N^2 = 22500 checks. Each check is O(N) = 150 ops. Total ~3.3M ops. Very fast.
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
