"""QAP solver: Hungarian initialization + fast O(1) delta local search with incremental precomputation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Initialize via Hungarian on C = a @ b (linear relaxation).
    2. Local search using precomputed dot_row/dot_col matrices for O(1) delta lookups.
    3. Incremental O(n) updates to precomputed matrices after each swap.
    4. Iterate: Local Search -> Hungarian Refinement -> Local Search.
    """
    n = len(a)
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    np.random.seed(42)

    def compute_cost(p):
        """Compute exact QAP cost."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def init_precomp(p):
        """Initialize dot_row and dot_col matrices.
        dot_row[i, l] = sum_k a[i, k] * b[l, p[k]]
        dot_col[i, l] = sum_k a[k, i] * b[p[k], l]
        """
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        
        dot_row = a.astype(np.float64) @ b_col_p.astype(np.float64).T
        dot_col = a.T.astype(np.float64) @ b_row_p.astype(np.float64)
        
        return dot_row, dot_col

    def hungarian_init():
        """Initialize via Hungarian on C = a @ b."""
        C = a.astype(np.float64) @ b.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_refine(p, dot_row, dot_col):
        """Refine using Hungarian on linearized cost matrix based on current p."""
        # The linearized cost for assigning facility i to location j given current assignment p is:
        # Cost_ij = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        #         = dot_row[i, j] + dot_col[i, j]
        C = dot_row + dot_col
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    def local_search_fast(p, cost, dot_row, dot_col, max_time, num_trials=50000):
        """Fast local search with O(1) delta lookups and O(n) precomp updates."""
        improved = True
        trial_count = 0
        while improved and time.time() < max_time:
            improved = False
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                trial_count += 1
                
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                
                # Calculate delta using precomputed matrices
                buu = b[u, u]
                bvv = b[v, v]
                buv = b[u, v]
                bvu = b[v, u]
                
                # Old Sum
                old_ii = a[i, i] * buu
                old_ij = a[i, j] * buv
                old_ji = a[j, i] * bvu
                old_jj = a[j, j] * bvv
                old_5 = dot_row[i, u] - a[i, i]*buu - a[i, j]*buv
                old_6 = dot_col[i, u] - a[i, i]*buu - a[j, i]*bvu
                old_7 = dot_row[j, v] - a[j, i]*bvu - a[j, j]*bvv
                old_8 = dot_col[j, v] - a[i, j]*buv - a[j, j]*bvv
                old_total = old_ii + old_ij + old_ji + old_jj + old_5 + old_6 + old_7 + old_8
                
                # New Sum
                new_ii = a[i, i] * bvv
                new_ij = a[i, j] * bvu
                new_ji = a[j, i] * buv
                new_jj = a[j, j] * buu
                new_5 = dot_row[i, v] - a[i, i]*bvu - a[i, j]*bvv
                new_6 = dot_col[i, v] - a[i, i]*buv - a[j, i]*bvv
                new_7 = dot_row[j, u] - a[j, i]*buv - a[j, j]*buu
                new_8 = dot_col[j, u] - a[i, j]*buu - a[j, j]*bvu
                new_total = new_ii + new_ij + new_ji + new_jj + new_5 + new_6 + new_7 + new_8
                
                delta = new_total - old_total
                
                if delta < -1e-9:
                    # Accept swap
                    p[i], p[j] = v, u
                    cost += int(delta)
                    improved = True
                    
                    # Update dot_row and dot_col in O(n)
                    b_diff = b[:, v] - b[:, u]
                    diff_a_col = a[:, i] - a[:, j]
                    dot_row += np.outer(diff_a_col, b_diff)
                    
                    diff_a_row = a[i, :] - a[j, :]
                    b_diff_T = b[v, :] - b[u, :]
                    dot_col += np.outer(diff_a_row, b_diff_T)
                    
                    # Periodic re-normalization to prevent drift
                    if trial_count % 5000 == 0:
                        dot_row, dot_col = init_precomp(p)

        return p, cost, dot_row, dot_col

    # --- Phase 1: Hungarian-based initialization ---
    best_p = hungarian_init()
    best_cost = compute_cost(best_p)
    dot_row, dot_col = init_precomp(best_p)

    # --- Phase 2: Initial local search ---
    best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 50.0)

    # --- Phase 3: Hungarian refinement + local search ---
    if time.time() < deadline - 35.0:
        p_hung = hungarian_refine(best_p, dot_row, dot_col)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            dot_row, dot_col = init_precomp(best_p)
            best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 25.0)

    # --- Phase 4: Second Hungarian refinement ---
    if time.time() < deadline - 20.0:
        p_hung = hungarian_refine(best_p, dot_row, dot_col)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            dot_row, dot_col = init_precomp(best_p)
            best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 10.0)

    # --- Phase 5: Final local search ---
    if time.time() < deadline - 2.0:
        best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 1.0)

    # --- Phase 6: High-precision polish pass ---
    if time.time() < deadline - 0.5:
        # Re-compute exact cost to verify
        exact_cost = compute_cost(best_p)
        if abs(exact_cost - best_cost) > 1:
            best_cost = exact_cost
            dot_row, dot_col = init_precomp(best_p)
        
        # Run a focused polish with tighter tolerance
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            p_polish, cost_polish, _, _ = local_search_fast(
                best_p.copy(), best_cost, dot_row.copy(), dot_col.copy(), 
                deadline - 0.5, num_trials=10000
            )
            if cost_polish < best_cost:
                best_p = p_polish
                best_cost = cost_polish
                dot_row, dot_col = init_precomp(best_p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
