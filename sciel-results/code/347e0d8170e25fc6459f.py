"""QAP solver: Hungarian initialization + O(1) delta local search with incremental precomputation."""
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
    deadline = time.time() + 112.0  # Safety margin before 120s limit
    np.random.seed(42)

    def compute_cost(p):
        """Compute exact QAP cost."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def init_precomp(p):
        """Initialize dot_row and dot_col matrices.
        dot_row[i, l] = sum_k a[i, k] * b[l, p[k]]
        dot_col[i, l] = sum_k a[k, i] * b[p[k], l]
        """
        # b_col_p[j, k] = b[j, p[k]]
        b_col_p = b[:, p]
        # b_row_p[k, j] = b[p[k], j]
        b_row_p = b[p, :]
        
        # dot_row[i, l] = sum_k a[i, k] * b[l, p[k]] = (a @ b_col_p.T)[i, l]
        dot_row = a.astype(np.float64) @ b_col_p.astype(np.float64).T
        # dot_col[i, l] = sum_k a[k, i] * b[p[k], l] = (a.T @ b_row_p)[i, l]
        dot_col = a.T.astype(np.float64) @ b_row_p.astype(np.float64)
        
        return dot_row, dot_col

    def hungarian_init():
        """Initialize via Hungarian on C = a @ b."""
        C = a.astype(np.float64) @ b.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_refine(p):
        """Refine using Hungarian on linearized cost matrix based on current p."""
        b_col_p = b[:, p]   # shape (n,n): b_col_p[j,k] = b[j,p[k]]
        b_row_p = b[p, :]   # shape (n,n): b_row_p[k,j] = b[p[k],j]
        
        # Linearized cost for assigning facility i to location j:
        # Cost_ij = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        #         = dot_row[i, j] + dot_col[i, j] (if dot_row/col were defined with current p)
        # However, dot_row/col are maintained for the *current* p.
        # So we can just use the current dot_row and dot_col!
        # Wait, the definition of dot_row[i, l] = sum_k a[i, k] b[l, p[k]].
        # The term sum_k a[i,k] b[j, p[k]] is exactly dot_row[i, j].
        # The term sum_k a[k,i] b[p[k], j] is exactly dot_col[i, j].
        # So the linearized cost matrix is simply dot_row + dot_col.
        
        C = dot_row + dot_col
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    def local_search_fast(p, cost, dot_row, dot_col, max_time, num_trials=50000):
        """Fast local search with O(1) delta lookups and O(n) precomp updates."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                
                # Calculate delta using precomputed matrices
                # Delta = (new cost - old cost)
                # Terms involving i and j positions change.
                #
                # Let's derive the O(1) formula.
                # Old cost terms involving i or j:
                # 1. a[i,i] * b[u,u]
                # 2. a[i,j] * b[u,v]
                # 3. a[j,i] * b[v,u]
                # 4. a[j,j] * b[v,v]
                # 5. sum_{k!=i,j} a[i,k] * b[u, p[k]]
                # 6. sum_{k!=i,j} a[k,i] * b[p[k], u]
                # 7. sum_{k!=i,j} a[j,k] * b[v, p[k]]
                # 8. sum_{k!=i,j} a[k,j] * b[p[k], v]
                #
                # New cost terms (swap u,v):
                # 1'. a[i,i] * b[v,v]
                # 2'. a[i,j] * b[v,u]
                # 3'. a[j,i] * b[u,v]
                # 4'. a[j,j] * b[u,u]
                # 5'. sum_{k!=i,j} a[i,k] * b[v, p[k]]
                # 6'. sum_{k!=i,j} a[k,i] * b[p[k], v]
                # 7'. sum_{k!=i,j} a[j,k] * b[u, p[k]]
                # 8'. sum_{k!=i,j} a[k,j] * b[p[k], u]
                #
                # Using dot_row[i, l] = sum_k a[i,k] b[l, p[k]]:
                # sum_{k!=i,j} a[i,k] b[u, p[k]] = dot_row[i, u] - a[i,i]b[u,u] - a[i,j]b[u,v]
                # sum_{k!=i,j} a[i,k] b[v, p[k]] = dot_row[i, v] - a[i,i]b[v,u] - a[i,j]b[v,v]
                #
                # Using dot_col[i, l] = sum_k a[k,i] b[p[k], l]:
                # sum_{k!=i,j} a[k,i] b[p[k], u] = dot_col[i, u] - a[i,i]b[u,u] - a[j,i]b[v,u]
                # sum_{k!=i,j} a[k,i] b[p[k], v] = dot_col[i, v] - a[i,i]b[u,v] - a[j,i]b[v,v]
                #
                # Similarly for j:
                # sum_{k!=i,j} a[j,k] b[v, p[k]] = dot_row[j, v] - a[j,i]b[v,u] - a[j,j]b[v,v]
                # sum_{k!=i,j} a[j,k] b[u, p[k]] = dot_row[j, u] - a[j,i]b[u,v] - a[j,j]b[u,u]
                #
                # sum_{k!=i,j} a[k,j] b[p[k], v] = dot_col[j, v] - a[i,j]b[u,v] - a[j,j]b[v,v]
                # sum_{k!=i,j} a[k,j] b[p[k], u] = dot_col[j, u] - a[i,j]b[u,u] - a[j,j]b[v,u]
                
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
                    # dot_row[k, l] = sum_m a[k,m] b[l, p[m]]
                    # Change in p: p[i] u->v, p[j] v->u
                    # For k != i,j:
                    # dot_row[k, l] += a[k,i](b[l,v] - b[l,u]) + a[k,j](b[l,u] - b[l,v])
                    #                = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    # For k = i:
                    # dot_row[i, l] += a[i,i](b[l,v] - b[l,u]) + a[i,j](b[l,u] - b[l,v])
                    #                = (a[i,i] - a[i,j]) * (b[l,v] - b[l,u])
                    # For k = j:
                    # dot_row[j, l] += a[j,i](b[l,v] - b[l,u]) + a[j,j](b[l,u] - b[l,v])
                    #                = (a[j,i] - a[j,j]) * (b[l,v] - b[l,u])
                    
                    b_diff = b[:, v] - b[:, u] # vector of size n, b_diff[l] = b[l,v] - b[l,u]
                    
                    # Update all rows k
                    # dot_row[k, :] += (a[k,i] - a[k,j]) * b_diff
                    # We can do this vectorized:
                    # diff_a_col = a[:, i] - a[:, j]  (size n)
                    # dot_row += np.outer(diff_a_col, b_diff)
                    
                    diff_a_col = a[:, i] - a[:, j]
                    dot_row += np.outer(diff_a_col, b_diff)
                    
                    # Update dot_col
                    # dot_col[k, l] = sum_m a[m,k] b[p[m], l]
                    # Change in p: p[i] u->v, p[j] v->u
                    # For m != i,j:
                    # dot_col[k, l] += a[i,k](b[v,l] - b[u,l]) + a[j,k](b[u,l] - b[v,l])
                    #                = (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
                    # For m = i:
                    # dot_col[k, l] += a[i,k](b[v,l] - b[u,l]) + a[j,k](b[u,l] - b[v,l]) -- wait, a[i,k] is coefficient for p[i]
                    # Let's re-derive dot_col update.
                    # dot_col[k, l] = sum_m a[m,k] b[p[m], l]
                    # New p: p'[i]=v, p'[j]=u, p'[m]=p[m] for m!=i,j
                    # dot_col'[k, l] = a[i,k]b[v,l] + a[j,k]b[u,l] + sum_{m!=i,j} a[m,k] b[p[m],l]
                    # dot_col[k, l]  = a[i,k]b[u,l] + a[j,k]b[v,l] + sum_{m!=i,j} a[m,k] b[p[m],l]
                    # Delta = a[i,k](b[v,l]-b[u,l]) + a[j,k](b[u,l]-b[v,l])
                    #       = (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
                    # Note: This formula holds for ALL k, including k=i and k=j.
                    # Because the sum over m covers all m, and the change is only in terms m=i and m=j.
                    # The coefficients a[i,k] and a[j,k] are fixed.
                    
                    # So: dot_col[:, l] += (a[i, :] - a[j, :]) * (b[v, l] - b[u, l])
                    # This is: dot_col += np.outer(a[i, :] - a[j, :], b[v, :] - b[u, :])
                    
                    diff_a_row = a[i, :] - a[j, :]
                    b_diff_T = b[v, :] - b[u, :]
                    dot_col += np.outer(diff_a_row, b_diff_T)

        return p, cost, dot_row, dot_col

    # --- Phase 1: Hungarian-based initialization ---
    best_p = hungarian_init()
    best_cost = compute_cost(best_p)
    dot_row, dot_col = init_precomp(best_p)

    # --- Phase 2: Initial local search ---
    best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 50.0)

    # --- Phase 3: Hungarian refinement + local search ---
    if time.time() < deadline - 35.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            dot_row, dot_col = init_precomp(best_p)
            best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 25.0)

    # --- Phase 4: Second Hungarian refinement ---
    if time.time() < deadline - 20.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            dot_row, dot_col = init_precomp(best_p)
            best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 10.0)

    # --- Phase 5: Final local search ---
    if time.time() < deadline - 2.0:
        best_p, best_cost, dot_row, dot_col = local_search_fast(best_p, best_cost, dot_row, dot_col, deadline - 1.0)

    return best_p.tolist()
# EVOLVE-BLOCK-END
