"""QAP solver: Hungarian init + O(1) delta local search with incremental precomputation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].

    Strategy:
    1. Initialize via Hungarian on C = a @ b (linear relaxation).
    2. Local search using precomputed dot_row/dot_col for O(1) delta lookups.
    3. Incremental O(n) updates to precomputed matrices after each swap.
    4. Iterate: Local Search -> Hungarian Refinement -> Local Search.
    5. Final exhaustive polish pass.
    """
    n = len(a)
    deadline = time.time() + 112.0  # safety margin before 120s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def _init_precomp(p):
        """Precompute b_p[k,l]=b[p[k],l], b_pT[k,l]=b[l,p[k]], dot_row, dot_col."""
        b_p = b[p, :].copy()       # b_p[i, j] = b[p[i], j]
        b_pT = b[:, p].copy()      # b_pT[i, j] = b[j, p[i]]
        # dot_row[i, j] = sum_k a[i,k] * b[p[k], j] = (a @ b_p)[i, j]
        dot_row = a.astype(np.float64) @ b_p.astype(np.float64)
        # dot_col[i, j] = sum_k a[k,i] * b[j, p[k]] = (a.T @ b_pT.T)[i, j] = (b_pT @ a)[i, j]... 
        # Actually: dot_col[i,j] = sum_k a[k,i] * b[j, p[k]]
        # b_pT[k,j] = b[j, p[k]], so dot_col[i,j] = sum_k a[k,i] * b_pT[k,j] = (a.T @ b_pT)[i,j]
        dot_col = a.T.astype(np.float64) @ b_pT.astype(np.float64)
        return b_p, b_pT, dot_row, dot_col

    def _update_precomp_swap(i, j, p, b_p, b_pT, dot_row, dot_col):
        """Incrementally update precomputed structures after swapping positions i and j."""
        u = p[i]
        v = p[j]
        if u == v:
            return

        # Swap rows in b_p
        b_p[[i, j], :] = b_p[[j, i], :]
        # Swap columns in b_pT
        b_pT[:, [i, j]] = b_pT[:, [j, i]]

        # Update dot_row: dot_row[k, l] = sum_m a[k,m] * b[p[m], l]
        # Only rows i and j of dot_row change (since a[i,:] and a[j,:] are fixed,
        # but the p[m] for m=i,j changed)
        # Actually dot_row depends on ALL p[m] through b[p[m], l].
        # When we swap p[i] and p[j], for each row k:
        #   dot_row[k, l] changes by: a[k,i]*(b[v,l]-b[u,l]) + a[k,j]*(b[u,l]-b[v,l])
        #                            = (a[k,i]-a[k,j])*(b[v,l]-b[u,l])
        # So: dot_row[:, :] += (a[:, i] - a[:, j])[:, None] * (b[v, :] - b[u, :])[None, :]
        diff_a = (a[:, i] - a[:, j]).astype(np.float64)
        diff_b = (b[v, :] - b[u, :]).astype(np.float64)
        dot_row += np.outer(diff_a, diff_b)

        # Update dot_col: dot_col[k, l] = sum_m a[m,k] * b[l, p[m]]
        # When we swap p[i] and p[j], for each col k:
        #   dot_col[k, l] changes by: a[i,k]*(b[l,v]-b[l,u]) + a[j,k]*(b[l,u]-b[l,v])
        #                            = (a[i,k]-a[j,k])*(b[l,v]-b[l,u])
        # So: dot_col[:, :] += (a[i, :] - a[j, :])[None, :] * (b[:, v] - b[:, u])[:, None]
        # Wait: dot_col[k,l] = sum_m a[m,k] * b[l, p[m]]
        # Change for term m=i: a[i,k]*(b[l,v] - b[l,u])
        # Change for term m=j: a[j,k]*(b[l,u] - b[l,v])
        # Total: (a[i,k] - a[j,k]) * (b[l,v] - b[l,u])
        # So dot_col[k,l] += (a[i,k]-a[j,k])*(b[l,v]-b[l,u])
        # In matrix form: dot_col += (a[i,:]-a[j,:])[None,:] * (b[:,v]-b[:,u])[:,None]
        diff_a2 = (a[i, :] - a[j, :]).astype(np.float64)
        diff_b2 = (b[:, v] - b[:, u]).astype(np.float64)
        dot_col += np.outer(diff_b2, diff_a2)

    def swap_delta_fast(i, j, p, b_p, b_pT, dot_row, dot_col):
        """Compute delta for swapping positions i and j. O(1) with precomputed matrices."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0

        # delta = sum_{k,l} a[k,l] * (b[p_new[k], p_new[l]] - b[p[k], p[l]])
        # Only positions i and j change.
        # Terms involving i or j:
        # 1. (i,i): a[i,i]*(b[v,v]-b[u,u])
        # 2. (i,j): a[i,j]*(b[v,u]-b[u,v])
        # 3. (j,i): a[j,i]*(b[u,v]-b[v,u])
        # 4. (j,j): a[j,j]*(b[u,u]-b[v,v])
        # 5. (i,k) k!=i,j: a[i,k]*(b[v,p[k]]-b[u,p[k]])
        # 6. (j,k) k!=i,j: a[j,k]*(b[u,p[k]]-b[v,p[k]])
        # 7. (k,i) k!=i,j: a[k,i]*(b[p[k],v]-b[p[k],u])
        # 8. (k,j) k!=i,j: a[k,j]*(b[p[k],u]-b[p[k],v])

        delta = 0.0
        # Terms 1-4
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        delta += a[j, j] * (b[u, u] - b[v, v])

        # Terms 5: sum_{k!=i,j} a[i,k]*(b[v,p[k]]-b[u,p[k]])
        # = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,u]-b[u,u])
        # sum_k a[i,k]*b[v,p[k]] = sum_k a[i,k]*b_pT[k,v] = dot_row[i,v]
        # sum_k a[i,k]*b[u,p[k]] = sum_k a[i,k]*b_pT[k,u] = dot_row[i,u]
        delta += (dot_row[i, v] - dot_row[i, u]) - a[i, i] * (b[v, u] - b[u, u]) - a[i, j] * (b[v, u] - b[u, u])

        # Terms 6: sum_{k!=i,j} a[j,k]*(b[u,p[k]]-b[v,p[k]])
        # = sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]]) - a[j,i]*(b[u,u]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        # sum_k a[j,k]*b[u,p[k]] = dot_row[j,u]
        # sum_k a[j,k]*b[v,p[k]] = dot_row[j,v]
        delta += (dot_row[j, u] - dot_row[j, v]) - a[j, i] * (b[u, u] - b[v, u]) - a[j, j] * (b[u, u] - b[v, v])

        # Terms 7: sum_{k!=i,j} a[k,i]*(b[p[k],v]-b[p[k],u])
        # = sum_k a[k,i]*(b[p[k],v]-b[p[k],u]) - a[i,i]*(b[u,v]-b[u,u]) - a[j,i]*(b[v,v]-b[v,u])
        # sum_k a[k,i]*b[p[k],v] = sum_k a[k,i]*b_p[k,v] = dot_col[i,v]
        # sum_k a[k,i]*b[p[k],u] = sum_k a[k,i]*b_p[k,u] = dot_col[i,u]
        delta += (dot_col[i, v] - dot_col[i, u]) - a[i, i] * (b[u, v] - b[u, u]) - a[j, i] * (b[v, v] - b[v, u])

        # Terms 8: sum_{k!=i,j} a[k,j]*(b[p[k],u]-b[p[k],v])
        # = sum_k a[k,j]*(b[p[k],u]-b[p[k],v]) - a[i,j]*(b[u,u]-b[u,v]) - a[j,j]*(b[v,u]-b[v,v])
        # sum_k a[k,j]*b[p[k],u] = dot_col[j,u]
        # sum_k a[k,j]*b[p[k],v] = dot_col[j,v]
        delta += (dot_col[j, u] - dot_col[j, v]) - a[i, j] * (b[u, u] - b[u, v]) - a[j, j] * (b[v, u] - b[v, v])

        return delta

    def local_search_2swap(p, cost, b_p, b_pT, dot_row, dot_col, max_time, num_trials=30000):
        """Hill-climbing 2-swap local search with O(1) delta."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p, b_p, b_pT, dot_row, dot_col)
                if d < -0.5:
                    p[i], p[j] = p[j], p[i]
                    cost += int(d)
                    _update_precomp_swap(i, j, p, b_p, b_pT, dot_row, dot_col)
                    improved = True
        return p, cost, b_p, b_pT, dot_row, dot_col

    def local_search_exhaustive(p, cost, b_p, b_pT, dot_row, dot_col, max_time):
        """Exhaustive 2-swap local search: check all pairs in order."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    d = swap_delta_fast(i, j, p, b_p, b_pT, dot_row, dot_col)
                    if d < -0.5:
                        p[i], p[j] = p[j], p[i]
                        cost += int(d)
                        _update_precomp_swap(i, j, p, b_p, b_pT, dot_row, dot_col)
                        improved = True
        return p, cost, b_p, b_pT, dot_row, dot_col

    def hungarian_init():
        """Initialize via Hungarian on C = a @ b (linear relaxation of QAP)."""
        C = a.astype(np.float64) @ b.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_refine(p):
        """Build linear cost matrix from current p, solve assignment, return new p."""
        b_col_p = b[:, p]   # shape (n,n): b_col_p[j,k] = b[j,p[k]]
        b_row_p = b[p, :]   # shape (n,n): b_row_p[k,j] = b[p[k],j]
        C = a.astype(np.float64) @ b_col_p.astype(np.float64).T + a.T.astype(np.float64) @ b_row_p.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    # --- Phase 1: Hungarian-based initialization ---
    best_p = hungarian_init()
    best_cost = compute_cost(best_p)
    b_p, b_pT, dot_row, dot_col = _init_precomp(best_p)

    # --- Phase 2: Initial local search ---
    best_p, best_cost, b_p, b_pT, dot_row, dot_col = local_search_2swap(
        best_p, best_cost, b_p, b_pT, dot_row, dot_col, deadline - 55.0)

    # --- Phase 3: Hungarian refinement + local search ---
    if time.time() < deadline - 40.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            b_p, b_pT, dot_row, dot_col = _init_precomp(best_p)
            best_p, best_cost, b_p, b_pT, dot_row, dot_col = local_search_2swap(
                best_p, best_cost, b_p, b_pT, dot_row, dot_col, deadline - 25.0)

    # --- Phase 4: Second Hungarian refinement ---
    if time.time() < deadline - 15.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            b_p, b_pT, dot_row, dot_col = _init_precomp(best_p)
            best_p, best_cost, b_p, b_pT, dot_row, dot_col = local_search_2swap(
                best_p, best_cost, b_p, b_pT, dot_row, dot_col, deadline - 8.0)

    # --- Phase 5: Final exhaustive polish ---
    if time.time() < deadline - 2.0:
        best_p, best_cost, b_p, b_pT, dot_row, dot_col = local_search_exhaustive(
            best_p, best_cost, b_p, b_pT, dot_row, dot_col, deadline - 1.0)

    return best_p.tolist()
# EVOLVE-BLOCK-END
