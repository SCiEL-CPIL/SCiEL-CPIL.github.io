"""Rewritten QAP solver: continuous relaxation + Hungarian + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching initial permutation.
    2. Hungarian-based linearization refinement (multiple rounds).
    3. Fast local search with precomputed swap deltas.
    4. 3-swap moves to escape 2-swap local optima.
    """
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    def elapsed():
        return time.time() - t0

    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    # --- Initial solution via rank-matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # --- Fast swap_delta using precomputed b_p = b[p[...]] ---
    def _update_b_p(best_p):
        """Precompute b_p[k,l] = b[p[k],l] and b_pT[k,l] = b[l,p[k]]."""
        return b[best_p, :], b[:, best_p]

    b_p, b_pT = _update_b_p(best_p)

    def swap_delta_fast(i, j):
        """Compute delta for swapping positions i and j in best_p. O(n) time."""
        u = best_p[i]
        v = best_p[j]
        if u == v:
            return 0
        
        delta = 0.0
        
        # Diagonal/cross terms
        delta += a[i,i]*(b[v,v] - b[u,u])
        delta += a[i,j]*(b[v,u] - b[u,v])
        delta += a[j,i]*(b[u,v] - b[v,u])
        delta += a[j,j]*(b[u,u] - b[v,v])
        
        # Row i, k not in {i,j}
        row_i_diff = b_pT[:, v] - b_pT[:, u]
        delta += np.dot(a[i], row_i_diff) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,u]-b[u,u])
        
        # Row j, k not in {i,j}
        row_j_diff = b_pT[:, u] - b_pT[:, v]
        delta += np.dot(a[j], row_j_diff) - a[j,i]*(b[u,u]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        # Col i, k not in {i,j}
        col_i_diff = b_p[:, v] - b_p[:, u]
        delta += np.dot(a[:, i], col_i_diff) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[u,v]-b[v,u])
        
        # Col j, k not in {i,j}
        col_j_diff = b_p[:, u] - b_p[:, v]
        delta += np.dot(a[:, j], col_j_diff) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        return delta

    # --- Local search with fast swap_delta ---
    def local_search_2swap(p, cost, b_p, b_pT, max_time):
        """Hill-climbing 2-swap local search with precomputed b_p, b_pT."""
        nonlocal best_p, best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            num_trials = 30000
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j)
                if d < -0.5:
                    u, v = p[i], p[j]
                    p[i], p[j] = v, u
                    cost += d
                    b_p[i, :], b_p[j, :] = b_p[j, :].copy(), b_p[i, :].copy()
                    b_pT[:, i], b_pT[:, j] = b_pT[:, j].copy(), b_pT[:, i].copy()
                    best_p = p
                    best_cost = cost
                    improved = True
        return p, cost

    # --- Hungarian refinement ---
    def hungarian_refine(p, cost, max_time):
        """Use Hungarian on linearized cost to improve solution."""
        nonlocal best_p, best_cost
        if time.time() > max_time:
            return p, cost
        
        b_p, b_pT = _update_b_p(p)
        
        # Build linear cost matrix
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < cost:
            p = p_hung
            cost = cost_hung
            best_p = p.copy()
            best_cost = cost
        return p, cost

    # --- 3-swap moves ---
    def swap3_delta(i, j, k):
        """Compute delta for 3-cycle: p[i]->p[j]->p[k]->p[i]."""
        u, v, w = best_p[i], best_p[j], best_p[k]
        if u == v or v == w or u == w:
            return 0
        
        delta = 0.0
        p_new = best_p.copy()
        p_new[i], p_new[j], p_new[k] = v, w, u
        
        for x in [i, j, k]:
            for y in [i, j, k]:
                delta += a[x,y]*(b[p_new[x],p_new[y]] - b[best_p[x],best_p[y]])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = mask[j] = mask[k] = False
        others = np.where(mask)[0]
        
        for x in [i, j, k]:
            diff = b[p_new[x], others] - b[best_p[x], others]
            delta += np.dot(diff, a[x, others])
        
        for y in [i, j, k]:
            diff = b[best_p[others], p_new[y]] - b[best_p[others], best_p[y]]
            delta += np.dot(diff, a[others, y])
        
        return delta

    def local_search_3swap(p, cost, b_p, b_pT, max_time):
        """Try 3-cycle moves to escape 2-swap local optima."""
        nonlocal best_p, best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            num_trials = 5000
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j, k = np.random.randint(0, n, 3)
                if len(set([i, j, k])) < 3:
                    continue
                d = swap3_delta(i, j, k)
                if d < -0.5:
                    u, v, w = p[i], p[j], p[k]
                    p[i], p[j], p[k] = v, w, u
                    cost += d
                    b_p[i, :], b_p[j, :], b_p[k, :] = b_p[j, :].copy(), b_p[k, :].copy(), b_p[i, :].copy()
                    b_pT[:, i], b_pT[:, j], b_pT[:, k] = b_pT[:, j].copy(), b_pT[:, k].copy(), b_pT[:, i].copy()
                    best_p = p
                    best_cost = cost
                    improved = True
        return p, cost

    # --- Main loop ---
    # Phase 1: Initial local search
    best_p, best_cost = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 30.0)
    
    # Phase 2: Hungarian refinement
    if time.time() < deadline - 20.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost, deadline - 20.0)
    
    # Phase 3: More local search with 3-swaps
    if time.time() < deadline - 10.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_3swap(best_p, best_cost, b_p, b_pT, deadline - 10.0)
    
    # Phase 4: Final 2-swap polish
    if time.time() < deadline - 2.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 2.0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
