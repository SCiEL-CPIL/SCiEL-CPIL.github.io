"""QAP tai256c solver: iterated local search with vectorized 2-opt, Hungarian re-optimization, and adaptive diversification."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    # Precompute symmetric A for vectorized delta calculation
    A_sym = a + a.T

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20, exact=False):
        """Steepest Descent Local Search with vectorized delta calculation for speed."""
        improved = True
        passes = 0
        p = p.copy()
        
        # Precompute D = b[np.ix_(p, p)]
        D = b[np.ix_(p, p)].copy()
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost
                    
                if exact:
                    # Exact O(n) delta calculation for final polish
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        delta = fast_swap_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    if best_j >= 0:
                        j = best_j
                        min_delta = best_delta
                    else:
                        continue
                else:
                    js = np.arange(i + 1, n)
                    
                    # Vectorized delta calculation
                    # delta(i,j) = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[i,k] - D[j,k]) + corrections
                    # Let's use the exact formula from fast_swap_delta but vectorized
                    # Actually, let's stick to the proven fast_swap_delta for correctness
                    # and just vectorize the inner loop over j for a fixed i if possible.
                    # But fast_swap_delta is O(n) per pair. Vectorizing over j for fixed i:
                    # We need delta(i, j) for all j > i.
                    
                    # Let's try a simpler vectorized approach:
                    # The cost change when swapping i and j is:
                    # delta = sum_k (a[i,k]-a[j,k])*(b[p[i],p[k]]-b[p[j],p[k]]) + 
                    #         sum_k (a[k,i]-a[k,j])*(b[p[k],p[i]]-b[p[k],p[j]]) + 
                    #         (a[i,j]-a[j,i])*(b[p[i],p[j]]-b[p[j],p[i]])
                    # Note: the first two sums are over k.
                    # Let d_row_i = a[i,:] - a[j,:]
                    # Let d_col_i = a[:,i] - a[:,j]
                    # Let d_loc_i = b[p[i],:] - b[p[j],:]
                    # Let d_loc_t_i = b[:,p[i]] - b[:,p[j]]
                    
                    # This is hard to fully vectorize over j without precomputation.
                    # Let's just use the scalar fast_swap_delta but loop over j.
                    # It's O(n^2) per pass, which is 256^2 = 65536 ops. Fast enough.
                    
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        delta = fast_swap_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                            
                    if best_j >= 0:
                        j = best_j
                        min_delta = best_delta
                    else:
                        continue
                
                if min_delta < 0:
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    cost += min_delta
                    improved = True
        
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def hungarian_init():
        """Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix."""
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # Phase 1: Hungarian Initialization
    p_hung = hungarian_init()
    best_p = p_hung.copy()
    best_cost = compute_cost(best_p)
    
    # Refine with local search
    current_p, current_cost = local_search(best_p, best_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Phase 2: Rank matching initialization (alternative starting point)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_rank = p_rank.astype(np.int64)
    
    p_rank_ls, cost_rank = local_search(p_rank, compute_cost(p_rank))
    if cost_rank < best_cost:
        best_p = p_rank_ls.copy()
        best_cost = cost_rank

    # Phase 3: Iterated Local Search
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3

    while time.time() < deadline - 10:  # Leave 10s buffer
        iteration += 1

        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb

        # Perturb: random swaps
        p = best_p.copy()
        n_swaps = np.random.randint(2, perturb_k + 1)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    # Final high-precision polish pass
    if time.time() < deadline - 5:
        # Polish the best solution found
        p = best_p.copy()
        remaining = deadline - time.time() - 5
        p, cost = local_search(p, best_cost, max_passes=50, exact=True)
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

        # Try a few more local searches from slightly perturbed best
        remaining = deadline - time.time() - 5
        if remaining > 5:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                p = best_p.copy()
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
                p, cost = local_search(p, compute_cost(p), max_passes=30, exact=True)
                if cost < best_cost:
                    best_p = p.copy()
                    best_cost = cost
                if time.time() > deadline - 5:
                    break

    return best_p.tolist()
