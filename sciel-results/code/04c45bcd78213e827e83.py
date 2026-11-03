"""QAP tai256c solver: iterated local search with vectorized steepest-descent and Hungarian re-optimization."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0  # safety margin under 120s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        """Compute cost delta for swapping positions u and v in permutation p."""
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

    def vectorized_local_search(p, cost, max_passes=50):
        """
        Vectorized steepest-descent local search.
        For each position u, compute delta for swapping with all v > u in parallel,
        pick the best improving swap, and apply it. Recompute cost incrementally.
        """
        p = p.copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for u in range(n):
                if time.time() > deadline:
                    return p, cost
                # Compute deltas for swapping u with all v != u
                # We use the fast_swap_delta but vectorize over v
                # For efficiency, compute for all v at once using numpy
                pu = p[u]
                
                # d_uv[v] = b[p[v], :] - b[pu, :] for all v
                # This is b[p, :] - b[pu, :]
                d_uv_all = b[p, :] - b[pu, :]  # shape (n, n)
                
                # s1[v] = dot(a[u,:] - a[v,:], d_uv_all[v, p])
                # a[u,:] - a[v,:] for all v: shape (n, n)
                a_diff_row = a[u, :] - a  # shape (n, n)
                
                # s1[v] = sum_k a_diff_row[v, k] * d_uv_all[v, p[k]]
                # = sum_k a_diff_row[v, k] * b[p[v], p[k]] - a_diff_row[v, k] * b[pu, p[k]]
                # Let's compute this differently
                # s1[v] = dot(a[u,:] - a[v,:], b[p[v], p] - b[pu, p])
                
                # b[p[v], p] for all v: b[np.ix_(p, p)]
                Bpp = b[np.ix_(p, p)]  # shape (n, n), Bpp[v, k] = b[p[v], p[k]]
                # b[pu, p] for all v: b[pu, p] is the same for all v
                b_pu_p = b[pu, p]  # shape (n,)
                
                # d_uv_all[v, k] = Bpp[v, k] - b_pu_p[k]
                # s1[v] = sum_k (a[u,k] - a[v,k]) * (Bpp[v,k] - b_pu_p[k])
                
                # Expand: s1[v] = sum_k a[u,k]*Bpp[v,k] - sum_k a[u,k]*b_pu_p[k]
                #            - sum_k a[v,k]*Bpp[v,k] + sum_k a[v,k]*b_pu_p[k]
                
                # Term 1: sum_k a[u,k] * Bpp[v,k] = a @ Bpp.T  (shape n)
                term1 = a @ Bpp.T  # shape (n,)
                # Term 2: sum_k a[u,k] * b_pu_p[k] = a[u] @ b_pu_p (scalar, same for all v)
                term2 = np.dot(a[u, :], b_pu_p)
                # Term 3: sum_k a[v,k] * Bpp[v,k] = sum over v of diag(a * Bpp)
                term3 = np.sum(a * Bpp, axis=1)  # shape (n,)
                # Term 4: sum_k a[v,k] * b_pu_p[k] = a @ b_pu_p (shape n)
                term4 = a @ b_pu_p  # shape (n,)
                
                s1_all = term1 - term2 - term3 + term4  # shape (n,)
                
                # Similarly for s2: s2[v] = dot(a[:,u] - a[:,v], b[p, p[v]] - b[p, pu])
                # b[p, p[v]] = Bpp[:, v]
                # b[p, pu] = Bpp[:, u] (since p[u]=pu)
                # d_vu_all[v, k] = Bpp[k, v] - Bpp[k, u]
                
                a_diff_col = a[:, u] - a.T  # shape (n, n), a_diff_col[k, v] = a[k,u] - a[k,v]
                
                # s2[v] = sum_k a_diff_col[k, v] * (Bpp[k, v] - Bpp[k, u])
                # = sum_k a_diff_col[k,v]*Bpp[k,v] - sum_k a_diff_col[k,v]*Bpp[k,u]
                
                term5 = np.sum(a_diff_col * Bpp, axis=0)  # shape (n,), term5[v] = sum_k a_diff_col[k,v]*Bpp[k,v]
                # For term6: sum_k a_diff_col[k,v]*Bpp[k,u] = a_diff_col[:, u] @ ... no
                # a_diff_col[:, u] = a[:,u] - a[:,u] = 0? No.
                # a_diff_col[k, v] = a[k, u] - a[k, v]
                # a_diff_col[:, u] = a[:, u] - a[:, u] = 0
                # So term6 for v=u is 0. For general v:
                # term6[v] = sum_k (a[k,u] - a[k,v]) * Bpp[k,u]
                # = sum_k a[k,u]*Bpp[k,u] - sum_k a[k,v]*Bpp[k,u]
                # = (a[:,u] @ Bpp[:,u]) - (a @ Bpp[:,u])
                term6_1 = np.dot(a[:, u], Bpp[:, u])  # scalar
                term6_2 = a @ Bpp[:, u]  # shape (n,)
                term6 = term6_1 - term6_2  # shape (n,)
                
                s2_all = term5 - term6  # shape (n,)
                
                # Cross term: cross[v] = (a[u,v] - a[v,u]) * (Bpp[v, u] - Bpp[u, v])
                # Bpp[v,u] = b[p[v], p[u]] = b[p[v], pu]
                # Bpp[u,v] = b[p[u], p[v]] = b[pu, p[v]]
                cross_all = (a[u, :] - a[:, u]) * (b[p, pu] - b[pu, p])  # shape (n,)
                # Wait, cross = (a[u,v] - a[v,u]) * (b[pv,pu] - b[pu,pv])
                # b[pv,pu] = b[p[v], p[u]] = Bpp[v, u]
                # b[pu,pv] = b[p[u], p[v]] = Bpp[u, v]
                # So cross[v] = (a[u,v] - a[v,u]) * (Bpp[v,u] - Bpp[u,v])
                # Bpp[v,u] = b[p[v], pu]
                # Bpp[u,v] = b[pu, p[v]]
                cross_all = (a[u, :] - a[:, u]) * (Bpp[np.arange(n), u] - Bpp[u, np.arange(n)])
                # Hmm, let me just use the direct formula
                cross_all = (a[u, :] - a.T[u, :]) * (b[p, pu] - b[pu, p])
                # a.T[u,:] = a[:,u]
                # b[p, pu] = Bpp[:, u]
                # b[pu, p] = Bpp[u, :]
                cross_all = (a[u, :] - a[:, u]) * (Bpp[:, u] - Bpp[u, :])
                
                deltas = s1_all + s2_all + cross_all  # shape (n,)
                
                # Set delta[u] to 0 (can't swap with self)
                deltas[u] = 0
                # Also, we should only consider v where the swap makes sense.
                # Actually, any v != u is valid.
                
                # Find the best v
                v_best = np.argmin(deltas)
                delta_best = deltas[v_best]
                
                if delta_best < 0 and v_best != u:
                    p[u], p[v_best] = p[v_best], p[u]
                    cost += delta_best
                    improved = True
                    # Continue to next u (first-improvement within the pass, but steepest for this u)
                    
        return p, cost

    def local_search_first_improve(p, cost, max_passes=20):
        """First-improvement local search with scalar delta computation."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def tabu_kick(p, cost, n_kicks=5):
        """Apply several random tabu-like kicks to diversify, keeping best result."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 10
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            i, j = np.random.choice(n, 2, replace=False)
            key = (min(i, j), max(i, j))
            
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            delta = fast_swap_delta(p, i, j)
            
            if not is_tabu or (cost + delta < best_cost):
                p[i], p[j] = p[j], p[i]
                cost += delta
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            iteration += 1
        
        return best_p, best_cost

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Try multiple initial solutions and keep the best after local search
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search using vectorized method
    current_p, current_cost = vectorized_local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Try a few random restarts for initial solution
    for r in range(5):
        if time.time() > deadline - 30:
            break
        p_rand = np.random.permutation(n).astype(np.int64)
        cost_rand = compute_cost(p_rand)
        p_rand, cost_rand = vectorized_local_search(p_rand, cost_rand)
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    while time.time() < deadline - 15:
        iteration += 1

        # Perturb: combine random swaps with tabu kick
        p = best_p.copy()
        
        n_swaps = np.random.randint(3, 10)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        cost = compute_cost(p)
        p, cost = tabu_kick(p, cost, n_kicks=3)

        # Local search from perturbed
        p, cost = vectorized_local_search(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 4 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = vectorized_local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Final polish
    if time.time() < deadline - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = local_search_first_improve(final_p, final_cost)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost

    return best_p.tolist()
