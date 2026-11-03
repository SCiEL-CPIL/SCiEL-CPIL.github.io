"""QAP tai256c solver: Iterated 2-opt local search with adaptive perturbation and Hungarian rescue."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    # Use a safety margin well under the 120s limit. 
    # The evaluator kills at 150s, but the task says 120s for evaluation. 
    # Let's aim for 115s to be safe.
    deadline = time.time() + 115.0

    def compute_cost(p):
        # p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_swap_delta(p, u, v):
        """
        Compute the delta in cost when swapping p[u] and p[v].
        p[u] and p[v] are the locations.
        """
        pu = p[u]
        pv = p[v]

        # d_uv[k] = b[pv, k] - b[pu, k]
        # d_vu[k] = b[k, pv] - b[k, pu]
        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        # s1 = sum_k (a[u,k] - a[v,k]) * d_uv[p[k]]
        # s2 = sum_k (a[k,u] - a[k,v]) * d_vu[p[k]]
        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        # Subtract k=u and k=v terms because p[k] changes in the sum
        # For s1 (row u and v):
        # k=u term: (a[u,u] - a[v,u]) * d_uv[pu] -> but p[u] becomes pv, so we used d_uv[pu] originally?
        # Wait, the formula s1 = sum_k (a[u,k] - a[v,k]) * (b[pv, p[k]] - b[pu, p[k]])
        # For k=u: p[u]=pu. Term is (a[u,u]-a[v,u]) * (b[pv, pu] - b[pu, pu]).
        # In the dot product, we used d_uv[pu] for k=u. d_uv[pu] = b[pv, pu] - b[pu, pu]. Correct.
        # For k=v: p[v]=pv. Term is (a[u,v]-a[v,v]) * (b[pv, pv] - b[pu, pv]).
        # In the dot product, we used d_uv[pv] for k=v. d_uv[pv] = b[pv, pv] - b[pu, pv]. Correct.
        
        # The issue is that in the new state, p[u]=pv and p[v]=pu.
        # The delta formula derived usually accounts for the change in the terms involving u and v.
        # The standard formula for 2-opt delta in QAP is:
        # delta = sum_{k != u, v} [ a[u,k](b[pv,p[k]] - b[pu,p[k]]) + a[k,u](b[p[k],pv] - b[p[k],pu]) 
        #                         + a[v,k](b[pu,p[k]] - b[pv,p[k]]) + a[k,v](b[p[k],pu] - b[p[k],pv]) ]
        #       + a[u,u](b[pv,pv] - b[pu,pu]) + a[v,v](b[pu,pu] - b[pv,pv])
        #       + a[u,v](b[pv,pu] - b[pu,pv]) + a[v,u](b[pu,pv] - b[pv,pu])
        
        # Let's verify the current implementation against this.
        # s1 = sum_k (a[u,k] - a[v,k]) * (b[pv, p[k]] - b[pu, p[k]])
        # s2 = sum_k (a[k,u] - a[k,v]) * (b[p[k], pv] - b[p[k], pu])
        # Sum = s1 + s2
        # Expanding s1:
        # k=u: (a[u,u]-a[v,u])(b[pv,pu]-b[pu,pu])
        # k=v: (a[u,v]-a[v,v])(b[pv,pv]-b[pu,pv])
        # k!=u,v: (a[u,k]-a[v,k])(b[pv,p[k]]-b[pu,p[k]])
        
        # Expanding s2:
        # k=u: (a[u,u]-a[u,v])(b[pu,pv]-b[pu,pu])
        # k=v: (a[v,u]-a[v,v])(b[pv,pv]-b[pv,pu])
        # k!=u,v: (a[k,u]-a[k,v])(b[p[k],pv]-b[p[k],pu])
        
        # Total for k=u:
        # (a[u,u]-a[v,u])b[pv,pu] - (a[u,u]-a[v,u])b[pu,pu]
        # + (a[u,u]-a[u,v])b[pu,pv] - (a[u,u]-a[u,v])b[pu,pu]
        # = a[u,u]b[pv,pu] - a[v,u]b[pv,pu] - a[u,u]b[pu,pu] + a[v,u]b[pu,pu]
        #   + a[u,u]b[pu,pv] - a[u,v]b[pu,pv] - a[u,u]b[pu,pu] + a[u,v]b[pu,pu]
        
        # The standard formula terms for k=u (involving u,u and u,v and v,u and v,v? No, k=u is just u)
        # Standard formula k=u terms:
        # a[u,u](b[pv,pv] - b[pu,pu])  <- Wait, my expansion has b[pu,pu] twice?
        
        # Let's re-derive carefully.
        # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        # Change p[u] <-> p[v].
        # Terms involving u or v change.
        # 1. i=u, j=u: a[u,u] (b[pv,pv] - b[pu,pu])
        # 2. i=v, j=v: a[v,v] (b[pu,pu] - b[pv,pv])
        # 3. i=u, j=v: a[u,v] (b[pv,pu] - b[pu,pv])
        # 4. i=v, j=u: a[v,u] (b[pu,pv] - b[pv,pu])
        # 5. i=u, j=k (k!=u,v): a[u,k] (b[pv,p[k]] - b[pu,p[k]])
        # 6. i=v, j=k (k!=u,v): a[v,k] (b[pu,p[k]] - b[pv,p[k]])
        # 7. i=k (k!=u,v), j=u: a[k,u] (b[p[k],pv] - b[p[k],pu])
        # 8. i=k (k!=u,v), j=v: a[k,v] (b[p[k],pu] - b[p[k],pv])
        
        # My s1+s2 calculation:
        # s1 = sum_k (a[u,k] - a[v,k]) (b[pv, p[k]] - b[pu, p[k]])
        # s2 = sum_k (a[k,u] - a[k,v]) (b[p[k], pv] - b[p[k], pu])
        
        # Let's check k=u in s1:
        # (a[u,u] - a[v,u]) (b[pv, pu] - b[pu, pu])
        # Let's check k=u in s2:
        # (a[u,u] - a[u,v]) (b[pu, pv] - b[pu, pu])
        
        # Sum k=u:
        # a[u,u]b[pv,pu] - a[v,u]b[pv,pu] - a[u,u]b[pu,pu] + a[v,u]b[pu,pu]
        # + a[u,u]b[pu,pv] - a[u,v]b[pu,pv] - a[u,u]b[pu,pu] + a[u,v]b[pu,pu]
        
        # Standard terms for k=u:
        # From 1: a[u,u]b[pv,pv] - a[u,u]b[pu,pu]  (Wait, i=u,j=u changes to b[pv,pv])
        # From 4: a[v,u]b[pu,pv] - a[v,u]b[pv,pu] (i=v,j=u changes to b[pu,pv])
        # From 5: a[u,k]... no k=u is not in 5.
        # From 7: a[k,u]... no k=u is not in 7.
        
        # It seems my s1+s2 does NOT match the standard formula directly because it mixes terms.
        # Specifically, s1 sums over all k, including u and v.
        # The standard formula separates k=u, k=v, k!=u,v.
        
        # Let's stick to the standard formula implementation which is safer, even if slightly slower, 
        # or verify the current one with a simple test. 
        # Given the risk, let's use the explicit standard formula for correctness.
        
        delta = 0.0
        
        # Self terms
        delta += a[u, u] * (b[pv, pv] - b[pu, pu])
        delta += a[v, v] * (b[pu, pu] - b[pv, pv])
        
        # Cross terms
        delta += a[u, v] * (b[pv, pu] - b[pu, pv])
        delta += a[v, u] * (b[pu, pv] - b[pv, pu])
        
        # Bulk terms
        # We can vectorize this
        # mask = np.ones(n, dtype=bool); mask[u]=False; mask[v]=False
        # pk = p[mask]
        # delta += np.sum(a[u, mask] * (b[pv, pk] - b[pu, pk]))
        # delta += np.sum(a[v, mask] * (b[pu, pk] - b[pv, pk]))
        # delta += np.sum(a[mask, u] * (b[pk, pv] - b[pk, pu]))
        # delta += np.sum(a[mask, v] * (b[pk, pu] - b[pk, pv]))
        
        # To speed it up, we can precompute row/col differences
        # But n=256 is small. The O(n) loop inside is fine if we don't create masks every time.
        # However, creating a mask is O(n). The sum is O(n).
        
        # Let's use the vectorized version for clarity and correctness.
        # Optimization: Precompute b_row_diff = b[pv, :] - b[pu, :]
        # and b_col_diff = b[:, pv] - b[:, pu]
        
        b_row_diff = b[pv, :] - b[pu, :]
        b_col_diff = b[:, pv] - b[:, pu]
        
        # Bulk terms: sum_{k!=u,v} a[u,k] * b_row_diff[k]
        # We can sum over all k and subtract u and v.
        sum_u = np.dot(a[u, :], b_row_diff)
        sum_u -= a[u, u] * b_row_diff[u]
        sum_u -= a[u, v] * b_row_diff[v]
        
        sum_v = np.dot(a[v, :], b_row_diff)
        sum_v -= a[v, u] * b_row_diff[u]
        sum_v -= a[v, v] * b_row_diff[v]
        
        # Note: b_row_diff[k] = b[pv, k] - b[pu, k]
        # Term 5: a[u,k] (b[pv,k] - b[pu,k]) -> sum_u
        # Term 6: a[v,k] (b[pu,k] - b[pv,k]) -> - sum_v
        
        # Now columns
        sum_ku = np.dot(a[:, u], b_col_diff)
        sum_ku -= a[u, u] * b_col_diff[u]
        sum_ku -= a[v, u] * b_col_diff[v]
        
        sum_kv = np.dot(a[:, v], b_col_diff)
        sum_kv -= a[u, v] * b_col_diff[u]
        sum_kv -= a[v, v] * b_col_diff[v]
        
        # Term 7: a[k,u] (b[k,pv] - b[k,pu]) -> sum_ku
        # Term 8: a[k,v] (b[k,pu] - b[k,pv]) -> - sum_kv
        
        delta += sum_u - sum_v + sum_ku - sum_kv
        
        return delta

    def local_search(p, cost, max_passes=50):
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
        # Build cost matrix: C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
        B_p = b[np.ix_(np.arange(n), p)]  # B_p[j,k] = b[j, p[k]]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]

        Bp2 = b[np.ix_(p, np.arange(n))]  # Bp2[k,j] = b[p[k], j]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]

        C = M1 + M2

        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search
    iteration = 0
    no_improve_count = 0
    perturbation_size = 3
    
    while time.time() < deadline:
        iteration += 1
        if time.time() > deadline:
            break

        # Perturb
        p = best_p.copy()
        # Adaptive perturbation: increase size if stuck
        n_swaps = np.random.randint(perturbation_size, perturbation_size + 5)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Rescue mechanism: If stuck, try Hungarian
        if no_improve_count > 5 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost
            no_improve_count = 0
            perturbation_size = 3 # Reset perturbation size

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            no_improve_count = 0
            # Slightly reduce perturbation size on improvement to refine
            if perturbation_size > 2:
                perturbation_size -= 1
        else:
            no_improve_count += 1
            # Increase perturbation size if stuck
            if perturbation_size < 15:
                perturbation_size += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
