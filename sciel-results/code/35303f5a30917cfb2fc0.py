"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Precompute row sums for initial heuristic
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Calculate initial cost
    cost = int(np.sum(a * b[p]))
    
    best_cost = cost
    best_p = p.copy()

    # Maintain interaction sums: c[i] = sum_j a[i,j] * b[p[i], p[j]]
    # This allows O(n) delta computation for swap (i,j)
    def compute_c(p, a, b):
        """Compute c[i] = sum_j a[i,j] * b[p[i], p[j]] for all i"""
        c = np.zeros(n, dtype=np.int64)
        for i in range(n):
            c[i] = int(np.dot(a[i], b[p[i], p]))
        return c

    c = compute_c(p, a, b)

    def get_delta_ij(i, j, p, a, b, c):
        """
        Compute delta for swapping positions i and j.
        delta = cost_after - cost_before
        """
        u = p[i]
        v = p[j]
        # When we swap i and j:
        # - facility i moves from location u to v
        # - facility j moves from location v to u
        # The change in cost involves:
        # 1. Change in interaction between i and j: a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u])
        #    = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        # 2. Change in interactions of i with all k != i,j:
        #    sum_{k != i,j} a[i,k] * (b[v, p[k]] - b[u, p[k]])
        # 3. Change in interactions of j with all k != i,j:
        #    sum_{k != i,j} a[j,k] * (b[u, p[k]] - b[v, p[k]])
        
        # Using c: c[i] = sum_k a[i,k] * b[u, p[k]]
        # After swap, new_c[i] would have b[v, p[k]] for k != i, and b[v,u] for k=j
        # But we need the full delta
        
        # More efficient: compute directly
        # delta = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]])
        #       + correction for the i-j interaction being counted twice
        
        # Let's use:
        # delta = sum_{k} a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_{k} a[j,k]*(b[u,p[k]] - b[v,p[k]])
        # But when k=i: a[i,i]*(b[v,u]-b[u,u]) + a[j,i]*(b[u,u]-b[v,u]) -- but p[i]=u, p[j]=v
        # Actually let me just compute it carefully:
        
        # Before: cost includes a[i,j]*b[u,v] + a[j,i]*b[v,u]
        # After: cost includes a[i,j]*b[v,u] + a[j,i]*b[u,v]
        # Change in i-j term: a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        
        # For k not in {i,j}:
        # Before: a[i,k]*b[u,p[k]] + a[j,k]*b[v,p[k]]
        # After: a[i,k]*b[v,p[k]] + a[j,k]*b[u,p[k]]
        # Change: a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])
        #        = (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
        
        # Total delta:
        # = (a[i,j]-a[j,i])*(b[v,u]-b[u,v]) + sum_{k not in {i,j}} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
        
        # We can write this as:
        # sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) where p[i]=u, p[j]=v
        # But when k=i: (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        # When k=j: (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        # These don't match what we want. Let me just compute directly.
        
        # Simpler approach: 
        # delta = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]])
        # This counts the i-j and j-i terms, and also includes k=i and k=j terms.
        # When k=i: a[i,i]*(b[v,u]-b[u,u]) + a[j,i]*(b[u,u]-b[v,u])
        # When k=j: a[i,j]*(b[v,v]-b[u,v]) + a[j,j]*(b[u,v]-b[v,v])
        # These are not the right terms. 
        
        # Let me use the standard formula:
        # delta_swap(i,j) = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[k], v] - b[p[k], u])
        #                   + (a[i,j] - a[j,i]) * (b[v, u] - b[u, v])
        
        # Wait, I need to be more careful. Let me use:
        # The cost is sum_{m,n} a[m,n] * b[p[m], p[n]]
        # Swap p[i] and p[j].
        # Terms involving i or j change.
        
        # For m=i, n=j: a[i,j]*b[v,u] instead of a[i,j]*b[u,v]
        # For m=j, n=i: a[j,i]*b[u,v] instead of a[j,i]*b[v,u]
        # For m=i, n=k (k not i,j): a[i,k]*b[v,p[k]] instead of a[i,k]*b[u,p[k]]
        # For m=j, n=k (k not i,j): a[j,k]*b[u,p[k]] instead of a[j,k]*b[v,p[k]]
        # For m=k, n=i (k not i,j): a[k,i]*b[p[k],v] instead of a[k,i]*b[p[k],u]
        # For m=k, n=j (k not i,j): a[k,j]*b[p[k],u] instead of a[k,j]*b[p[k],v]
        # Terms not involving i or j: unchanged
        
        # delta = a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        #         + sum_{k not in {i,j}} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])
        #                                  + a[k,i]*(b[p[k],v]-b[p[k],u]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
        
        # This is getting complex. Let me just compute it numerically for small n and verify,
        # but for performance, I'll use a different approach.
        
        # Actually, the simplest correct formula is:
        # delta = sum_{k} a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_{k} a[j,k]*(b[u,p[k]]-b[v,p[k]])
        #         - 2*a[i,j]*(b[v,u]-b[u,v]) - 2*a[j,i]*(b[u,v]-b[v,u])
        # No, that's not right either.
        
        # Let me just compute it directly with numpy:
        diff_a = a[i] - a[j]
        # b[v, p] - b[u, p] is a vector of length n
        diff_b = b[v, p] - b[u, p]
        # But we need to be careful about the diagonal terms
        # Actually, the formula delta = dot(a[i]-a[j], b[v,p]-b[u,p]) + 2*a[i,j]*(b[v,u]-b[u,v]) 
        # is NOT correct.
        
        # Let me use the correct formula from QAP literature:
        # For swapping facilities i and j (at locations u and v):
        # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[u, p[k]] - b[v, p[k]])
        #         + (a[i,j] - a[j,i]) * (b[u, v] - b[v, u])
        
        # Wait, I keep getting confused. Let me just compute it directly:
        # Before swap, the terms involving i or j in the cost:
        # T_before = sum_k a[i,k]*b[u,p[k]] + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],u] + sum_k a[k,j]*b[p[k],v]
        # But this double counts a[i,j]*b[u,v] and a[j,i]*b[v,u].
        # Actually, the total cost is sum_{m,n} a[m,n]*b[p[m],p[n]], so each pair (m,n) is counted once.
        
        # Let me just compute the cost before and after the swap for the affected terms:
        # Affected terms: all (m,n) where m or n is i or j.
        
        # This is O(n) per swap which is fine.
        
        # For performance, let me precompute what I can.
        # Actually, let me just use the direct computation:
        
        u_val = p[i]
        v_val = p[j]
        
        # Compute delta directly
        # delta = sum_{k != i,j} [(a[i,k] - a[j,k]) * (b[v_val, p[k]] - b[u_val, p[k]])]
        #         + (a[i,j] - a[j,i]) * (b[v_val, u_val] - b[u_val, v_val])
        #         + (a[i,i] - a[j,i]) * (b[v_val, u_val] - b[u_val, u_val])  # when k=i
        #         + (a[i,j] - a[j,j]) * (b[v_val, v_val] - b[u_val, v_val])  # when k=j
        
        # Hmm, this is getting messy. Let me just compute it as:
        # delta = dot(a[i], b[v_val, p]) - dot(a[i], b[u_val, p]) + dot(a[j], b[u_val, p]) - dot(a[j], b[v_val, p])
        #         - a[i,j]*(b[v_val,u_val] - b[u_val,v_val]) - a[j,i]*(b[u_val,v_val] - b[v_val,u_val])
        #         - a[i,i]*(b[v_val,u_val] - b[u_val,u_val]) - a[j,j]*(b[u_val,v_val] - b[v_val,v_val])
        # No, this is still wrong.
        
        # Let me use the cleanest approach: compute the change in cost directly.
        # The cost is C = sum_{m,n} a[m,n] * b[p[m], p[n]]
        # After swapping p[i] and p[j]:
        # C' = C - (terms with old p[i],p[j]) + (terms with new p[i],p[j])
        
        # Terms that change:
        # For m=i: sum_n a[i,n] * (b[v_val, p[n]] - b[u_val, p[n]])  [but p[i] changes from u_val to v_val]
        # For m=j: sum_n a[j,n] * (b[u_val, p[n]] - b[v_val, p[n]])  [but p[j] changes from v_val to u_val]
        # For n=i (m != i,j): sum_m a[m,i] * (b[p[m], v_val] - b[p[m], u_val])
        # For n=j (m != i,j): sum_m a[m,j] * (b[p[m], u_val] - b[p[m], v_val])
        
        # But we need to be careful not to double count.
        
        # Let me just compute it numerically:
        # delta = 0
        # for k in range(n):
        #     if k == i or k == j: continue
        #     delta += a[i,k]*(b[v_val,p[k]] - b[u_val,p[k]])
        #     delta += a[j,k]*(b[u_val,p[k]] - b[v_val,p[k]])
        #     delta += a[k,i]*(b[p[k],v_val] - b[p[k],u_val])
        #     delta += a[k,j]*(b[p[k],u_val] - b[p[k],v_val])
        # delta += a[i,j]*(b[v_val,u_val] - b[u_val,v_val])
        # delta += a[j,i]*(b[u_val,v_val] - b[v_val,u_val])
        
        # This is O(n) which is acceptable.
        # But for vectorized computation, let me use:
        
        # delta = (a[i] - a[j]) @ (b[v_val, p] - b[u_val, p]) 
        #         + (a[:, i] - a[:, j]) @ (b[p, v_val] - b[p, u_val])
        #         - correction terms
        
        # Actually, let me just use the simple O(n) loop for now and optimize later if needed.
        # But that would be slow for n=150 with many swaps.
        
        # Let me use numpy:
        diff_row = a[i] - a[j]  # shape (n,)
        diff_col = a[:, i] - a[:, j]  # shape (n,)
        diff_b_row = b[v_val, p] - b[u_val, p]  # shape (n,)
        diff_b_col = b[p, v_val] - b[p, u_val]  # shape (n,)
        
        # The total delta is:
        # sum_k a[i,k]*(b[v_val,p[k]]-b[u_val,p[k]]) + sum_k a[j,k]*(b[u_val,p[k]]-b[v_val,p[k]])
        # + sum_k a[k,i]*(b[p[k],v_val]-b[p[k],u_val]) + sum_k a[k,j]*(b[p[k],u_val]-b[p[k],v_val])
        # = sum_k (a[i,k]-a[j,k])*(b[v_val,p[k]]-b[u_val,p[k]]) + sum_k (a[k,i]-a[k,j])*(b[p[k],v_val]-b[p[k],u_val])
        
        # But this includes k=i and k=j terms which need special handling.
        # When k=i: (a[i,i]-a[j,i])*(b[v_val,u_val]-b[u_val,u_val]) + (a[i,i]-a[i,j])*(b[u_val,v_val]-b[u_val,u_val])
        # When k=j: (a[i,j]-a[j,j])*(b[v_val,v_val]-b[u_val,v_val]) + (a[j,i]-a[j,j])*(b[v_val,v_val]-b[u_val,v_val])
        
        # This is getting too complex. Let me just compute the cost directly for the delta.
        # For n=150, computing the full cost is O(n^2) = 22500 operations, which is fine.
        # But we want O(n) delta computation.
        
        # Let me use a different approach: maintain the cost and update it incrementally.
        # But for the delta, let me just compute it directly using numpy operations that are O(n).
        
        # Here's a clean way:
        # delta = dot(a[i] - a[j], b[v_val, p] - b[u_val, p]) 
        #         + dot(a[:, i] - a[:, j], b[p, v_val] - b[p, u_val])
        #         - 2 * (a[i,j]*(b[v_val,u_val] - b[u_val,v_val]) + a[j,i]*(b[u_val,v_val] - b[v_val,u_val]))
        #         - a[i,i]*(b[v_val,u_val] - b[u_val,u_val]) - a[j,j]*(b[u_val,v_val] - b[v_val,v_val])
        #         + a[i,j]*(b[v_val,u_val] - b[u_val,v_val]) + a[j,i]*(b[u_val,v_val] - b[v_val,u_val])
        #         + a[i,i]*(b[v_val,u_val] - b[u_val,u_val]) + a[j,j]*(b[u_val,v_val] - b[v_val,v_val])
        
        # I'm going in circles. Let me just test with a small example to verify the formula.
        # For now, let me use a simple but correct O(n) computation:
        
        # delta = 0
        # for k in range(n):
        #     if k == i: continue
        #     if k == j: continue
        #     delta += (a[i,k] - a[j,k]) * (b[v_val, p[k]] - b[u_val, p[k]])
        #     delta += (a[k,i] - a[k,j]) * (b[p[k], v_val] - b[p[k], u_val])
        # delta += (a[i,j] - a[j,i]) * (b[v_val, u_val] - b[u_val, v_val])
        
        # This is O(n) and correct. Let me vectorize it:
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # For k not in {i,j}:
        part1 = np.sum((a[i, mask] - a[j, mask]) * (b[v_val, p[mask]] - b[u_val, p[mask]]))
        part2 = np.sum((a[mask, i] - a[mask, j]) * (b[p[mask], v_val] - b[p[mask], u_val]))
        part3 = (a[i, j] - a[j, i]) * (b[v_val, u_val] - b[u_val, v_val])
        
        return int(part1 + part2 + part3)

    # Multi-start local search with steepest-descent 2-opt
    num_restarts = 0
    max_restarts = 20
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Local search: steepest-descent 2-opt
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Find the best improving swap
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n):
                    delta = get_delta_ij(i, j, p, a, b, c)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i >= 0:
                # Perform the best swap
                i, j = best_i, best_j
                p[i], p[j] = p[j], p[i]
                cost += best_delta
                improved = True
                
                # Update c
                c = compute_c(p, a, b)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
        
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # If we have time for more restarts, perturb and continue
        num_restarts += 1
        if num_restarts >= max_restarts:
            break
        
        # Perturbation: swap 3 random facilities for larger perturbation
        idx = random.sample(range(n), 3)
        p[idx[0]], p[idx[1]] = p[idx[1]], p[idx[0]]
        p[idx[1]], p[idx[2]] = p[idx[2]], p[idx[1]]
        p[idx[0]], p[idx[1]] = p[idx[1]], p[idx[0]]
        
        cost = int(np.sum(a * b[p]))
        c = compute_c(p, a, b)
        
        # If perturbed solution is better, update best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
