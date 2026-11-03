"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Helper to compute cost
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    # 1. Initial Heuristic: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute for efficient delta calculation
    # We want to compute the cost change for swapping two facilities i and j.
    # Let's define a matrix C where C[i, loc] is the contribution of facility i to location loc?
    # No, the cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # This is a quadratic form.
    # Let's maintain an array `row_contrib` where row_contrib[i] = sum_j a[i,j] * b[p[i], p[j]]?
    # No, that's not quite right because b[p[i], p[j]] depends on both i and j.
    
    # A more efficient way to evaluate swaps:
    # Let's define `cost_i` as the cost contribution involving facility i.
    # cost_i = sum_{j != i} a[i,j] b[p[i], p[j]] + a[i,i] b[p[i], p[i]]
    # If we swap i and j, the change in cost is:
    # delta = (new cost_i + new cost_j) - (old cost_i + old cost_j) + cross terms?
    # Actually, the total cost is sum_i cost_i / 2? No, sum_{i,j} counts each pair twice if a is symmetric, but a is not necessarily symmetric.
    # The cost is exactly sum_{i,j} a[i,j] b[p[i], p[j]].
    
    # Let's use the standard QAP swap delta formula.
    # For swapping facilities i and j (at locations l_i, l_j):
    # delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[l_i, p[k]] - b[l_j, p[k]]) 
    #         + a[i,j] * (b[l_i, l_j] - b[l_j, l_i]) ... wait, this is getting complicated.
    
    # Let's stick to the O(n) vectorized delta calculation which is fast enough for n=100.
    # delta = sum_{k} (a[i,k] - a[j,k]) * (b[l_j, p[k]] - b[l_i, p[k]]) 
    #         + sum_{k} (a[k,i] - a[k,j]) * (b[p[k], l_j] - b[p[k], l_i])
    # This formula was derived in the previous step and seems correct if we handle the i,j terms carefully.
    # Actually, the previous derivation showed:
    # delta = np.sum((a[i, :] - a[j, :]) * diff1) + np.sum((a[:, i] - a[:, j]) * diff2)
    # where diff1 = b[l_j, p] - b[l_i, p] and diff2 = b[p, l_j] - b[p, l_i]
    # This formula includes the i and j terms in the sums, which might be double-counted or incorrect.
    # Let's re-verify for n=2 or small n.
    # For n=2, i=0, j=1.
    # diff1 = [b[1,0]-b[0,0], b[1,1]-b[0,1]]
    # diff2 = [b[0,1]-b[0,0], b[1,1]-b[1,0]]
    # term1 = (a[0,0]-a[1,0])*diff1[0] + (a[0,1]-a[1,1])*diff1[1]
    # term2 = (a[0,0]-a[1,0])*diff2[0] + (a[0,1]-a[1,1])*diff2[1]
    # This seems messy.
    
    # Let's use a simpler, robust O(n^2) calculation for delta if O(n) is tricky,
    # but for n=100, O(n^2) is 10,000 ops. 100k swaps/sec -> 10^9 ops/sec.
    # NumPy can do 10^9 ops/sec easily.
    # So we can just compute the full cost difference by summing the affected terms.
    # Affected terms are those where i or j is involved.
    # There are 2*n - 2 such terms (excluding i,i and j,j? No, including them).
    # Actually, all terms a[u,v] where u==i or v==i or u==j or v==j.
    # That is at most 4n - 4 terms.
    # We can compute the old and new cost for these terms in O(n).
    
    def swap_delta(i, j, p):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        
        # Old cost contribution for pairs involving i or j
        # We need sum_{u,v} a[u,v] b[p[u], p[v]] where u in {i,j} or v in {i,j}
        # This set of indices is:
        # (i, all), (all, i), (j, all), (all, j)
        # Union: (i, all) U (all, i) U (j, all) U (all, j)
        
        # Let's just sum the 4 vectors
        # old_i = a[i, :] @ b[li, p]  # sum_k a[i,k] b[li, p[k]]
        # old_ii = a[:, i] @ b[p, li] # sum_k a[k,i] b[p[k], li]
        # old_j = a[j, :] @ b[lj, p]
        # old_jj = a[:, j] @ b[p, lj]
        
        # But this double counts (i,i), (i,j), (j,i), (j,j)
        # Specifically:
        # (i,i) is in old_i (k=i) and old_ii (k=i)
        # (i,j) is in old_i (k=j)
        # (j,i) is in old_ii (k=j)
        # (j,j) is in old_j (k=j) and old_jj (k=j)
        # (j,i) is in old_j (k=i)
        # (i,j) is in old_jj (k=i)
        
        # This is error-prone. Let's just compute the total cost change by
        # summing over all k != i, j the change in terms involving k,
        # plus the change in the 4 terms involving i and j directly.
        
        # Terms involving k (k != i, j):
        # Old: a[i,k]b[li,pk] + a[k,i]b[pk,li] + a[j,k]b[lj,pk] + a[k,j]b[pk,lj]
        # New: a[i,k]b[lj,pk] + a[k,i]b[pk,lj] + a[j,k]b[li,pk] + a[k,j]b[pk,li]
        # Delta_k = (a[i,k]-a[j,k])(b[lj,pk]-b[li,pk]) + (a[k,i]-a[k,j])(b[pk,lj]-b[pk,li])
        
        # We can compute this vectorized for all k, then subtract k=i and k=j.
        # diff1 = b[lj, p] - b[li, p]
        # diff2 = b[p, lj] - b[p, li]
        # delta_all = sum_k (a[i,k]-a[j,k])diff1[k] + (a[k,i]-a[k,j])diff2[k]
        
        # Then subtract the k=i and k=j parts from delta_all.
        # And add the correct delta for the i,j interaction terms.
        
        diff1 = b[lj, p] - b[li, p]
        diff2 = b[p, lj] - b[p, li]
        
        # Vectorized sum over all k
        delta_sum = np.sum((a[i, :] - a[j, :]) * diff1) + np.sum((a[:, i] - a[:, j]) * diff2)
        
        # Now correct for k=i and k=j
        # At k=i:
        # Term in delta_sum: (a[i,i]-a[j,i])diff1[i] + (a[i,i]-a[i,j])diff2[i]
        # diff1[i] = b[lj, li] - b[li, li]
        # diff2[i] = b[li, lj] - b[li, li]
        # True delta for terms involving i and i (self) and i and j:
        # This is getting too complicated.
        
        # Let's just do the O(n) loop for the 4 terms directly and the vectorized sum for the rest.
        # Actually, for n=100, a Python loop of 100 is fine.
        # But vectorized is faster.
        
        # Let's try a different approach:
        # Compute the cost of the current assignment for the subset of facilities {i, j} and their interactions with others.
        
        # Let's just use the full O(n^2) cost calculation for the delta?
        # No, that's too slow.
        
        # Let's use the formula:
        # delta = 2 * (sum_{k != i, j} a[i,k] (b[lj, p[k]] - b[li, p[k]]) + a[k,i] (b[p[k], lj] - b[p[k], li]))
        #        + a[i,j] (b[lj, li] - b[li, lj]) + a[j,i] (b[li, lj] - b[lj, li])
        #        + a[i,i] (b[lj, lj] - b[li, li]) + a[j,j] (b[li, li] - b[lj, lj])
        # Wait, the factor of 2 is wrong.
        
        # Let's go back to basics.
        # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Let S be the set of indices {i, j}.
        # Terms are partitioned into:
        # 1. u in S, v in S
        # 2. u in S, v not in S
        # 3. u not in S, v in S
        # 4. u not in S, v not in S (unchanged)
        
        # Delta = (New_1 + New_2 + New_3) - (Old_1 + Old_2 + Old_3)
        
        # New_1 = a[i,i]b[lj,lj] + a[i,j]b[lj,li] + a[j,i]b[li,lj] + a[j,j]b[li,li]
        # Old_1 = a[i,i]b[li,li] + a[i,j]b[li,lj] + a[j,i]b[lj,li] + a[j,j]b[lj,lj]
        # Delta_1 = a[i,i](b[lj,lj]-b[li,li]) + a[i,j](b[lj,li]-b[li,lj]) + a[j,i](b[li,lj]-b[lj,li]) + a[j,j](b[li,li]-b[lj,lj])
        
        # New_2 = sum_{k not in S} a[i,k]b[lj,pk] + a[j,k]b[li,pk]
        # Old_2 = sum_{k not in S} a[i,k]b[li,pk] + a[j,k]b[lj,pk]
        # Delta_2 = sum_{k not in S} (a[i,k]-a[j,k])(b[lj,pk]-b[li,pk])
        
        # New_3 = sum_{k not in S} a[k,i]b[pk,lj] + a[k,j]b[pk,li]
        # Old_3 = sum_{k not in S} a[k,i]b[pk,li] + a[k,j]b[pk,lj]
        # Delta_3 = sum_{k not in S} (a[k,i]-a[k,j])(b[pk,lj]-b[pk,li])
        
        # So Delta = Delta_1 + Delta_2 + Delta_3
        
        # We can compute Delta_2 and Delta_3 using vectorized sums over all k,
        # then subtract the terms where k=i or k=j.
        
        # Let V1[k] = (a[i,k]-a[j,k])(b[lj,pk]-b[li,pk])
        # Let V2[k] = (a[k,i]-a[k,j])(b[pk,lj]-b[pk,li])
        
        # Delta_2 = sum_{k not in S} V1[k] = sum_all V1[k] - V1[i] - V1[j]
        # Delta_3 = sum_{k not in S} V2[k] = sum_all V2[k] - V2[i] - V2[j]
        
        diff1 = b[lj, p] - b[li, p]
        diff2 = b[p, lj] - b[p, li]
        
        V1 = (a[i, :] - a[j, :]) * diff1
        V2 = (a[:, i] - a[:, j]) * diff2
        
        sum_V1 = np.sum(V1)
        sum_V2 = np.sum(V2)
        
        # Subtract V1[i], V1[j], V2[i], V2[j]
        # V1[i] = (a[i,i]-a[j,i]) * (b[lj,li]-b[li,li])
        # V1[j] = (a[i,j]-a[j,j]) * (b[lj,lj]-b[li,lj])
        # V2[i] = (a[i,i]-a[i,j]) * (b[li,lj]-b[li,li])
        # V2[j] = (a[j,i]-a[j,j]) * (b[lj,lj]-b[lj,li])
        
        delta_2 = sum_V1 - V1[i] - V1[j]
        delta_3 = sum_V2 - V2[i] - V2[j]
        
        delta_1 = (a[i,i]*(b[lj,lj]-b[li,li]) + 
                   a[i,j]*(b[lj,li]-b[li,lj]) + 
                   a[j,i]*(b[li,lj]-b[lj,li]) + 
                   a[j,j]*(b[li,li]-b[lj,lj]))
        
        return int(delta_1 + delta_2 + delta_3)

    # 2. Simulated Annealing
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    T = 10000.0
    T_min = 1.0
    alpha = 0.999
    
    # Use a better initial solution?
    # Let's try a few random restarts if time permits.
    
    while True:
        if time.time() - start_time > timeout:
            break
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = swap_delta(i, j, p)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
