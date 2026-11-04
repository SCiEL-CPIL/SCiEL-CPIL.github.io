"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def solve(a, b):
    n = len(a)
    
    # 1. Initial assignment using rank-matching heuristic (heavy flows to short distances)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Precompute for efficiency
    # We want to minimize sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This is equivalent to minimizing sum_i C_i where C_i = sum_j a[i,j] * b[p[i], p[j]]
    # But since the cost is symmetric in a sense for the total, let's use a 2-opt style local search
    # with a more efficient evaluation.
    
    # Let's compute the current cost
    def compute_cost(p):
        # p is a permutation of 0..n-1
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This can be computed as:
        # Let B_p be the matrix b permuted by p in both indices? No.
        # b[p[i], p[j]] is the distance between location p[i] and p[j].
        # So we form a matrix D where D[i,j] = b[p[i], p[j]].
        # Then cost = sum_{i,j} a[i,j] * D[i,j] = sum(a * D)
        # We can compute D by indexing b with p[:, None] and p[None, :]
        D = b[p[:, None], p[None, :]]
        return np.sum(a * D)
    
    current_cost = compute_cost(p)
    
    # 2. Local search: 2-opt moves
    # Swap p[i] and p[j] and see if cost decreases.
    # Delta cost = sum_{k} a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #            + sum_{k} a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    #            - a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  (the cross term where k=j or k=i needs care)
    
    # Actually, let's derive the delta for swapping positions i and j (i.e., swapping facilities at these locations? No, p[i] is the location of facility i. Swapping p[i] and p[j] means facility i goes to location p[j] and facility j goes to location p[i]).
    # Let x = p[i], y = p[j].
    # New cost terms involving i or j:
    # Old: sum_k a[i,k] b[x, p[k]] + sum_k a[k,i] b[p[k], x] + sum_k a[j,k] b[y, p[k]] + sum_k a[k,j] b[p[k], y]
    #      - a[i,j] b[x,y] - a[j,i] b[y,x]  (since the k=j and k=i terms are included in the sums above, we need to be careful not to double count or miss the cross term)
    # Actually, the total cost is sum_{u,v} a[u,v] b[p[u], p[v]].
    # Terms involving i or j as first index:
    #   sum_v a[i,v] b[x, p[v]] and sum_v a[j,v] b[y, p[v]]
    # Terms involving i or j as second index:
    #   sum_u a[u,i] b[p[u], x] and sum_u a[u,j] b[p[u], y]
    # When we swap x and y (p[i]=y, p[j]=x):
    #   sum_v a[i,v] b[y, p'[v]] + sum_v a[j,v] b[x, p'[v]]
    #   sum_u a[u,i] b[p'[u], y] + sum_u a[u,j] b[p'[u], x]
    # where p' is the new permutation.
    # For v != i,j, p'[v] = p[v]. For v=i, p'[i]=y. For v=j, p'[j]=x.
    
    # Let's compute the delta more simply by just recalculating the affected terms.
    # Affected terms are those where u in {i,j} or v in {i,j}.
    # This is O(n) per swap.
    
    # Let's precompute row and column sums of b weighted by a? No, it depends on p.
    
    # Efficient delta calculation:
    # Let x = p[i], y = p[j].
    # Old contribution of i and j to the total cost:
    # C_old = sum_{k != i, j} (a[i,k] b[x, p[k]] + a[k,i] b[p[k], x] + a[j,k] b[y, p[k]] + a[k,j] b[p[k], y])
    #       + a[i,j] b[x,y] + a[j,i] b[y,x]
    #       + a[i,i] b[x,x] + a[j,j] b[y,y]  (self terms, b[x,x]=0 usually, but let's keep them)
    
    # Actually, a simpler way:
    # Delta = New_Cost - Old_Cost
    # New_Cost = Old_Cost - (terms involving x or y in the positions of i and j) + (new terms)
    
    # Let's define a function to compute the cost contribution of a facility i at location x given the current permutation p.
    # But this is still O(n).
    
    # Let's just do a simple local search with a few iterations, using vectorized operations where possible.
    # Given the time limit (120s) and n=100, O(n^3) or even O(n^4) might be too slow if not optimized.
    # O(n^2) swaps * O(n) delta = O(n^3) = 10^6 operations, which is fine.
    
    # Let's implement the O(n) delta calculation for swapping p[i] and p[j].
    # Let x = p[i], y = p[j].
    # The change in cost when swapping x and y at positions i and j:
    # Terms that change are those involving i or j as either row or column index in the cost sum.
    # Let S1 = sum_{k != i, j} a[i,k] * (b[y, p[k]] - b[x, p[k]])  # i moves from x to y
    # Let S2 = sum_{k != i, j} a[k,i] * (b[p[k], y] - b[p[k], x])  # i moves from x to y (col)
    # Let S3 = sum_{k != i, j} a[j,k] * (b[x, p[k]] - b[y, p[k]])  # j moves from y to x
    # Let S4 = sum_{k != i, j} a[k,j] * (b[p[k], x] - b[p[k], y])  # j moves from y to x (col)
    # Let S5 = a[i,j] * (b[y, x] - b[x, y])  # i,j interaction
    # Let S6 = a[j,i] * (b[x, y] - b[y, x])  # j,i interaction
    # Note: b is symmetric, so b[y,x] = b[x,y], so S5 and S6 cancel out? No, a[i,j] and a[j,i] might be different.
    # But b[y,x] - b[x,y] = 0 if b is symmetric. If b is not symmetric, it's not zero.
    # Also, we need to account for the self terms if i==j, but i!=j here.
    # What about the terms a[i,i] and a[j,j]? They don't change because p[i] and p[j] are swapped, but b[x,x] and b[y,y] are the same.
    # Wait, a[i,i] * b[p[i], p[i]] becomes a[i,i] * b[y, y] instead of a[i,i] * b[x, x].
    # So we need to add: a[i,i] * (b[y,y] - b[x,x]) + a[j,j] * (b[x,x] - b[y,y])
    
    # Let's refine:
    # Delta = S1 + S2 + S3 + S4 + S5 + S6 + a[i,i]*(b[y,y]-b[x,x]) + a[j,j]*(b[x,x]-b[y,y])
    
    # We can precompute for each location l, the vector of b[l, :] and b[:, l].
    # Then S1 = a[i, :][exclude i,j] dot (b[y, :][exclude i,j] - b[x, :][exclude i,j])
    # This is still O(n) per swap.
    
    # Let's try a simpler approach: just do a few rounds of random swaps and keep the best.
    # Or, use a deterministic local search: for each i, find the best j to swap with.
    
    # Let's implement a simple 2-opt local search with a time limit.
    import time
    start_time = time.time()
    time_limit = 110  # Leave some time for the rest of the program
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            x = p[i]
            best_j = -1
            best_delta = 0
            for j in range(i + 1, n):
                y = p[j]
                # Calculate delta
                # S1 = sum_{k != i, j} a[i,k] * (b[y, p[k]] - b[x, p[k]])
                # We can compute this as:
                # sum_k a[i,k] * (b[y, p[k]] - b[x, p[k]]) - a[i,i]*(b[y,x]-b[x,x]) - a[i,j]*(b[y,y]-b[x,y])
                # Let's compute the full sum and then subtract the i and j terms.
                # full_sum_1 = a[i, :] dot (b[y, p] - b[x, p])
                # But b[y, p] is b[y, p[k]] for all k.
                # Let's precompute b_rows = b[p, :] and b_cols = b[:, p]
                # Then b[x, p[k]] = b[x, p[k]] which is not directly b_rows or b_cols.
                # b_rows[k, l] = b[p[k], p[l]]
                # So b[x, p[k]] is not in b_rows unless x is one of the p's.
                # x = p[i], so b[x, p[k]] = b[p[i], p[k]] = b_rows[i, k]
                # Similarly, b[y, p[k]] = b[p[j], p[k]] = b_rows[j, k]
                # So S1_full = a[i, :] dot (b_rows[j, :] - b_rows[i, :])
                # Then subtract the k=i and k=j terms:
                # k=i: a[i,i] * (b_rows[j,i] - b_rows[i,i]) = a[i,i] * (b[y,x] - b[x,x])
                # k=j: a[i,j] * (b_rows[j,j] - b_rows[i,j]) = a[i,j] * (b[y,y] - b[x,y])
                # So S1 = S1_full - a[i,i]*(b[y,x]-b[x,x]) - a[i,j]*(b[y,y]-b[x,y])
                
                # Similarly for S2 = sum_{k != i, j} a[k,i] * (b[p[k], y] - b[p[k], x])
                # b[p[k], y] = b[p[k], p[j]] = b_rows[k, j]
                # b[p[k], x] = b[p[k], p[i]] = b_rows[k, i]
                # S2_full = a[:, i] dot (b_rows[:, j] - b_rows[:, i])
                # Subtract k=i and k=j:
                # k=i: a[i,i] * (b_rows[i,j] - b_rows[i,i]) = a[i,i] * (b[x,y] - b[x,x])
                # k=j: a[j,i] * (b_rows[j,j] - b_rows[j,i]) = a[j,i] * (b[y,y] - b[y,x])
                # So S2 = S2_full - a[i,i]*(b[x,y]-b[x,x]) - a[j,i]*(b[y,y]-b[y,x])
                
                # S3 = sum_{k != i, j} a[j,k] * (b[x, p[k]] - b[y, p[k]])
                # = - sum_{k != i, j} a[j,k] * (b[y, p[k]] - b[x, p[k]])
                # S3_full = a[j, :] dot (b_rows[i, :] - b_rows[j, :])
                # Subtract k=i and k=j:
                # k=i: a[j,i] * (b_rows[i,i] - b_rows[j,i]) = a[j,i] * (b[x,x] - b[y,x])
                # k=j: a[j,j] * (b_rows[i,j] - b_rows[j,j]) = a[j,j] * (b[x,y] - b[y,y])
                # So S3 = S3_full - a[j,i]*(b[x,x]-b[y,x]) - a[j,j]*(b[x,y]-b[y,y])
                
                # S4 = sum_{k != i, j} a[k,j] * (b[p[k], x] - b[p[k], y])
                # = - sum_{k != i, j} a[k,j] * (b[p[k], y] - b[p[k], x])
                # S4_full = a[:, j] dot (b_rows[:, i] - b_rows[:, j])
                # Subtract k=i and k=j:
                # k=i: a[i,j] * (b_rows[i,i] - b_rows[i,j]) = a[i,j] * (b[x,x] - b[x,y])
                # k=j: a[j,j] * (b_rows[j,i] - b_rows[j,j]) = a[j,j] * (b[y,x] - b[y,y])
                # So S4 = S4_full - a[i,j]*(b[x,x]-b[x,y]) - a[j,j]*(b[y,x]-b[y,y])
                
                # S5 = a[i,j] * (b[y,x] - b[x,y])
                # S6 = a[j,i] * (b[x,y] - b[y,x])
                
                # Self terms:
                # S_self = a[i,i]*(b[y,y]-b[x,x]) + a[j,j]*(b[x,x]-b[y,y])
                
                # Let's compute b_rows once outside the loop.
                # But b_rows depends on p, which changes.
                # So we need to recompute b_rows after each swap.
                # This is O(n^2) per swap, which is too slow.
                
                # Alternative: just compute the delta by directly accessing b.
                # S1 = sum_{k != i, j} a[i,k] * (b[y, p[k]] - b[x, p[k]])
                # We can vectorize this over k.
                # Let mask = np.ones(n, dtype=bool); mask[i]=False; mask[j]=False
                # S1 = np.sum(a[i, mask] * (b[y, p[mask]] - b[x, p[mask]]))
                # This is O(n) and vectorized.
                
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                pk = p[mask]
                S1 = np.sum(a[i, mask] * (b[y, pk] - b[x, pk]))
                S2 = np.sum(a[mask, i] * (b[pk, y] - b[pk, x]))
                S3 = np.sum(a[j, mask] * (b[x, pk] - b[y, pk]))
                S4 = np.sum(a[mask, j] * (b[pk, x] - b[pk, y]))
                S5 = a[i, j] * (b[y, x] - b[x, y])
                S6 = a[j, i] * (b[x, y] - b[y, x])
                S_self = a[i, i] * (b[y, y] - b[x, x]) + a[j, j] * (b[x, x] - b[y, y])
                
                delta = S1 + S2 + S3 + S4 + S5 + S6 + S_self
                
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j != -1 and best_delta < 0:
                # Swap p[i] and p[best_j]
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_delta
                improved = True
    
    return p.tolist()
# EVOLVE-BLOCK-END
