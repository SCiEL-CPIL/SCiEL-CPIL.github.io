"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(a)
    # Efficient cost calculation: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can vectorize this by computing b_p = b[p][:, p] and then sum(a * b_p)
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)


def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap positions i and j in permutation p.
    p is the current permutation (facility at location k is p[k]... wait, let's clarify).
    
    Convention: p[i] is the location assigned to facility i.
    So facility i is at p[i], facility j is at p[j].
    If we swap, facility i goes to p[j] and facility j goes to p[i].
    
    The cost is C = sum_{u,v} a[u,v] * b[p[u], p[v]].
    
    Let's derive the delta for swapping facilities i and j.
    Let li = p[i], lj = p[j].
    Before swap:
      - Terms involving i and j:
        a[i,i]*b[li,li] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj]
        + sum_{k != i,j} (a[i,k]*b[li, p[k]] + a[k,i]*b[p[k], li] + a[j,k]*b[lj, p[k]] + a[k,j]*b[p[k], lj])
    
    After swap (p'[i]=lj, p'[j]=li, p'[k]=p[k] for k!=i,j):
      - Terms involving i and j:
        a[i,i]*b[lj,lj] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
        + sum_{k != i,j} (a[i,k]*b[lj, p[k]] + a[k,i]*b[p[k], lj] + a[j,k]*b[li, p[k]] + a[k,j]*b[p[k], li])
    
    Delta = After - Before.
    The terms a[i,i]*b[li,li] and a[j,j]*b[lj,lj] swap with a[i,i]*b[lj,lj] and a[j,j]*b[li,li].
    Actually, it's easier to just compute the full delta by considering all pairs (u,v) where u or v is i or j.
    
    Let's compute the contribution of all pairs (u,v) where u in {i,j} or v in {i,j}.
    Before:
      S_before = sum_{v} a[i,v]*b[li, p[v]] + sum_{u} a[u,j]*b[p[u], lj] - a[i,j]*b[li,lj]  (subtract double counted term? No, let's be careful)
    
    Actually, the total cost is sum_{u,v} a[u,v] b[p[u], p[v]].
    Let's define a function f(u, v) = a[u,v] * b[p[u], p[v]].
    We want to compute sum_{u,v} f'(u,v) - sum_{u,v} f(u,v) where p' differs from p only at i and j.
    
    The only terms that change are those where u is in {i,j} or v is in {i,j}.
    Specifically:
    1. u=i, v=i: a[i,i] * (b[lj,lj] - b[li,li])
    2. u=i, v=j: a[i,j] * (b[lj,li] - b[li,lj])
    3. u=j, v=i: a[j,i] * (b[li,lj] - b[lj,li])
    4. u=j, v=j: a[j,j] * (b[li,li] - b[lj,lj])
    5. u=i, v=k (k!=i,j): a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    6. u=k (k!=i,j), v=i: a[k,i] * (b[p[k], lj] - b[p[k], li])
    7. u=j, v=k (k!=i,j): a[j,k] * (b[li, p[k]] - b[lj, p[k]])
    8. u=k (k!=i,j), v=j: a[k,j] * (b[p[k], li] - b[p[k], lj])
    
    This can be computed efficiently with vectorized operations.
    """
    li = p[i]
    lj = p[j]
    
    # Diagonal terms
    delta = a[i,i] * (b[lj,lj] - b[li,li]) + a[i,j] * (b[lj,li] - b[li,lj]) + \
            a[j,i] * (b[li,lj] - b[lj,li]) + a[j,j] * (b[li,li] - b[lj,lj])
    
    # Off-diagonal terms with i
    # sum_{k != i,j} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    # = sum_{k} a[i,k] * (b[lj, p[k]] - b[li, p[k]]) - a[i,i]*(b[lj,li]-b[li,li]) - a[i,j]*(b[lj,lj]-b[li,lj])
    # Wait, the k=i and k=j terms are already accounted for in the diagonal terms above?
    # Let's just sum over all k and subtract the i and j parts if we double count.
    # Actually, it's safer to sum over all k != i, j.
    
    # Vectorized computation for terms involving i and j with other facilities
    # Let's compute the full sums and then subtract the i and j contributions to avoid double counting.
    
    # Sum over all k: a[i,k] * b[lj, p[k]]
    sum_i_lj = np.dot(a[i, :], b[lj, p])
    # Sum over all k: a[i,k] * b[li, p[k]]
    sum_i_li = np.dot(a[i, :], b[li, p])
    
    # Sum over all k: a[k,i] * b[p[k], lj]
    sum_k_i_lj = np.dot(a[:, i], b[p, lj])
    # Sum over all k: a[k,i] * b[p[k], li]
    sum_k_i_li = np.dot(a[:, i], b[p, li])
    
    # Sum over all k: a[j,k] * b[li, p[k]]
    sum_j_li = np.dot(a[j, :], b[li, p])
    # Sum over all k: a[j,k] * b[lj, p[k]]
    sum_j_lj = np.dot(a[j, :], b[lj, p])
    
    # Sum over all k: a[k,j] * b[p[k], li]
    sum_k_j_li = np.dot(a[:, j], b[p, li])
    # Sum over all k: a[k,j] * b[p[k], lj]
    sum_k_j_lj = np.dot(a[:, j], b[p, lj])
    
    # Now, the delta for the off-diagonal parts (u=i or v=i or u=j or v=j, excluding u=v=i, u=v=j, u=i,v=j, u=j,v=i)
    # The terms are:
    # (sum_i_lj - sum_i_li) for u=i, v=k (all k)
    # (sum_k_i_lj - sum_k_i_li) for u=k, v=i (all k)
    # (sum_j_li - sum_j_lj) for u=j, v=k (all k)
    # (sum_k_j_li - sum_k_j_lj) for u=k, v=j (all k)
    
    # But we have double counted:
    # - For u=i, v=k: when k=i, we have a[i,i]*(b[lj,li]-b[li,li]) which is part of diagonal term 1.
    #   When k=j, we have a[i,j]*(b[lj,lj]-b[li,lj]) which is part of diagonal term 2.
    # - For u=k, v=i: when k=i, we have a[i,i]*(b[li,lj]-b[li,li]) which is part of diagonal term 1 (transposed).
    #   When k=j, we have a[j,i]*(b[lj,lj]-b[lj,li]) which is part of diagonal term 3.
    # - Similarly for j.
    
    # So, let's subtract the diagonal contributions from these sums.
    # Diagonal contributions to sum_i_lj - sum_i_li:
    #   k=i: a[i,i]*(b[lj,li]-b[li,li])
    #   k=j: a[i,j]*(b[lj,lj]-b[li,lj])
    # Diagonal contributions to sum_k_i_lj - sum_k_i_li:
    #   k=i: a[i,i]*(b[li,lj]-b[li,li])
    #   k=j: a[j,i]*(b[lj,lj]-b[lj,li])
    # Diagonal contributions to sum_j_li - sum_j_lj:
    #   k=i: a[j,i]*(b[li,li]-b[lj,li])
    #   k=j: a[j,j]*(b[li,lj]-b[lj,lj])
    # Diagonal contributions to sum_k_j_li - sum_k_j_lj:
    #   k=i: a[i,j]*(b[li,li]-b[li,lj])
    #   k=j: a[j,j]*(b[lj,li]-b[lj,lj])
    
    # This is getting complicated. Let's just compute the delta by direct evaluation for small n or use a simpler method.
    # For n=256, a full cost evaluation is O(n^2) = 65536 ops, which is fast.
    # We can just evaluate the cost before and after the swap.
    # But we need to do many swaps, so we want it faster.
    
    # Alternative: Just return the difference of the two full costs. It's O(n^2) per swap, which might be too slow for many iterations.
    # Let's stick with the analytical formula but be very careful.
    
    # Let's re-derive:
    # Delta = sum_{u,v} a[u,v] * (b[p'[u], p'[v]] - b[p[u], p[v]])
    # Non-zero only if u in {i,j} or v in {i,j}.
    
    # Let's define:
    # D1 = sum_{v} a[i,v] * (b[lj, p'[v]] - b[li, p[v]])
    #    = a[i,i]*(b[lj,lj]-b[li,li]) + a[i,j]*(b[lj,li]-b[li,lj]) + sum_{v!=i,j} a[i,v]*(b[lj,p[v]]-b[li,p[v]])
    # D2 = sum_{u} a[u,j] * (b[p'[u], li] - b[p[u], lj])
    #    = a[i,j]*(b[lj,li]-b[li,lj]) + a[j,j]*(b[li,li]-b[lj,lj]) + sum_{u!=i,j} a[u,j]*(b[p[u],li]-b[p[u],lj])
    # D3 = sum_{v} a[j,v] * (b[li, p'[v]] - b[lj, p[v]])
    #    = a[j,i]*(b[li,lj]-b[lj,li]) + a[j,j]*(b[li,li]-b[lj,lj]) + sum_{v!=i,j} a[j,v]*(b[li,p[v]]-b[lj,p[v]])
    # D4 = sum_{u} a[u,i] * (b[p'[u], lj] - b[p[u], li])
    #    = a[i,i]*(b[lj,lj]-b[li,li]) + a[j,i]*(b[li,lj]-b[lj,li]) + sum_{u!=i,j} a[u,i]*(b[p[u],lj]-b[p[u],li])
    
    # Wait, the total delta is D1 + D2 + D3 + D4 - (double counted terms).
    # The terms a[i,j] and a[j,i] appear in multiple D's.
    # Specifically:
    # - a[i,j]*(b[lj,li]-b[li,lj]) appears in D1 (v=j) and D2 (u=i).
    # - a[j,i]*(b[li,lj]-b[lj,li]) appears in D3 (v=i) and D4 (u=j).
    # - a[i,i]*(b[lj,lj]-b[li,li]) appears in D1 (v=i) and D4 (u=i).
    # - a[j,j]*(b[li,li]-b[lj,lj]) appears in D2 (u=j) and D3 (v=j).
    
    # So if we sum D1+D2+D3+D4, we double count these four terms.
    # But actually, the cost function is sum_{u,v} a[u,v]b[p[u],p[v]].
    # The change is sum_{u,v} a[u,v](b[p'[u],p'[v]]-b[p[u],p[v]]).
    # This is exactly D1+D2+D3+D4 if we define D's as sums over all u,v where the first index is fixed or the second is fixed, but that's not right.
    
    # Let's just use the direct formula:
    # Delta = a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj]) + \
    #         a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li]) + \
    #         sum_{k!=i,j} a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + \
    #         sum_{k!=i,j} a[k,i]*(b[p[k],lj]-b[p[k],li]) + \
    #         sum_{k!=i,j} a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + \
    #         sum_{k!=i,j} a[k,j]*(b[p[k],li]-b[p[k],lj])
    
    # We can compute the sums over k!=i,j by computing the sum over all k and subtracting k=i and k=j.
    
    # Sum_{k} a[i,k]*(b[lj,p[k]]-b[li,p[k]]) = np.dot(a[i,:], b[lj,p] - b[li,p])
    # Subtract k=i: a[i,i]*(b[lj,li]-b[li,li])
    # Subtract k=j: a[i,j]*(b[lj,lj]-b[li,lj])
    
    s1 = np.dot(a[i, :], b[lj, p] - b[li, p]) - a[i, i] * (b[lj, li] - b[li, li]) - a[i, j] * (b[lj, lj] - b[li, lj])
    s2 = np.dot(a[:, i], b[p, lj] - b[p, li]) - a[i, i] * (b[li, lj] - b[li, li]) - a[j, i] * (b[lj, lj] - b[lj, li])
    s3 = np.dot(a[j, :], b[li, p] - b[lj, p]) - a[j, i] * (b[li, li] - b[lj, li]) - a[j, j] * (b[li, lj] - b[lj, lj])
    s4 = np.dot(a[:, j], b[p, li] - b[p, lj]) - a[i, j] * (b[li, li] - b[li, lj]) - a[j, j] * (b[lj, li] - b[lj, lj])
    
    delta_diag = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj]) + \
                 a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
    
    return delta_diag + s1 + s2 + s3 + s4


def solve(a, b):
    n = len(a)
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    
    start_time = time.time()
    max_time = 115.0  # Leave some buffer
    
    # Hill climbing with 2-swap moves
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Random order of pairs to avoid bias
        # For efficiency, we can sample a limited number of swaps per iteration
        # Or iterate through all pairs if n is small enough. n=256, so n^2/2 = 32640 pairs.
        # Each swap cost evaluation is O(n) = 256 ops. Total per full pass: 32640 * 256 ~ 8.3M ops.
        # In Python/NumPy, this might take ~1-2 seconds per full pass.
        # So we can do ~50-100 passes in 115 seconds.
        
        # To be safe, let's do random sampling of swaps if full passes are too slow.
        # Let's try a full pass first.
        
        # Generate all pairs
        i_list, j_list = np.triu_indices(n, k=1)
        # Shuffle the order
        perm = np.random.permutation(len(i_list))
        i_list = i_list[perm]
        j_list = j_list[perm]
        
        for idx in range(len(i_list)):
            if (time.time() - start_time) > max_time:
                break
            i = i_list[idx]
            j = j_list[idx]
            
            delta = _swap_cost_delta(a, b, p, i, j)
            if delta < 0:
                # Perform swap
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                improved = True
    
    return p.tolist()
# EVOLVE-BLOCK-END
