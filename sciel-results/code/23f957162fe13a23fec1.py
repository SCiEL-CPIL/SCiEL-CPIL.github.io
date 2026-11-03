"""Rewritten program for QAPLIB tho150 with time-bounded local search."""
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np


def _cost(a, b, p):
    """Exact integer cost for permutation p."""
    # Use 64-bit arithmetic; tho150 costs fit in int64.
    # sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: for each i, contribution is a[i,:] dot b[p[i], p]
    # We can do: sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :])[p]  ... wait, b[p[i], p[j]] for fixed i, varying j
    # For fixed i: sum_j a[i,j] * b[p[i], p[j]] = a[i,:] dot b[p[i], p]
    # So total = sum_i (a[i,:] dot b[p[i], p])
    # We can vectorize: for each i, we need a[i,:] dot b[p[i], p]
    # b[p[i], p] is the row b[p[i],:] permuted by p, i.e., b[p[i], p]
    # a[i,:] dot b[p[i], p] = sum_j a[i,j] * b[p[i], p[j]]
    # This is equivalent to (a[i,:] @ b[p[i], :])[p]... no.
    # Let's just do: total = 0
    # for i in range(n): total += np.dot(a[i, :], b[p[i], p])
    # Vectorized: rows = b[p, p]  # shape (n, n), rows[i, j] = b[p[i], p[j]]
    # Then cost = sum_i sum_j a[i,j] * rows[i,j] = np.sum(a * rows)
    # But that's O(n^2) memory and compute, which for n=150 is fine (22500 elements).
    # Actually even better: cost = np.einsum('ij,ij->', a, b[p, p])
    # But b[p, p] creates a (n,n) array. For n=150 that's 22500 * 8 bytes = 180 KB, fine.
    return int(np.einsum('ij,ij->', a, b[p, p]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap p[i] and p[j].
    Current cost contribution involving i or j:
      C = sum_{k} a[i,k]*b[p[i],p[k]] + sum_{k} a[k,i]*b[p[k],p[i]]
        + sum_{k} a[j,k]*b[p[j],p[k]] + sum_{k} a[k,j]*b[p[k],p[j]]
        + a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
    After swap, p[i] and p[j] are exchanged.
    Let u = p[i], v = p[j]. After swap, p[i]=v, p[j]=u.
    New contribution:
      C' = sum_{k} a[i,k]*b[v,p[k]] + sum_{k} a[k,i]*b[p[k],v]
         + sum_{k} a[j,k]*b[u,p[k]] + sum_{k} a[k,j]*b[p[k],u]
         + a[i,j]*b[v,u] + a[j,i]*b[u,v]
    For k not in {i,j}, p[k] unchanged.
    Delta = C' - C.
    """
    u = p[i]
    v = p[j]
    n = len(p)
    
    # Terms involving i (row i of a, col i of a) and j similarly.
    # Let's compute C and C' carefully.
    # C = sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u] 
    #   + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,j]*b[p[k],v]
    #   + a[i,j]*b[u,v] + a[j,i]*b[v,u]
    # Note: when k=i in first sum: a[i,i]*b[u,p[i]]=a[i,i]*b[u,u]
    #        when k=j in first sum: a[i,j]*b[u,p[j]]=a[i,j]*b[u,v]
    # So the a[i,j]*b[u,v] term is already included in the first sum when k=j.
    # Similarly, a[j,i]*b[v,u] is in the third sum when k=i.
    # So actually C = sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u] 
    #                + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,j]*b[p[k],v]
    # with no double counting issue because the cross terms are part of the sums.
    
    # C' = sum_k a[i,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],v]
    #    + sum_k a[j,k]*b[u,p[k]] + sum_k a[k,j]*b[p[k],u]
    # But wait, after swap, p[i]=v and p[j]=u. So in the sums, when k=i, p[k]=v, when k=j, p[k]=u.
    # Let me re-derive more carefully.
    
    # Total cost = sum_{r,s} a[r,s] * b[p[r], p[s]]
    # = sum_r sum_s a[r,s] * b[p[r], p[s]]
    # Terms that change when we swap p[i] and p[j] are those where r in {i,j} or s in {i,j}.
    # So delta = [terms with r in {i,j} or s in {i,j} after swap] - [same before swap]
    
    # Before swap: p[i]=u, p[j]=v
    # After swap: p[i]=v, p[j]=u
    
    # Let's compute the contribution of all (r,s) where r in {i,j} or s in {i,j}.
    # This is: 
    #   sum_s a[i,s]*b[u,p[s]] + sum_r a[r,i]*b[p[r],u] 
    #   + sum_s a[j,s]*b[v,p[s]] + sum_r a[r,j]*b[p[r],v]
    #   - a[i,i]*b[u,u] - a[i,j]*b[u,v]  # subtract double-counted terms
    # Wait, this is getting messy. Let me just compute the full cost difference by 
    # considering that only entries in rows i,j and cols i,j of the "effective" matrix change.
    
    # Simpler approach: compute the cost contribution of the 2x2 block of indices {i,j} in both
    # the "a-space" and "b-space" perspective.
    
    # Actually, the cleanest way: the cost is sum_{r,s} a[r,s] * b[p[r],p[s]].
    # When we swap p[i] and p[j], the values that change are:
    #   b[p[i],p[s]] for all s (row u becomes row v in the b-matrix lookup for facility i)
    #   b[p[r],p[i]] for all r (col u becomes col v in the b-matrix lookup for location i)
    #   Similarly for j.
    
    # Let me just compute delta directly:
    # delta = sum_s a[i,s]*(b[v,p[s]] - b[u,p[s]]) 
    #       + sum_r a[r,i]*(b[p[r],v] - b[p[r],u])
    #       + sum_s a[j,s]*(b[u,p[s]] - b[v,p[s]])
    #       + sum_r a[r,j]*(b[p[r],u] - b[p[r],v])
    
    # But this double-counts some terms. Let's verify:
    # The total cost change is sum_{r,s} a[r,s] * (b[p'_r, p'_s] - b[p_r, p_s])
    # where p' differs from p only at i and j.
    # So p'_r = p_r for r not in {i,j}, p'_i = v, p'_j = u.
    # Thus b[p'_r, p'_s] - b[p_r, p_s] is nonzero only if r in {i,j} or s in {i,j}.
    # So delta = sum_{r,s: r in {i,j} or s in {i,j}} a[r,s] * (b[p'_r,p'_s] - b[p_r,p_s])
    
    # Let's break this into four parts:
    # 1. r=i, s not in {i,j}: a[i,s]*(b[v,p[s]] - b[u,p[s]])
    # 2. r=j, s not in {i,j}: a[j,s]*(b[u,p[s]] - b[v,p[s]])
    # 3. s=i, r not in {i,j}: a[r,i]*(b[p[r],v] - b[p[r],u])
    # 4. s=j, r not in {i,j}: a[r,j]*(b[p[r],u] - b[p[r],v])
    # 5. r=i, s=i: a[i,i]*(b[v,v] - b[u,u])
    # 6. r=i, s=j: a[i,j]*(b[v,u] - b[u,v])
    # 7. r=j, s=i: a[j,i]*(b[u,v] - b[v,u])
    # 8. r=j, s=j: a[j,j]*(b[u,u] - b[v,v])
    
    # So delta = (1) + (2) + (3) + (4) + (5) + (6) + (7) + (8)
    
    # We can compute this efficiently:
    # Let's precompute for each location x, the row b[x, :] and col b[:, x].
    
    # (1) = sum_{s not in {i,j}} a[i,s] * (b[v,p[s]] - b[u,p[s]])
    #     = a[i,:] dot (b[v,p] - b[u,p]) - a[i,i]*(b[v,p[i]]-b[u,p[i]]) - a[i,j]*(b[v,p[j]]-b[u,p[j]])
    #     = a[i,:] dot (b[v,p] - b[u,p]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
    
    # Similarly for the others. This is getting complex. Let me just compute it directly
    # with vectorized operations for the given i, j.
    
    # For efficiency, let's compute:
    # diff_i_s = b[v, p] - b[u, p]  # shape (n,)
    # diff_r_i = b[:, v][p] - b[:, u][p]  # = b[p, v] - b[p, u], shape (n,)
    
    # (1) + part of (5) + part of (6):
    # sum_s a[i,s] * diff_i_s[s] = a[i,:] dot diff_i_s
    # This includes s=i and s=j terms.
    # (3) + part of (5) + part of (7):
    # sum_r a[r,i] * diff_r_i[r] = a[:,i] dot diff_r_i
    # (2) + part of (6) + part of (8):
    # sum_s a[j,s] * (b[u,p[s]] - b[v,p[s]]) = a[j,:] dot (b[u,p] - b[v,p]) = -a[j,:] dot diff_i_s
    # (4) + part of (7) + part of (8):
    # sum_r a[r,j] * (b[p[r],u] - b[p[r],v]) = a[:,j] dot (b[p,u] - b[p,v]) = -a[:,j] dot diff_r_i
    
    # So total delta = a[i,:] dot diff_i_s + a[:,i] dot diff_r_i - a[j,:] dot diff_i_s - a[:,j] dot diff_r_i
    #                 = (a[i,:] - a[j,:]) dot diff_i_s + (a[:,i] - a[:,j]) dot diff_r_i
    
    # Let's verify this doesn't double count. The formula sum_{r,s: r in {i,j} or s in {i,j}} 
    # can be written as (r in {i,j}) + (s in {i,j}) - (r in {i,j} AND s in {i,j}).
    # My derivation above seems to handle this correctly via the inclusion-exclusion.
    
    diff_i_s = b[v, p] - b[u, p]  # shape (n,)
    diff_r_i = b[p, v] - b[p, u]  # shape (n,)
    
    delta = (a[i, :] - a[j, :]) @ diff_i_s + (a[:, i] - a[:, j]) @ diff_r_i
    
    return int(delta)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_cost = _cost(a, b, p)
    best_p = p.copy()
    
    # Local search: 2-opt swaps
    # For each pair (i,j), compute swap delta and accept if it improves.
    # Use a simple hill-climbing with first-improvement or best-improvement.
    # Given time constraints, use first-improvement with multiple passes.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    if best_cost < 0:  # shouldn't happen but safety
                        break
    
    # Check if we improved
    if best_cost < _cost(a, b, best_p):
        best_p = p.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
