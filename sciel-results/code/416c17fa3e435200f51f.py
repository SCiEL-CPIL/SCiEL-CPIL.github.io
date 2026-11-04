"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for a given permutation p."""
    n = len(p)
    # p is a permutation: facility i goes to location p[i]
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this as trace(a @ b_p) where b_p[i,j] = b[p[i], p[j]]
    # But that's O(n^3). Let's use a more efficient approach.
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :]) @ ones... no
    # Actually: for each i, sum_j a[i,j] * b[p[i], p[j]] = (a[i,:] * b[p[i], p[:]])... no
    # Let's think: b[p[i], p[j]] for fixed i, varying j is b[p[i], p] (a row of b indexed by p)
    # So for fixed i: sum_j a[i,j] * b[p[i], p[j]] = dot(a[i, :], b[p[i], p])
    # Total = sum_i dot(a[i, :], b[p[i], p])
    # This is still O(n^2) if we do it naively, which is fine for n=256.
    # b[p[i], p] is row p[i] of b, then indexed by p -> b[p[i], p]
    # So: total = sum_i np.dot(a[i, :], b[p[i], p])
    # Vectorized: 
    # For each i, we need dot(a[i,:], b[p[i], p])
    # b[p[i], p] = b[p[i], p] - this is a fancy indexing
    # Let's do: 
    # b_perm = b[np.arange(n)[:, None], p]  # b_perm[i, j] = b[i, p[j]] ... no
    # We want b[p[i], p[j]]. 
    # Create matrix B where B[i,j] = b[p[i], p[j]]
    # B = b[np.array(p)[:, None], np.array(p)[None, :]]
    # Then cost = sum(a * B) = np.sum(a * B)
    p_arr = np.asarray(p, dtype=np.int64)
    B = b[p_arr[:, None], p_arr[None, :]]
    return int(np.sum(a * B))


def _swap_cost_delta(a, b, p, i, j):
    """Compute the change in cost if we swap facilities i and j in permutation p.
    p is a list/array where p[i] is the location of facility i.
    Swapping i and j means: new_p = p with p[i] and p[j] swapped.
    delta = cost(new_p) - cost(p)
    
    We can compute this in O(n) instead of O(n^2).
    cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    When we swap p[i] and p[j], the terms that change are those where k or l is i or j.
    Let's denote old_pi = p[i], old_pj = p[j]. After swap: new_pi = old_pj, new_pj = old_pi.
    
    The change involves:
    - Terms where k=i, l=i: a[i,i]*b[old_pi, old_pi] -> a[i,i]*b[old_pj, old_pj]
    - Terms where k=i, l=j: a[i,j]*b[old_pi, old_pj] -> a[i,j]*b[old_pj, old_pi]
    - Terms where k=j, l=i: a[j,i]*b[old_pj, old_pi] -> a[j,i]*b[old_pi, old_pj]
    - Terms where k=j, l=j: a[j,j]*b[old_pj, old_pj] -> a[j,j]*b[old_pi, old_pi]
    - Terms where k=i, l=k (k!=i,j): a[i,k]*b[old_pi, p[k]] -> a[i,k]*b[old_pj, p[k]]
    - Terms where k=j, l=k (k!=i,j): a[j,k]*b[old_pj, p[k]] -> a[j,k]*b[old_pi, p[k]]
    - Terms where l=i, k (k!=i,j): a[k,i]*b[p[k], old_pi] -> a[k,i]*b[p[k], old_pj]
    - Terms where l=j, k (k!=i,j): a[k,j]*b[p[k], old_pj] -> a[k,j]*b[p[k], old_pi]
    
    So delta = sum over these changes.
    """
    p_arr = np.asarray(p, dtype=np.int64)
    old_pi = p_arr[i]
    old_pj = p_arr[j]
    
    # Terms involving i and j directly
    delta = 0
    # k=i, l=i
    delta += a[i, i] * (b[old_pj, old_pj] - b[old_pi, old_pi])
    # k=i, l=j
    delta += a[i, j] * (b[old_pj, old_pi] - b[old_pi, old_pj])
    # k=j, l=i
    delta += a[j, i] * (b[old_pi, old_pj] - b[old_pj, old_pi])
    # k=j, l=j
    delta += a[j, j] * (b[old_pi, old_pi] - b[old_pj, old_pj])
    
    # Terms where k=i, l=k for k not in {i,j}
    # a[i,k] * (b[old_pj, p[k]] - b[old_pi, p[k]])
    # Terms where k=j, l=k for k not in {i,j}
    # a[j,k] * (b[old_pi, p[k]] - b[old_pj, p[k]])
    # Terms where l=i, k not in {i,j}
    # a[k,i] * (b[p[k], old_pj] - b[p[k], old_pi])
    # Terms where l=j, k not in {i,j}
    # a[k,j] * (b[p[k], old_pi] - b[p[k], old_pj])
    
    # Combine: for each k not in {i,j}:
    # a[i,k]*(b[old_pj,p[k]] - b[old_pi,p[k]]) + a[j,k]*(b[old_pi,p[k]] - b[old_pj,p[k]])
    # + a[k,i]*(b[p[k],old_pj] - b[p[k],old_pi]) + a[k,j]*(b[p[k],old_pi] - b[p[k],old_pj])
    # = (a[i,k] - a[j,k]) * (b[old_pj,p[k]] - b[old_pi,p[k]])
    # + (a[k,i] - a[k,j]) * (b[p[k],old_pj] - b[p[k],old_pi])
    
    # Vectorize over k
    mask = np.ones(len(p_arr), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    if len(ks) > 0:
        p_ks = p_arr[ks]
        # b[old_pj, p_ks] - b[old_pi, p_ks]
        diff1 = b[old_pj, p_ks] - b[old_pi, p_ks]
        # b[p_ks, old_pj] - b[p_ks, old_pi]
        diff2 = b[p_ks, old_pj] - b[p_ks, old_pi]
        
        # (a[i,k] - a[j,k]) for k in ks
        coeff1 = a[i, ks] - a[j, ks]
        # (a[k,i] - a[k,j]) for k in ks
        coeff2 = a[ks, i] - a[ks, j]
        
        delta += np.sum(coeff1 * diff1) + np.sum(coeff2 * diff2)
    
    return int(delta)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p_list = p.tolist()
    
    best_cost = _qap_cost(a, b, p_list)
    best_p = p_list[:]
    
    # Local search with 2-opt (swap) moves
    # Use a priority-based or random approach
    # For n=256, we can do multiple passes
    
    # Precompute some structures for speed
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    
    # Use float64 for intermediate computations but track integer cost
    # Actually, let's keep using the integer cost function but use float for delta computation
    
    # Local search: iterate over all pairs, apply improving swaps
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Randomize the order of pairs to avoid getting stuck
        order = np.random.permutation(n * (n - 1) // 2)
        for idx in order:
            if time.time() > deadline - 10.0:
                break
            # Convert linear index to (i, j) with i < j
            # idx = i*(n-1) - i*(i+1)//2 + (j-i-1) ... this is complex
            # Let's just iterate over i, j in random order
            pass
        
        # Simpler: iterate over all i, j pairs in random order
        for _ in range(n * (n - 1) // 2):
            if time.time() > deadline - 10.0:
                break
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            delta = _swap_cost_delta(a_arr, b_arr, p_list, i, j)
            if delta < 0:
                # Apply swap
                p_list[i], p_list[j] = p_list[j], p_list[i]
                best_cost += delta
                improved = True
    
    # If we have time, do a final polish with systematic 2-opt
    if time.time() < deadline - 20.0:
        # Systematic 2-opt: iterate over all pairs multiple times
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 10.0:
                        break
                    delta = _swap_cost_delta(a_arr, b_arr, p_list, i, j)
                    if delta < 0:
                        p_list[i], p_list[j] = p_list[j], p_list[i]
                        best_cost += delta
                        improved = True
            if not improved:
                break
    
    # Final check: if best_p is better than current p_list, return best_p
    # Actually, we've been updating p_list and best_cost together, so p_list is the best
    
    # Verify the cost
    final_cost = _qap_cost(a, b, p_list)
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = p_list[:]
    
    return best_p
# EVOLVE-BLOCK-END
