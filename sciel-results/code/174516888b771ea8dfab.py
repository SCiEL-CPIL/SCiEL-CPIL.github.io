"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    n = len(a)
    # Efficient: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[:, p[i]]) @ ... 
    # More directly: for each i, contribution is a[i,:] dotted with b[:, p[i]]? No.
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's compute via: for each i, for each j: a[i,j]*b[p[i],p[j]]
    # Vectorized: sum over i of (a[i,:] * b[p[i], p]) .sum()
    # b[p[i], p] is a row vector of length n
    total = 0
    for i in range(n):
        total += np.dot(a[i, :], b[p[i], p])
    return total


def _greedy_initial(a, b):
    """Rank-matching initial assignment."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search_2opt(a, b, p, deadline, max_iters=100000):
    """2-opt local search: try swapping pairs of facilities to reduce cost."""
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute row sums of a for fast delta calculation
    # Delta for swapping p[i] and p[j]:
    # New cost - Old cost = sum_k a[i,k]*(b[p[i],p[k]]_new - b[p[i],p[k]]_old)
    #   + sum_k a[j,k]*(b[p[j],p[k]]_new - b[p[j],p[k]]_old)
    #   + sum_k a[k,i]*(b[p[k],p[i]]_new - b[p[k],p[i]]_old)
    #   + sum_k a[k,j]*(b[p[k],p[j]]_new - b[p[k],p[j]]_old)
    # Where new/old refer to the swapped positions.
    
    # Let's use a simpler approach: compute the full delta by considering the effect
    # of swapping positions i and j in the permutation.
    
    # When we swap p[i] <-> p[j], the cost change is:
    # For all k != i,j: 
    #   a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
    # becomes
    #   a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
    # Plus the i-j and j-i terms change.
    
    # This is complex. Let's use a simpler 1-swap (relocate) or a direct delta computation.
    # For 2-opt swap of facilities at positions i and j:
    # delta = sum_{k!=i,j} [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    #         + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
    #       + a[i,j]*(b[p[j],p[j]] - b[p[i],p[i]]) + a[j,i]*(b[p[i],p[i]] - b[p[j],p[j]])
    #       + a[i,j]*(b[p[j],p[j]] - b[p[i],p[i]]) ... wait, let me be more careful.
    
    # Actually, let's just compute the delta by recomputing the affected terms.
    # The terms involving i or j in the sum sum_{u,v} a[u,v]*b[p[u],p[v]] change.
    # Terms: (i,v) for all v, (v,i) for all v, (j,v) for all v, (v,j) for all v.
    # But (i,i), (i,j), (j,i), (j,j) are counted once each.
    
    # Let's use a precomputed structure for speed.
    # Precompute for each facility i and location l: 
    #   col_i[l] = sum_{k} a[i,k] * b[l, p[k]]  (contribution of facility i to location l)
    #   row_i[l] = sum_{k} a[k,i] * b[p[k], l]  (contribution of location l to facility i)
    
    # This is O(n^2) to update per swap, which might be too slow for n=256 with many iterations.
    
    # Alternative: use a simpler but effective heuristic - random restarts with greedy + 2-opt.
    # Given time constraints, let's use a simpler local search: try all pairs, compute delta exactly.
    
    # For n=256, there are ~32k pairs. If we can evaluate each in O(1) or O(n), 
    # we can do several passes.
    
    # Let's precompute:
    # S1[i][l] = sum_{k} a[i,k] * b[l, p[k]]  -- cost contribution of assigning facility i to location l
    # S2[i][l] = sum_{k} a[k,i] * b[p[k], l]  -- cost contribution of assigning facility i to location l (from the other side)
    
    # Then the total cost is sum_i (S1[i][p[i]] + S2[i][p[i]]) but this double counts diagonal?
    # Actually cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
    # = sum_i sum_j a[i,j]*b[p[i],p[j]]
    # Let's define: for facility i at location l=p[i], its "out" contribution is sum_j a[i,j]*b[l, p[j]]
    # and its "in" contribution is sum_j a[j,i]*b[p[j], l]
    # Total = sum_i (out_i + in_i) but this counts each pair (i,j) twice? No, (i,j) is counted in out_i and in_j.
    # Actually sum_i out_i = sum_{i,j} a[i,j]*b[p[i],p[j]] = total cost. So we just need sum_i out_i.
    
    # Let's precompute out[i][l] = sum_j a[i,j] * b[l, p[j]] for all i, l.
    # This is O(n^3) to compute initially, which for n=256 is ~16M operations, feasible.
    
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for l in range(n):
            out[i, l] = np.dot(a[i, :], b[l, p])
    
    # Now current cost = sum_i out[i, p[i]]
    current_cost = np.sum(out[np.arange(n), p])
    
    # For swapping positions i and j (facilities at these positions swap locations):
    # New p': p'[i] = p[j], p'[j] = p[i], p'[k] = p[k] for k != i,j
    # Delta = (out[i, p[j]] + out[j, p[i]]) - (out[i, p[i]] + out[j, p[j]])
    # BUT this is not correct because out[i,l] depends on p[k] for all k, so changing p[i] and p[j] 
    # changes out for all facilities, not just i and j.
    
    # So this simple delta doesn't work. We need to account for the change in b[l, p[k]] for all k.
    
    # Given the complexity, let's use a different approach: 
    # 1. Start with greedy initial solution
    # 2. Use random restarts with perturbation
    # 3. Use a simple 2-opt that recomputes cost for candidate swaps (slower but correct)
    
    # Actually, let's use a simpler but effective method: 
    # - Multiple random restarts
    # - For each, do greedy + a few rounds of pairwise swaps with exact cost evaluation
    
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    # Try pairwise swaps
    for iteration in range(max_iters):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try a random pair
        i, j = random.sample(range(n), 2)
        if i == j:
            continue
        
        # Compute delta by recomputing affected terms
        # Save old values
        old_pi = p[i]
        old_pj = p[j]
        
        # Swap
        p[i], p[j] = old_pj, old_pi
        
        # Compute new cost - this is O(n^2) which is 65536 for n=256
        # Too slow for many iterations. Let's limit the number of iterations.
        
        new_cost = _qap_cost(a, b, p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p.copy()
        else:
            # Revert
            p[i], p[j] = old_pi, old_pj
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # Initial solution
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Local search with time limit
    best_p, best_cost = _local_search_2opt(a, b, p, deadline, max_iters=50000)
    
    # Random restarts if time permits
    for restart in range(10):
        if time.time() - start_time > deadline - 10:
            break
        
        # Perturb the best solution
        p = best_p.copy()
        for _ in range(n // 4):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        
        current_cost = _qap_cost(a, b, p)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Local search from this point
        p2, cost2 = _local_search_2opt(a, b, p, deadline, max_iters=10000)
        if cost2 < best_cost:
            best_cost = cost2
            best_p = p2
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
