"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # Efficient: C = b[p[:,None], p[None,:]] then cost = (a * C).sum()
    C = b[np.ix_(p, p)]
    return int(np.sum(a * C))


def _greedy_initial(a, b, n, rng):
    """
    Greedy construction: iteratively assign the unassigned facility with the
    highest 'pressure' to the location that minimizes marginal cost.
    Pressure for facility i given current partial assignment p_partial:
      sum over already-assigned j of a[i,j]*b[loc_i, p[j]] + a[j,i]*b[p[j], loc_i]
    We pick the facility with max total pressure (most constrained) and assign
    it to the location with minimum marginal cost.
    """
    p = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    # Precompute row sums of a for tie-breaking / initial pressure
    a_row_sum = a.sum(axis=1)
    
    # For speed, maintain running marginal costs
    # marginal[i, loc] = sum over assigned j of a[i,j]*b[loc, p[j]] + a[j,i]*b[p[j], loc]
    # But that's O(n^2) memory and O(n) update per assignment -> O(n^3) total, too slow for n=256? 
    # n=256, n^3 = 16M, that's fine actually.
    
    # Actually let's do a simpler greedy: 
    # 1. Start with random or rank-based initial
    # 2. Use local search
    
    # Better: use a fast greedy based on row/column sums
    # Assign facility i to location k to minimize sum_j a[i,j]*b[k,p[j]]
    # For a fast greedy, we can use the following heuristic:
    # Sort facilities by descending row sum of a
    # Sort locations by ascending row sum of b
    # Then do a 2-opt local search
    
    # Let's do a multi-start greedy with random perturbations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    # Start with rank-matching
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Add some randomness for diversity
    for _ in range(min(10, n)):
        i1 = rng.integers(n)
        i2 = rng.integers(n)
        p[i1], p[i2] = p[i2], p[i1]
    
    return p


def _local_search_2opt(a, b, p, max_iter, deadline, rng):
    """
    2-opt local search: try all pairs of swaps and accept the best improving swap.
    Use vectorized cost difference computation for speed.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Precompute: for a swap of positions i and j (facilities at locs p[i], p[j]),
    # the cost change can be computed efficiently.
    # Let u = p[i], v = p[j] (locations)
    # New cost = old cost - (terms involving u or v) + (new terms)
    # Specifically, for the QAP cost sum_{k,l} a[k,l]*b[p[k],p[l]]:
    # When we swap p[i] and p[j], only terms where k or l is i or j change.
    # Delta = sum_{k != i, k != j} a[k,i]*(b[p[k],v] - b[p[k],u]) + a[k,j]*(b[p[k],u] - b[p[k],v])
    #       + a[i,k]*(b[v,p[k]] - b[u,p[k]]) + a[j,k]*(b[u,p[k]] - b[v,p[k]])
    #       + a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u])
    #       + a[i,i]*(b[v,v] - b[u,u]) + a[j,j]*(b[u,u] - b[v,v])
    #       + a[i,i]*(b[v,u]... wait, let me be more careful.
    
    # Let's use a simpler approach: compute the full cost difference by 
    # only recomputing the rows/columns that change.
    # For swap of p[i] <-> p[j]:
    # The cost contribution of facility i changes from using location u to v
    # The cost contribution of facility j changes from using location v to u
    # And all other facilities' costs change because b[p[k], p[i]] and b[p[k], p[j]] change
    
    # Let's precompute B_p = b[np.ix_(p, p)] for fast access
    # But that's O(n^2) per iteration which is fine for n=256.
    
    # Actually, for n=256, a full 2-opt with O(n^2) swaps each taking O(n) to evaluate
    # is O(n^3) per pass = 16M operations, which is fast.
    # Let's do a full 2-opt pass: try all n*(n-1)/2 swaps, find best, if improving apply.
    
    while time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # For speed, do a random subset of swaps if n is large
        # For n=256, full 2-opt is 32640 swaps, each O(n) = 256 ops -> 8M ops per pass
        # That's about 10-50ms per pass in NumPy if vectorized, or 100ms+ in Python loops
        
        # Let's use a vectorized approach:
        # For each pair (i,j), compute the cost difference.
        # We can vectorize over j for each i.
        
        for i in range(n):
            if time.time() >= deadline:
                return best_p, best_cost
            u = best_p[i]
            # For all j != i, compute delta if we swap i and j
            js = np.arange(n)
            js = js[js != i]
            vs = best_p[js]  # locations for j
            
            # delta for swapping i and j where j has location v:
            # Terms involving i and j only:
            # Old: a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[i,i]*b[u,u] + a[j,j]*b[v,v]
            # New: a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[i,i]*b[v,v] + a[j,j]*b[u,u]
            # Self terms for i,j:
            #   a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v]) = (a[i,i]-a[j,j])*(b[v,v]-b[u,u])
            # Cross i,j terms:
            #   a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) = (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
            
            # Terms involving i and k (k != i, k != j):
            #   sum_k a[k,i]*(b[p[k],v]-b[p[k],u]) + a[i,k]*(b[v,p[k]]-b[u,p[k]])
            # Terms involving j and k (k != i, k != j):
            #   sum_k a[k,j]*(b[p[k],u]-b[p[k],v]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])
            
            # This is getting complex. Let's just recompute the cost for candidate swaps
            # using a fast method: maintain the cost matrix incrementally.
            
            # Simpler: just try random swaps and evaluate.
            pass
        
        # Let's use a different strategy: random restarts with 2-opt
        # Pick random pairs and try swapping
        for _ in range(2000):
            if time.time() >= deadline:
                return best_p, best_cost
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            new_p = best_p.copy()
            new_p[i], new_p[j] = new_p[j], new_p[i]
            new_cost = _qap_cost(a, b, new_p)
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p
                improved = True
        
        if not improved:
            # Try a larger perturbation: 3-opt or random restart
            if rng.random() < 0.3:
                # Random restart with small perturbation
                for _ in range(5):
                    i = rng.integers(n)
                    j = rng.integers(n)
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                best_cost = _qap_cost(a, b, best_p)
            else:
                break
    
    return best_p, best_cost


def _local_search_fast(a, b, p, deadline):
    """
    Fast local search using vectorized 2-opt.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Precompute B = b for fast indexing
    # For a swap of positions i, j with locations u, v:
    # We can compute delta efficiently using precomputed arrays.
    
    # Let's maintain C = b[np.ix_(p, p)] implicitly through p
    # For each swap (i,j), the cost change is:
    # delta = sum over all (k,l) of a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    # where p' is p with i,j swapped.
    # Only terms where k or l is in {i,j} change.
    
    # Let's precompute for speed:
    # For each location loc, and for each facility f, we have b[loc, *] and b[*, loc]
    
    # Vectorized 2-opt:
    # For a given i with location u = p[i]:
    # For all j, v = p[j]:
    # delta(i,j) = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v]) 
    #            + (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
    #            + sum_{k != i, k != j} [a[k,i]*(b[p[k],v]-b[p[k],u]) + a[i,k]*(b[v,p[k]]-b[u,p[k]])
    #                                  + a[k,j]*(b[p[k],u]-b[p[k],v]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])]
    
    # The sum over k can be rewritten:
    # sum_k a[k,i]*b[p[k],v] - sum_k a[k,i]*b[p[k],u] + sum_k a[i,k]*b[v,p[k]] - sum_k a[i,k]*b[u,p[k]]
    # + sum_k a[k,j]*b[p[k],u] - sum_k a[k,j]*b[p[k],v] + sum_k a[j,k]*b[u,p[k]] - sum_k a[j,k]*b[v,p[k]]
    
    # Let's define for each location loc:
    #   S1[loc] = sum_k a[k, *] * b[p[k], loc]  -- but this depends on which facility index
    # This is getting complicated. Let's just do the O(n) evaluation per swap in a tight loop.
    
    # For n=256, a full 2-opt pass is 32640 swaps. If each takes ~100ns (optimistic), 
    # that's 3ms per pass. In practice with Python, it'll be more like 10-50ms.
    # With 120s budget, we can do thousands of passes.
    
    # Let's do a proper vectorized 2-opt:
    # Precompute: for each pair of locations (u, v), we need b[u,v] and b[v,u].
    # For the sum over k, we can precompute:
    #   For each facility i and location loc: A_i_loc = sum_k a[k,i] * b[p[k], loc]
    #   But p changes, so we'd need to update these.
    
    # Alternative: just do random 2-opt with full cost reevaluation.
    # For n=256, _qap_cost takes O(n^2) = 65536 multiply-adds, which is ~0.1ms in NumPy.
    # So 2000 random swaps per pass = 0.2s per pass. Too slow.
    
    # Let's do a smarter approach: use the fact that only O(n) terms change per swap.
    
    # Fast delta computation for swap (i, j):
    # Let u = p[i], v = p[j]
    # delta = (a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[i,i]*b[v,v] + a[j,j]*b[u,u])
    #       - (a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[i,i]*b[u,u] + a[j,j]*b[v,v])
    #       + sum_{k != i, k != j} [a[k,i]*b[p[k],v] + a[i,k]*b[v,p[k]] + a[k,j]*b[p[k],u] + a[j,k]*b[u,p[k]]
    #                               - a[k,i]*b[p[k],u] - a[i,k]*b[u,p[k]] - a[k,j]*b[p[k],v] - a[j,k]*b[v,p[k]]]
    
    # Let's precompute column sums for fast evaluation:
    # col_sum[loc, f] = sum_k a[k,f] * b[p[k], loc]  -- but this is O(n^2) to update
    # Actually, let's just compute the delta in O(n) time per swap using NumPy vectorization.
    
    while time.time() < deadline:
        improved = False
        # Try all pairs in a random order
        indices = np.arange(n)
        rng = np.random.default_rng()
        
        # For speed, try a subset of swaps per pass
        for i in range(n):
            if time.time() >= deadline:
                break
            u = best_p[i]
            # Vectorize over all j != i
            js = np.concatenate([np.arange(i), np.arange(i+1, n)])
            vs = best_p[js]
            
            # Compute delta for each j
            # Self terms:
            # (a[i,i] - a[best_p_js_as_facility... wait, a is indexed by facility, not location
            # a[i,j] where j is facility index, but we're swapping positions i and j,
            # so facility i moves from u to v, facility j moves from v to u.
            
            # Let me redo this carefully.
            # p maps facility -> location. p[i] = location of facility i.
            # Swap: facility i goes to v, facility j goes to u.
            
            # Cost change for terms (k,l) where k or l is i or j:
            # Old contribution:
            #   sum_l a[i,l]*b[u,p[l]] + sum_k a[k,i]*b[p[k],u] 
            #   + sum_l a[j,l]*b[v,p[l]] + sum_k a[k,j]*b[p[k],v]
            #   - a[i,j]*b[u,v] - a[j,i]*b[v,u]  (double counted)
            #   + a[i,i]*b[u,u] + a[j,j]*b[v,v]  (self terms, already included above)
            # Wait, let's just use the full formula.
            
            # New contribution:
            #   sum_l a[i,l]*b[v,p'[l]] + sum_k a[k,i]*b[p'[k],v]
            #   + sum_l a[j,l]*b[u,p'[l]] + sum_k a[k,j]*b[p'[k],u]
            #   - a[i,j]*b[v,u] - a[j,i]*b[u,v]
            #   + a[i,i]*b[v,v] + a[j,j]*b[u,u]
            
            # Where p' differs from p only at i and j.
            
            # For k != i, k != j: p'[k] = p[k]
            # So the sum over k != i, j is the same in old and new except for the b values.
            
            # Let's compute delta = new_cost - old_cost:
            # delta = [sum_{k!=i,j} a[k,i]*(b[p[k],v]-b[p[k],u]) + a[i,k]*(b[v,p[k]]-b[u,p[k]])]
            #       + [sum_{k!=i,j} a[k,j]*(b[p[k],u]-b[p[k],v]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])]
            #       + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
            #       + a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v])
            
            # Vectorize over j:
            # For each j, compute the above.
            
            # Precompute for this i:
            #   a_i = a[i, :]  (row i of a)
            #   a_col_i = a[:, i]  (column i of a)
            
            a_i = a[i, :]
            a_col_i = a[:, i]
            
            # For each j in js:
            #   v = vs[j_idx]
            #   a_j = a[js[j_idx], :]
            #   a_col_j = a[:, js[j_idx]]
            
            # This is O(n) per j, so O(n^2) total for this i. Too slow for full 2-opt.
            
            # Let's just do random swaps with O(n) delta computation.
            pass
        
        # Random 2-opt with O(n) delta
        for _ in range(500):
            if time.time() >= deadline:
                return best_p, best_cost
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            u = best_p[i]
            v = best_p[j]
            
            # Compute delta in O(n)
            # Using the formula above
            a_i = a[i, :]
            a_j = a[j, :]
            a_col_i = a[:, i]
            a_col_j = a[:, j]
            
            # Mask out i and j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            pk = best_p[mask]
            
            # Term 1: sum_k a[k,i]*(b[p[k],v]-b[p[k],u]) + a[i,k]*(b[v,p[k]]-b[u,p[k]])
            t1 = np.sum(a_col_i[mask] * (b[pk, v] - b[pk, u]))
            t1 += np.sum(a_i[mask] * (b[v, pk] - b[u, pk]))
            
            # Term 2: sum_k a[k,j]*(b[p[k],u]-b[p[k],v]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])
            t2 = np.sum(a_col_j[mask] * (b[pk, u] - b[pk, v]))
            t2 += np.sum(a_j[mask] * (b[u, pk] - b[v, pk]))
            
            # Terms 3,4,5,6:
            t3 = a[i, j] * (b[v, u] - b[u, v])
            t4 = a[j, i] * (b[u, v] - b[v, u])
            t5 = a[i, i] * (b[v, v] - b[u, u])
            t6 = a[j, j] * (b[u, u] - b[v, v])
            
            delta = t1 + t2 + t3 + t4 + t5 + t6
            
            if delta < 0:
                best_p[i], best_p[j] = best_p[j], best_p[i]
                best_cost += int(delta)
                improved = True
        
        if not improved:
            # Perturb
            for _ in range(3):
                i = rng.integers(n)
                j = rng.integers(n)
                best_p[i], best_p[j] = best_p[j], best_p[i]
            best_cost = _qap_cost(a, b, best_p)
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Multi-start: generate several initial solutions and run local search on each
    best_p = None
    best_cost = np.iinfo(np.int64).max
    
    # Start 1: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    # Start 2: random
    p1 = rng.permutation(n).astype(np.int64)
    
    # Start 3: rank-matching with perturbation
    p2 = p0.copy()
    for _ in range(10):
        i = rng.integers(n)
        j = rng.integers(n)
        p2[i], p2[j] = p2[j], p2[i]
    
    starts = [p0, p1, p2]
    
    for idx, p_start in enumerate(starts):
        if time.time() >= deadline:
            break
        
        # Run local search
        p_ls, cost_ls = _local_search_fast(a, b, p_start.copy(), deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
        
        # Use remaining time for another start if possible
        if time.time() < deadline - 30.0 and idx < len(starts) - 1:
            continue
        else:
            break
    
    if best_p is None:
        best_p = p0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
