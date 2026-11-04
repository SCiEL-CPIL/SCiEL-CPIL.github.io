"""QAP solver with greedy construction + 2-opt local search + simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _init_greedy(a, b, n):
    """Greedy construction: place facilities one at a time minimizing incremental cost."""
    p = np.full(n, -1, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    # Precompute row sums of a for heuristic ordering
    a_row_sums = a.sum(axis=1)
    # Order facilities by descending row sum (place heavy-flow facilities first)
    order = np.argsort(-a_row_sums, kind='stable')
    
    # Incremental cost tracking: when we place facility f at location loc,
    # the added cost is sum over already-placed facilities g of a[f,g]*b[loc,loc_g] + a[g,f]*b[loc_g,loc]
    # We track for each unplaced facility f, the best location considering placed ones
    
    for idx in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if placed[loc]:
                continue
            # Compute incremental cost of placing idx at loc
            # Sum over placed facilities g: a[idx,g]*b[loc,p[g]] + a[g,idx]*b[p[g],loc]
            if np.any(placed):
                placed_locs = np.where(placed)[0]
                placed_facs = p[placed_locs]
                inc = np.sum(a[idx, placed_facs] * b[loc, placed_facs]) + \
                      np.sum(a[placed_facs, idx] * b[placed_facs, loc])
            else:
                inc = 0
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[best_loc] = idx
        placed[best_loc] = True
    
    # p[loc] = facility at location loc; we need p such that p[facility] = location
    # Currently p is location->facility mapping. Convert: result[facility] = location
    result = np.full(n, -1, dtype=np.int64)
    for loc in range(n):
        result[p[loc]] = loc
    return result


def _swap_delta(a, b, p, i, j):
    """Compute cost delta of swapping facilities at positions i and j.
    p is facility->location: p[f] = location of facility f.
    Swapping means: facility at loc_i goes to loc_j and vice versa.
    """
    # p[i] = loc_i, p[j] = loc_j
    # After swap: p[i] = loc_j, p[j] = loc_i
    loc_i = p[i]
    loc_j = p[j]
    
    # Delta = new_cost - old_cost
    # The only terms that change involve i or j in the (i,j) pair of the sum
    # sum_{k,l} a[k,l] * b[p[k], p[l]]
    # Terms involving i: a[i,l]*b[p[i],p[l]] for all l, and a[l,i]*b[p[l],p[i]] for all l
    # Terms involving j: a[j,l]*b[p[j],p[l]] for all l, and a[l,j]*b[p[l],p[j]] for all l
    # But terms a[i,j]*b[p[i],p[j]] and a[j,i]*b[p[j],p[i]] are counted in both, need care.
    
    # Old contribution from pairs involving i or j:
    # For l != i,j: a[i,l]*b[loc_i, p[l]] + a[l,i]*b[p[l], loc_i] + a[j,l]*b[loc_j, p[l]] + a[l,j]*b[p[l], loc_j]
    # Plus a[i,j]*b[loc_i, loc_j] + a[j,i]*b[loc_j, loc_i]
    
    # New: p[i]=loc_j, p[j]=loc_i
    # For l != i,j: a[i,l]*b[loc_j, p[l]] + a[l,i]*b[p[l], loc_j] + a[j,l]*b[loc_i, p[l]] + a[l,j]*b[p[l], loc_i]
    # Plus a[i,j]*b[loc_j, loc_i] + a[j,i]*b[loc_i, loc_j]
    
    others = np.ones(n, dtype=bool)
    others[i] = False
    others[j] = False
    other_locs = p[others]
    other_facs = np.where(others)[0]
    
    old_others = np.sum(a[i, other_facs] * b[loc_i, other_locs]) + \
                 np.sum(a[other_facs, i] * b[other_locs, loc_i]) + \
                 np.sum(a[j, other_facs] * b[loc_j, other_locs]) + \
                 np.sum(a[other_facs, j] * b[other_locs, loc_j])
    old_ij = a[i, j] * b[loc_i, loc_j] + a[j, i] * b[loc_j, loc_i]
    
    new_others = np.sum(a[i, other_facs] * b[loc_j, other_locs]) + \
                 np.sum(a[other_facs, i] * b[other_locs, loc_j]) + \
                 np.sum(a[j, other_facs] * b[loc_i, other_locs]) + \
                 np.sum(a[other_facs, j] * b[other_locs, loc_i])
    new_ij = a[i, j] * b[loc_j, loc_i] + a[j, i] * b[loc_i, loc_j]
    
    return (new_others + new_ij) - (old_others + old_ij)


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = 140.0  # safety margin
    
    # Phase 1: Greedy construction
    p = _init_greedy(a, b, n)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: 2-opt local search with multiple passes
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        for i in range(n):
            if (time.time() - t0) > deadline:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    # Phase 3: Simulated annealing
    if (time.time() - t0) < deadline:
        current_p = best_p.copy()
        current_c = best_cost
        T = max(1.0, best_cost * 0.001)
        T_end = max(1.0, best_cost * 1e-6)
        steps = 0
        max_steps = 500000
        while (time.time() - t0) < deadline and steps < max_steps:
            steps += 1
            # Pick random pair to swap
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = _swap_delta(a, b, current_p, i, j)
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_c += delta
                if current_c < best_cost:
                    best_cost = current_c
                    best_p = current_p.copy()
            # Cool down
            T = max(T_end, T * 0.9999)
    
    # Final check: ensure we return within deadline
    if (time.time() - t0) > deadline:
        # Return best found so far
        pass
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
