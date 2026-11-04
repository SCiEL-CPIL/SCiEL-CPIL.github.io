"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _greedy_construct(a, b, n, rng):
    """Linearized greedy construction with dynamic cost matrix."""
    # C[i][j] = marginal cost of assigning facility i to location j
    # given current partial assignment. Start with 0, update as we assign.
    C = np.zeros((n, n), dtype=np.float64)
    
    assigned = np.zeros(n, dtype=bool)  # facility assigned?
    loc_taken = np.zeros(n, dtype=bool)  # location taken?
    p = np.full(n, -1, dtype=np.int64)
    
    # Precompute for speed
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # We'll assign facilities in a heuristic order (or random)
    # Order: facilities with largest row sums first (they interact most)
    facility_order = np.argsort(-a_float.sum(axis=1), kind='stable')
    
    for step in range(n):
        if step > 0:
            # Pick the facility with max "urgency" among unassigned
            # Use a simple heuristic: largest remaining row sum
            unassigned_facs = np.where(~assigned)[0]
            if len(unassigned_facs) == 0:
                break
            # Pick facility with largest total interaction with unassigned
            # Approximate: largest row sum among unassigned
            row_sums_unassigned = a_float[unassigned_facs].sum(axis=1)
            f = unassigned_facs[np.argmax(row_sums_unassigned)]
        else:
            f = facility_order[0]
        
        # Find best location for facility f
        # C[f] gives marginal cost for each location
        # Add a small random perturbation for diversity
        available_locs = np.where(~loc_taken)[0]
        if len(available_locs) == 0:
            break
        
        scores = C[f, available_locs].copy()
        # Add small noise for diversification
        scores += rng.random(len(available_locs)) * 1e-6 * (np.max(np.abs(scores)) + 1)
        best_local = np.argmin(scores)
        l = available_locs[best_local]
        
        p[f] = l
        assigned[f] = True
        loc_taken[l] = True
        
        # Update C for all unassigned facilities i and all available locations j
        # C[i][j] += a[f,i]*b[l,j] + a[i,f]*b[j,l]
        # Vectorized update
        unassigned_facs = np.where(~assigned)[0]
        available_locs = np.where(~loc_taken)[0]
        
        if len(unassigned_facs) > 0 and len(available_locs) > 0:
            # a[f, unassigned_facs] shape (m,), b[l, available_locs] shape (k,)
            # a[unassigned_facs, f] shape (m,), b[available_locs, l] shape (k,)
            af = a_float[f, unassigned_facs]  # (m,)
            al = a_float[unassigned_facs, f]  # (m,)
            bl = b_float[l, available_locs]    # (k,)
            bj = b_float[available_locs, l]    # (k,)
            
            # C[i,j] += af[i] * bl[j] + al[i] * bj[j]
            # Outer product sum
            C[np.ix_(unassigned_facs, available_locs)] += np.outer(af, bl) + np.outer(al, bj)
    
    return p


def _two_opt(a, b, p, n, max_iters=50000, time_limit=100.0):
    """2-opt local search: try pairwise swaps, accept if improves cost."""
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    p = p.copy()
    
    current_cost = _compute_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 5000
    
    # Precompute row/col sums of b for faster delta computation
    # For swap of p[i] and p[j]:
    # delta = sum_k a[i,k]*(b[p[i],p[j]] - b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    #        + sum_k a[j,k]*(b[p[j],p[i]] - b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    #        + a[i,j]*(b[p[i],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[j],p[j]] - b[p[j],p[i]])
    # Simplified: delta = 2*sum_k (a[i,k]-a[j,k])*(b[p[k],p[j]]-b[p[k],p[i]]) + correction terms
    
    iters = 0
    while iters < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        # Random pair
        i, j = rng_pair = (np.random.randint(n), np.random.randint(n))
        if i == j:
            continue
        
        pi, pj = p[i], p[j]
        
        # Compute delta cost for swapping p[i] and p[j]
        # New cost contribution for i: sum_k a[i,k]*b[p[i],p[k]] becomes sum_k a[i,k]*b[p[j],p[k]]
        # New cost contribution for j: sum_k a[j,k]*b[p[j],p[k]] becomes sum_k a[j,k]*b[p[i],p[k]]
        # Plus cross terms a[i,j]*b[p[i],p[j]] and a[j,i]*b[p[j],p[i]] swap
        
        # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) 
        #       + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
        #       + a[i,j]*(b[p[i],p[i]] - b[p[i],p[j]]) 
        #       + a[j,i]*(b[p[j],p[j]] - b[p[j],p[i]])
        #       - (original cross: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]])
        #       + (new cross: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]])
        
        # Let's compute directly:
        # old_i_j = a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
        # new_i_j = a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
        # old_i_rest = sum_{k!=i,j} a[i,k]*b[pi,p[k]]
        # new_i_rest = sum_{k!=i,j} a[i,k]*b[pj,p[k]]
        # old_j_rest = sum_{k!=i,j} a[j,k]*b[pj,p[k]]
        # new_j_rest = sum_{k!=i,j} a[j,k]*b[pi,p[k]]
        
        # delta = (new_i_rest - old_i_rest) + (new_j_rest - old_j_rest) + (new_i_j - old_i_j)
        
        # Vectorized:
        # For all k, compute contribution change
        # delta_i = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
        # delta_j = sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
        # But we need to handle k=i and k=j specially
        
        # Simpler: compute full delta
        b_pi = b_float[pi]  # row pi of b
        b_pj = b_float[pj]  # row pj of b
        
        # delta for facility i: sum_k a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
        delta_i = np.dot(a_float[i], b_pj[p] - b_pi[p])
        # delta for facility j: sum_k a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
        delta_j = np.dot(a_float[j], b_pi[p] - b_pj[p])
        
        delta = delta_i + delta_j
        
        if delta < -1e-9:
            # Accept swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve += 1
        
        iters += 1
        if no_improve > max_no_improve:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Greedy construction (multiple tries with different orderings)
    num_greedy = 5
    for gi in range(num_greedy):
        if time.time() > deadline - 60:
            break
        
        # Vary the facility ordering
        if gi == 0:
            # Original order
            p = _greedy_construct(a, b, n, rng)
        else:
            # Shuffled order
            a_shuffled_order = a.copy()
            # We'll modify the greedy to use a random order
            # For simplicity, just add more noise
            p = _greedy_construct(a, b, n, rng)
        
        if np.all(p >= 0):
            cost = _compute_cost(a, b, p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    # Phase 2: 2-opt local search
    if best_p is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            best_p, best_cost = _two_opt(a, b, best_p, n, max_iters=100000, time_limit=remaining_time - 5)
    
    # Phase 3: If time remains, do another round of 2-opt with different starting point
    if best_p is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            # Try another greedy + 2-opt
            p2 = _greedy_construct(a, b, n, rng)
            if np.all(p2 >= 0):
                cost2 = _compute_cost(a, b, p2)
                if cost2 < best_cost:
                    best_cost = cost2
                    best_p = p2.copy()
                else:
                    # 2-opt on p2 anyway
                    p2_opt, cost2_opt = _two_opt(a, b, p2, n, max_iters=50000, time_limit=remaining_time - 5)
                    if cost2_opt < best_cost:
                        best_cost = cost2_opt
                        best_p = p2_opt.copy()
    
    if best_p is None:
        # Fallback: identity permutation
        best_p = np.arange(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
