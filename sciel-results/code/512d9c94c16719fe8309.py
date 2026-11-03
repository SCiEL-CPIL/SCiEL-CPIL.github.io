"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact integer QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _greedy_initial(a, b):
    """
    Greedy construction: assign facilities one by one to the best remaining location.
    Uses a heuristic based on row sums and pairwise interactions.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    remaining_locs = set(range(n))
    remaining_fac = set(range(n))
    
    # Precompute row sums for a quick heuristic
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    # Sort facilities by descending flow sum (heavy flows first)
    fac_order = np.argsort(-a_row_sums, kind="stable")
    
    for fac in fac_order:
        best_loc = -1
        best_cost = np.inf
        for loc in remaining_locs:
            # Cost contribution of placing fac at loc:
            # sum over already-placed facs j: a[fac, j] * b[loc, p[j]]
            # + sum over not-yet-placed facs j: a[fac, j] * b[loc, ?] -- approximate with b_col_sums
            # For speed, use a simple heuristic: minimize sum_{j in placed} a[fac,j]*b[loc,p[j]]
            # plus a proxy for future: a[fac].sum() * b[loc].mean()
            
            placed_cost = 0
            # This is O(n) per location, O(n^2) per facility, O(n^3) total -- too slow for n=256
            # Use vectorized approach instead
            
        # Vectorized: for each remaining loc, compute interaction with already placed
        if len(remaining_fac) > 1:
            # Build matrix of costs
            locs_arr = np.array(list(remaining_locs), dtype=np.int64)
            # For each candidate loc, cost = sum_{j in placed} a[fac, j] * b[loc, p[j]]
            placed_fac = [f for f in fac_order if f != fac]
            if placed_fac:
                placed_fac = np.array(placed_fac, dtype=np.int64)
                placed_locs = p[placed_fac]
                # a[fac, placed_fac] shape (k,), b[locs_arr, placed_locs] shape (m, k)
                costs = b[np.ix_(locs_arr, placed_locs)] @ a[fac, placed_fac]
            else:
                costs = np.zeros(len(locs_arr))
            # Add proxy for unplaced: a[fac, unplaced].sum() * b[loc, :].mean()
            unplaced = [f for f in fac_order if f != fac and f not in placed_fac]
            if unplaced:
                unplaced = np.array(unplaced, dtype=np.int64)
                proxy = a[fac, unplaced].sum() * b[locs_arr, :].mean(axis=1)
                costs = costs + proxy
            else:
                costs = costs + 0.0
            
            best_idx = np.argmin(costs)
            best_loc = locs_arr[best_idx]
        else:
            best_loc = next(iter(remaining_locs))
        
        p[fac] = best_loc
        remaining_locs.remove(best_loc)
        remaining_fac.discard(fac)
    
    return p


def _local_search_2opt(a, b, p, max_iters=100000, time_limit=100.0):
    """
    2-opt local search: try swapping pairs of facilities and accept if cost decreases.
    Uses efficient incremental cost update.
    """
    n = len(a)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    start_time = time.time()
    improved = True
    iter_count = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                # Compute delta cost of swapping p[i] and p[j]
                # Original: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
                # New: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
                # Plus interactions with all other k != i, j:
                # Original: sum_{k!=i,j} a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]]
                #          + sum_{k!=i,j} a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
                # New: sum_{k!=i,j} a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]]
                #     + sum_{k!=i,j} a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
                
                pi = best_p[i]
                pj = best_p[j]
                
                # Direct i-j interaction change
                delta_direct = (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
                                - a[i, j] * b[pi, pj] - a[j, i] * b[pj, pi])
                
                # Interaction with all k != i, j
                # For each k, the change is:
                # a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
                # - a[i,k]*b[pi,p[k]] - a[k,i]*b[p[k],pi] - a[j,k]*b[pj,p[k]] - a[k,j]*b[p[k],pj]
                
                # Vectorized over k
                k_idx = np.arange(n)
                mask = (k_idx != i) & (k_idx != j)
                pk = best_p[k_idx[mask]]
                
                # a[i,k] for k in mask: a[i, k_idx[mask]]
                ai_k = a[i, k_idx[mask]]
                aj_k = a[j, k_idx[mask]]
                ak_i = a[k_idx[mask], i]
                ak_j = a[k_idx[mask], j]
                
                # b[pi, pk], b[pj, pk], b[pk, pi], b[pk, pj]
                b_pi_pk = b[pi, pk]
                b_pj_pk = b[pj, pk]
                b_pk_pi = b[pk, pi]
                b_pk_pj = b[pk, pj]
                
                delta_k = (ai_k * b_pj_pk + ak_i * b_pk_pj + aj_k * b_pi_pk + ak_j * b_pk_pi
                          - ai_k * b_pi_pk - ak_i * b_pk_pi - aj_k * b_pj_pk - ak_j * b_pk_pj)
                
                total_delta = delta_direct + delta_k.sum()
                
                if total_delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += total_delta
                    improved = True
                    iter_count += 1
                    # Reset i to restart scan (simple restart)
                    break
            if improved:
                continue
    
    return best_p, best_cost


def solve(a, b):
    """
    Solve the QAP using greedy initialization + 2-opt local search.
    """
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Greedy initial solution
    p = _greedy_initial(a, b)
    cost = _qap_cost(a, b, p)
    
    # Local search
    p, cost = _local_search_2opt(a, b, p, time_limit=time_limit - (time.time() - start_time))
    
    # Final check
    if time.time() - start_time > time_limit:
        return p.tolist()
    
    # One more pass of local search if time permits
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        p, cost = _local_search_2opt(a, b, p, time_limit=remaining_time)
    
    return p.tolist()
# EVOLVE-BLOCK-END
