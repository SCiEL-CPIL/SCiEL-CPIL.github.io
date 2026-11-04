"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    time_limit = 115.0  # leave some margin for safety
    start_time = time.time()
    
    # Constructive heuristic initialization: place facilities one by one
    # choosing the location that minimizes incremental cost.
    # This is much better than simple rank-matching.
    
    # Precompute for speed
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    # Greedy constructive heuristic: place each facility at the location
    # that gives the smallest incremental cost given already-placed facilities.
    # We process facilities in order of decreasing total flow (most connected first).
    
    facility_order = np.argsort(-a_int.sum(axis=1), kind='stable')
    
    p = np.full(n, -1, dtype=np.int64)
    used_locs = np.zeros(n, dtype=bool)
    
    # For incremental cost when placing facility f at location loc:
    # delta = sum_{g already placed} [a[f,g]*b[loc,p[g]] + a[g,f]*b[p[g],loc]]
    
    for idx in facility_order:
        best_loc = -1
        best_cost = None
        # Compute incremental cost for each unused location
        # For each candidate location, compute sum over already-placed facilities
        placed_mask = p >= 0
        if not placed_mask.any():
            # First facility: pick location with smallest row sum in b
            row_sums = b_int.sum(axis=1)
            unused_indices = np.where(~used_locs)[0]
            best_loc = unused_indices[np.argmin(row_sums[unused_indices])]
        else:
            placed_indices = np.where(placed_mask)[0]
            # For each unused location, compute incremental cost
            # delta(loc) = sum_{g in placed} [a[idx,g]*b[loc,p[g]] + a[g,idx]*b[p[g],loc]]
            # = sum_g a[idx,g]*b[loc,p[g]] + sum_g a[g,idx]*b[p[g],loc]
            # = b[loc, p[placed_indices]].dot(a[idx, placed_indices]) + b[p[placed_indices], loc].dot(a[placed_indices, idx])
            
            p_placed = p[placed_indices]
            a_row = a_int[idx, placed_indices]
            a_col = a_int[placed_indices, idx]
            
            unused_indices = np.where(~used_locs)[0]
            # b[loc, p_placed] for all unused locs: b[unused_indices, :][:, p_placed].dot(a_row)
            # This is b[unused_indices][:, p_placed] @ a_row
            # And b[p_placed, unused_indices] @ a_col
            
            # Compute efficiently
            term1 = b_int[unused_indices][:, p_placed].dot(a_row)
            term2 = b_int[p_placed][:, unused_indices].dot(a_col)
            costs = term1 + term2
            best_loc = unused_indices[np.argmin(costs)]
        
        p[idx] = best_loc
        used_locs[best_loc] = True
    
    # Compute initial cost
    def compute_cost(p_arr):
        return int(np.sum(a_int * b_int[np.ix_(p_arr, p_arr)]))
    
    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated annealing with 2-opt swaps
    # Precompute row sums of a for potential use
    a_row_sums = a_int.sum(axis=1)
    a_col_sums = a_int.sum(axis=0)
    
    # SA parameters
    T = 1.0  # initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iter = 500000
    
    current_p = p.copy()
    current_cost_val = current_cost
    
    # Precompute b row sums for potential use
    b_row_sums = b_int.sum(axis=1)
    
    iter_count = 0
    no_improve_count = 0
    patience = 50000
    
    while time.time() - start_time < time_limit:
        # Check time periodically
        if iter_count % 1000 == 0 and time.time() - start_time > time_limit:
            break
        
        # Choose two random facilities to swap
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        
        pi = current_p[i]
        pj = current_p[j]
        
        # Compute delta cost for swapping positions of i and j
        # delta = sum_{k != i,j} [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi]) 
        #                   + a[j,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pi]-b[pk,pj])]
        #       + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        
        # Efficient computation:
        # Let's compute:
        # For k not in {i,j}:
        #   a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi])
        #   + a[j,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pi]-b[pk,pj])
        
        # = (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk]) + (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])
        
        # Let's just compute directly using vectorized operations
        # Create masks
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = current_p[ks]
            # term1: a[i,ks]*(b[pj,pk]-b[pi,pk])
            diff_b1 = b_int[pj, pk] - b_int[pi, pk]
            term1 = a_int[i, ks] * diff_b1
            # term2: a[ks,i]*(b[pk,pj]-b[pk,pi])
            diff_b2 = b_int[pk, pj] - b_int[pk, pi]
            term2 = a_int[ks, i] * diff_b2
            # term3: a[j,ks]*(b[pi,pk]-b[pj,pk])
            diff_b3 = b_int[pi, pk] - b_int[pj, pk]
            term3 = a_int[j, ks] * diff_b3
            # term4: a[ks,j]*(b[pk,pi]-b[pk,pj])
            diff_b4 = b_int[pk, pi] - b_int[pk, pj]
            term4 = a_int[ks, j] * diff_b4
            
            delta_other = np.sum(term1) + np.sum(term2) + np.sum(term3) + np.sum(term4)
        else:
            delta_other = 0
        
        # Direct i-j terms
        delta_ij = a_int[i, j] * (b_int[pj, pi] - b_int[pi, pj]) + a_int[j, i] * (b_int[pi, pj] - b_int[pj, pi])
        
        delta = delta_other + delta_ij
        
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / max(T, 1e-10))):
            # Accept the move
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost_val += delta
            
            if current_cost_val < best_cost:
                best_cost = current_cost_val
                best_p = current_p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            T *= cooling_rate
        else:
            no_improve_count += 1
        
        iter_count += 1
        
        # Reheat if stuck
        if no_improve_count > patience and T < 0.1:
            T = 1.0
            no_improve_count = 0
    
    # Return the best solution found
    return best_p.tolist()
# EVOLVE-BLOCK-END
