import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    # p is a numpy array of indices
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This is equivalent to:
    # sum_i sum_j a[i,j] * b[p[i], p[j]]
    # We can compute this by iterating over i and doing dot products?
    # Or just:
    # B_p = b[p[:, None], p[None, :]] # This creates an n x n matrix, memory heavy for n=256 (65k floats, fine)
    # cost = np.sum(a * B_p)
    
    # Alternative without full matrix creation:
    # cost = 0
    # for i in range(n):
    #    cost += np.dot(a[i], b[p[i], p])
    # This is n dot products of size n. n=256. 256 * 256 ops = 65k ops per cost eval? No, dot is n^2.
    # Total n^3 = 16M ops.
    # The matrix multiplication approach:
    # B_p = b[np.ix_(p, p)]
    # return np.sum(a * B_p)
    
    B_p = b[np.ix_(p, p)]
    return np.sum(a * B_p)

def _greedy_initial(a, b, n):
    """Create a good initial solution using a greedy heuristic."""
    a_row_sums = a.sum(axis=1)
    facility_order = np.argsort(-a_row_sums, kind='stable')
    
    assigned = np.full(n, -1, dtype=np.int64)
    used_locations = np.zeros(n, dtype=bool)
    
    for idx in facility_order:
        best_loc = -1
        best_cost = np.inf
        
        # Vectorize the incremental cost calculation for all unused locations?
        # inc_cost(loc) = a[idx, idx]*b[loc, loc] + 
        #                 sum_{k assigned} (a[idx, k]*b[loc, assigned[k]] + a[k, idx]*b[assigned[k], loc])
        
        # Let assigned_indices be the list of k where assigned[k] != -1
        # Let assigned_locs be the corresponding locations
        
        # We can compute the interaction with already assigned facilities:
        # Term 1: sum_k a[idx, k] * b[loc, assigned[k]]
        # This is like a dot product if we could index b[loc, :] by assigned[k].
        # But b is fixed.
        
        # For each candidate loc, we need:
        # 1. a[idx, idx] * b[loc, loc]
        # 2. sum_{k in assigned} a[idx, k] * b[loc, assigned[k]]
        # 3. sum_{k in assigned} a[k, idx] * b[assigned[k], loc]
        
        # Let's just loop over unused locations. n=256.
        # In the first step, 256 unused. In the last, 1 unused.
        # Average 128 unused.
        # For each unused loc, we sum over assigned k.
        # Total operations ~ sum_{m=0}^{n-1} (n-m) * m ~ n^3/6 = 2.8M ops.
        # This is fast enough in Python if not too many Python-level loops.
        # The inner sum is over assigned k. We can vectorize this.
        
        assigned_mask = assigned != -1
        assigned_locs = assigned[assigned_mask]
        assigned_facs = np.where(assigned_mask)[0]
        
        if len(assigned_locs) > 0:
            # a[idx, assigned_facs] is vector of flows from idx to assigned facs
            # b[loc, assigned_locs] is vector of distances from loc to assigned locs
            # We need to sum a[idx, k] * b[loc, assigned[k]] for each k.
            
            # We want to compute this for ALL unused locs at once.
            # unused_locs = np.where(~used_locations)[0]
            # For each unused loc L, cost_part1 = dot(a[idx, assigned_facs], b[L, assigned_locs])
            # This is a matrix multiplication: 
            # B_sub = b[unused_locs[:, None], assigned_locs[None, :]] # shape (num_unused, num_assigned)
            # A_row = a[idx, assigned_facs] # shape (num_assigned,)
            # costs1 = B_sub @ A_row # shape (num_unused,)
            
            # Similarly for cost_part2: sum a[k, idx] * b[assigned[k], L]
            # A_col = a[assigned_facs, idx] # shape (num_assigned,)
            # B_sub2 = b[assigned_locs[None, :], unused_locs[:, None]] # shape (num_assigned, num_unused)
            # costs2 = B_sub2.T @ A_col # shape (num_unused,)
            
            # cost_part0 = a[idx, idx] * b[unused_locs, unused_locs] # shape (num_unused,)
            
            unused_locs = np.where(~used_locations)[0]
            if len(unused_locs) > 0:
                B1 = b[unused_locs[:, None], assigned_locs[None, :]]
                c1 = B1 @ a[idx, assigned_facs]
                
                B2 = b[assigned_locs[None, :], unused_locs[:, None]]
                c2 = B2.T @ a[assigned_facs, idx]
                
                c0 = a[idx, idx] * b[unused_locs, unused_locs]
                
                total_costs = c0 + c1 + c2
                min_idx = np.argmin(total_costs)
                best_loc = unused_locs[min_idx]
                best_cost = total_costs[min_idx]
            else:
                # Should not happen
                best_loc = 0
        else:
            # No assigned facilities yet
            unused_locs = np.where(~used_locations)[0]
            # cost is just a[idx,idx]*b[loc,loc]
            # If a[idx,idx] is 0, all are 0, pick first.
            best_loc = unused_locs[0]
            
        assigned[idx] = best_loc
        used_locations[best_loc] = True
        
    return assigned.tolist()

def _local_search_2opt(a, b, p, time_limit=100):
    """Improve the solution using vectorized 2-opt style local search."""
    p = np.array(p, dtype=np.int64)
    n = len(p)
    start_time = time.time()
    current_cost = _qap_cost(a, b, p)
    
    random.seed(42)
    
    # Precompute nothing, just use vectorized delta.
    # We will do random swaps and accept if improvement.
    
    while time.time() - start_time < time_limit:
        i, j = random.sample(range(n), 2)
        if i == j: continue
        
        u = p[i]
        v = p[j]
        
        # Compute delta for swapping u and v
        # Delta = sum_{k != i, j} [ a[i,k](b[v, p[k]] - b[u, p[k]]) + a[j,k](b[u, p[k]] - b[v, p[k]]) + a[k,i](b[p[k], v] - b[p[k], u]) + a[k,j](b[p[k], u] - b[p[k], v]) ]
        #       + a[i,i](b[v,v] - b[u,u]) + a[j,j](b[u,u] - b[v,v]) + a[i,j](b[v,u] - b[u,v]) + a[j,i](b[u,v] - b[v,u])
        
        # Let's compute the sum over k efficiently.
        # We need vectors:
        # b_v_p = b[v, p]
        # b_u_p = b[u, p]
        # b_p_v = b[p, v]
        # b_p_u = b[p, u]
        
        # Note: b is symmetric? The problem statement says "tai256c". QAP matrices are usually symmetric?
        # tai256c is a QAP instance. Usually flow and distance matrices are symmetric.
        # If b is symmetric, b[v, p] == b[p, v].
        # Let's assume symmetry to halve the work, but verify with a check?
        # For safety, assume non-symmetric or just use the general formula.
        # Actually, QAP cost is sum a[i,j] b[p[i], p[j]]. If a and b are symmetric, cost is symmetric.
        # tai256c: checking literature, tai instances are typically symmetric.
        # Let's assume b is symmetric for speed, but if not, the formula above holds.
        # If b is symmetric:
        # delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) 
        #       + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
        #       + diagonal and cross terms.
        # If a is also symmetric, a[i,k]=a[k,i].
        # Then delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) * 2
        #       + a[i,i](b[v,v] - b[u,u]) + a[j,j](b[u,u] - b[v,v]) + 2*a[i,j]*(b[v,u] - b[u,v])
        
        # Let's implement the general case first to be safe, it's not much slower with vectorization.
        
        # Vectorized sum over k
        # We want to exclude i and j from the sum.
        # We can compute the full sum and subtract the terms for i and j.
        
        # Full sum S1 = sum_k a[i,k] * b[v, p[k]]
        # Full sum S2 = sum_k a[i,k] * b[u, p[k]]
        # Full sum S3 = sum_k a[j,k] * b[u, p[k]]
        # Full sum S4 = sum_k a[j,k] * b[v, p[k]]
        # Full sum S5 = sum_k a[k,i] * b[p[k], v]
        # Full sum S6 = sum_k a[k,i] * b[p[k], u]
        # Full sum S7 = sum_k a[k,j] * b[p[k], u]
        # Full sum S8 = sum_k a[k,j] * b[p[k], v]
        
        # This is 8 dot products of size n.
        # Each dot product is n multiplications. 8n ops.
        # n=256. 2048 ops per swap.
        # This is extremely fast.
        
        bv_p = b[v, p]
        bu_p = b[u, p]
        bp_v = b[p, v]
        bp_u = b[p, u]
        
        S1 = np.dot(a[i, :], bv_p)
        S2 = np.dot(a[i, :], bu_p)
        S3 = np.dot(a[j, :], bu_p)
        S4 = np.dot(a[j, :], bv_p)
        S5 = np.dot(a[:, i], bp_v)
        S6 = np.dot(a[:, i], bp_u)
        S7 = np.dot(a[:, j], bp_u)
        S8 = np.dot(a[:, j], bp_v)
        
        # Subtract terms where k=i and k=j
        # For S1 (k=i): a[i,i] * b[v, p[i]] = a[i,i] * b[v, u]
        # For S1 (k=j): a[i,j] * b[v, p[j]] = a[i,j] * b[v, v]
        S1_corr = S1 - a[i,i]*b[v,u] - a[i,j]*b[v,v]
        
        # For S2 (k=i): a[i,i] * b[u, u]
        # For S2 (k=j): a[i,j] * b[u, v]
        S2_corr = S2 - a[i,i]*b[u,u] - a[i,j]*b[u,v]
        
        # For S3 (k=i): a[j,i] * b[u, u]
        # For S3 (k=j): a[j,j] * b[u, v]
        S3_corr = S3 - a[j,i]*b[u,u] - a[j,j]*b[u,v]
        
        # For S4 (k=i): a[j,i] * b[v, u]
        # For S4 (k=j): a[j,j] * b[v, v]
        S4_corr = S4 - a[j,i]*b[v,u] - a[j,j]*b[v,v]
        
        # For S5 (k=i): a[i,i] * b[p[i], v] = a[i,i] * b[u, v]
        # For S5 (k=j): a[j,i] * b[p[j], v] = a[j,i] * b[v, v]
        S5_corr = S5 - a[i,i]*b[u,v] - a[j,i]*b[v,v]
        
        # For S6 (k=i): a[i,i] * b[u, u]
        # For S6 (k=j): a[j,i] * b[v, u]
        S6_corr = S6 - a[i,i]*b[u,u] - a[j,i]*b[v,u]
        
        # For S7 (k=i): a[i,j] * b[u, u]
        # For S7 (k=j): a[j,j] * b[v, u]
        S7_corr = S7 - a[i,j]*b[u,u] - a[j,j]*b[v,u]
        
        # For S8 (k=i): a[i,j] * b[u, v]
        # For S8 (k=j): a[j,j] * b[v, v]
        S8_corr = S8 - a[i,j]*b[u,v] - a[j,j]*b[v,v]
        
        # The sum part is:
        # (S1_corr - S2_corr) + (S3_corr - S4_corr) + (S5_corr - S6_corr) + (S7_corr - S8_corr)
        sum_delta = (S1_corr - S2_corr) + (S3_corr - S4_corr) + (S5_corr - S6_corr) + (S7_corr - S8_corr)
        
        # Diagonal and cross terms
        diag_delta = a[i,i]*(b[v,v] - b[u,u]) + a[j,j]*(b[u,u] - b[v,v]) + a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u])
        
        delta = sum_delta + diag_delta
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
    
    return p.tolist(), current_cost

def solve(a, b):
    n = len(a)
    
    # Get a good initial solution using greedy
    p = _greedy_initial(a, b, n)
    
    # Run local search to improve
    p, cost = _local_search_2opt(a, b, p, time_limit=110) # Leave 10s for overhead
    
    return p
