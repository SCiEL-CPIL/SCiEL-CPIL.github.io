"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _greedy_initial(a, b):
    """Greedy construction: assign facilities one by one to minimize incremental cost."""
    n = len(a)
    # Start with rank-matching as a base
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Try to improve with a simple greedy refinement
    # For each facility, try all locations and pick the best
    # This is O(n^3) but with n=256, that's manageable if we do it a few times
    # Actually, let's just return the rank-matching for now and rely on local search
    return p


def _local_search_2opt(a, b, p, max_iter=10000, time_limit=10.0):
    """2-opt local search for QAP."""
    n = len(a)
    p = p.copy()
    best_cost = _qap_cost(a, b, p)
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Compute delta cost for swapping p[i] and p[j]
                # delta = sum_k a[i,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + sum_k a[j,k]*(b[p[j],p[k]] - b[p[i],p[k]])
                #       + sum_k a[k,i]*(b[p[k],p[i]] - b[p[k],p[j]]) + sum_k a[k,j]*(b[p[k],p[j]] - b[p[k],p[i]])
                pi, pj = p[i], p[j]
                
                # Precompute the contributions
                # Term 1: i fixed, j changes from pj to pi
                # Term 2: j fixed, i changes from pi to pj
                # Term 3: all k, i's position changes
                # Term 4: all k, j's position changes
                
                # More efficient delta computation:
                # delta = sum_k != i,j a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) 
                #       + sum_k != i,j a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
                #       + sum_k != i,j a[k,i]*(b[p[k],pj] - b[p[k],pi])
                #       + sum_k != i,j a[k,j]*(b[p[k],pi] - b[p[k],pj])
                #       + a[i,j]*(b[pj,pj] - b[pi,pj]) + a[j,i]*(b[pi,pi] - b[pj,pi])
                #       + a[i,j]*(b[pj,pj] - b[pi,pj]) ... wait, let me be more careful
                
                # Actually, the cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
                # When we swap positions of facilities i and j:
                # - a[i,j]*b[p[i],p[j]] becomes a[i,j]*b[p[j],p[i]]
                # - a[j,i]*b[p[j],p[i]] becomes a[j,i]*b[p[i],p[j]]
                # - For k != i,j: a[i,k]*b[p[i],p[k]] becomes a[i,k]*b[p[j],p[k]]
                # - For k != i,j: a[j,k]*b[p[j],p[k]] becomes a[j,k]*b[p[i],p[k]]
                # - For k != i,j: a[k,i]*b[p[k],p[i]] becomes a[k,i]*b[p[k],p[j]]
                # - For k != i,j: a[k,j]*b[p[k],p[j]] becomes a[k,j]*b[p[k],p[i]]
                
                # delta = a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
                #       + sum_{k != i,j} a[i,k]*(b[pj,pk] - b[pi,pk])
                #       + sum_{k != i,j} a[j,k]*(b[pi,pk] - b[pj,pk])
                #       + sum_{k != i,j} a[k,i]*(b[pk,pj] - b[pk,pi])
                #       + sum_{k != i,j} a[k,j]*(b[pk,pi] - b[pk,pj])
                
                # Note that b is not necessarily symmetric, so b[pj,pi] != b[pi,pj]
                
                # Let's compute this efficiently using precomputed row sums
                # Actually, for n=256, let's just compute it directly but vectorized
                pi_val = pi
                pj_val = pj
                
                # Create mask for k != i, j
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                k_idx = np.where(mask)[0]
                
                if len(k_idx) == 0:
                    delta = a[i,j]*(b[pj_val, pi_val] - b[pi_val, pj_val]) + a[j,i]*(b[pi_val, pj_val] - b[pj_val, pi_val])
                else:
                    pk = p[k_idx]
                    delta = (a[i,j]*(b[pj_val, pi_val] - b[pi_val, pj_val]) 
                           + a[j,i]*(b[pi_val, pj_val] - b[pj_val, pi_val])
                           + np.sum(a[i, k_idx]*(b[pj_val, pk] - b[pi_val, pk]))
                           + np.sum(a[j, k_idx]*(b[pi_val, pk] - b[pj_val, pk]))
                           + np.sum(a[k_idx, i]*(b[pk, pj_val] - b[pk, pi_val]))
                           + np.sum(a[k_idx, j]*(b[pk, pi_val] - b[pk, pj_val])))
                
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
        
        if not improved:
            break
    
    return p, best_cost


def _hungarian_reassign(a, b, p, time_limit=5.0):
    """Use Hungarian algorithm on a linearized surrogate to reassign facilities."""
    n = len(a)
    start_time = time.time()
    
    # Linearize the QAP: for each facility i, the cost of placing it at location l
    # is approximately sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
    # This is a surrogate that ignores interactions between facilities in the assignment
    
    # Build the cost matrix for Hungarian
    # cost[i, l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
    #             = b[l, :].dot(a[i, :]) ... no, that's not right
    
    # cost[i, l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
    # Let's compute this matrix
    
    cost_mat = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        # For facility i, compute cost of placing at each location l
        # term1[l] = sum_j a[i,j] * b[l, p[j]] = b[l, :][p] dot a[i, :] ... 
        # Actually: sum_j a[i,j] * b[l, p[j]] = (b[l, :])[p] @ a[i, :]
        # term2[l] = sum_j a[j,i] * b[p[j], l] = (b[:, l])[p] @ a[:, i]
        
        # Let's vectorize over l
        # term1: for each l, sum_j a[i,j] * b[l, p[j]]
        # This is b[l, p] @ a[i, :]
        # So term1 = b[:, p] @ a[i, :]  -- wait, b[:, p] is n x n, a[i,:] is n
        # b[:, p] has shape (n, n), and we want for each row l: sum_j (b[:, p])[l, j] * a[i, j]
        # That's (b[:, p] @ a[i, :]) which gives a vector of length n
        
        # term2: for each l, sum_j a[j,i] * b[p[j], l]
        # b[p, :] has shape (n, n), and we want for each column l: sum_j b[p, :][j, l] * a[j, i]
        # That's (b[p, :] @ a[:, i]) which gives a vector of length n
        
        term1 = b[:, p] @ a[i, :]
        term2 = b[p, :] @ a[:, i]
        cost_mat[i, :] = term1 + term2
    
    # Now solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    
    # Check if this is better
    new_cost = _qap_cost(a, b, new_p)
    current_cost = _qap_cost(a, b, p)
    
    if new_cost < current_cost:
        return new_p, new_cost
    else:
        return p, current_cost


def _perturb(p, n, num_swaps=10):
    """Perturb the permutation by random swaps."""
    p = p.copy()
    for _ in range(num_swaps):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    """Solve the QAP instance tai256c."""
    n = len(a)
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Local search
    p, best_cost = _local_search_2opt(a, b, p, max_iter=5000, time_limit=30.0)
    
    # Multiple restarts with Hungarian reassignment
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    current = p.copy()
    restart_count = 0
    
    while time.time() < deadline - 20.0:
        restart_count += 1
        
        # Perturb
        current = _perturb(current, n, num_swaps=5 + restart_count)
        
        # Local search
        current, cost = _local_search_2opt(a, b, current, max_iter=3000, time_limit=15.0)
        
        # Hungarian reassignment
        current, cost = _hungarian_reassign(a, b, current, time_limit=5.0)
        
        # Final local search
        current, cost = _local_search_2opt(a, b, current, max_iter=2000, time_limit=10.0)
        
        if cost < best_cost:
            best_cost = cost
            best_p = current.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
