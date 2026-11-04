"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)

def _decompose_qap(a, b, p):
    """
    Decompose QAP into assignment problem.
    For a given permutation p, the QAP cost can be written as:
    sum_i,j a[i,j] * b[p[i], p[j]]
    
    We can decompose this by fixing some assignments and solving the rest
    as an assignment problem.
    """
    n = len(a)
    # The cost matrix for the assignment problem when we fix the permutation
    # is not straightforward. Instead, we use a different approach.
    return p

def _solve_assignment_subproblem(a, b, fixed_facs, fixed_locs, p_current):
    """
    Solve the assignment subproblem for the remaining facilities.
    fixed_facs: list of facilities that are fixed
    fixed_locs: list of locations that are fixed
    p_current: current permutation
    
    Returns: new permutation
    """
    n = len(a)
    p = p_current.copy()
    
    # Build the cost matrix for the assignment problem
    # For each unassigned facility i and unassigned location l,
    # the cost of assigning i to l is:
    # sum_{j fixed} a[i,j] * b[l, p[j]] + sum_{j fixed} a[j,i] * b[p[j], l]
    # + sum_{k unassigned, k != i} a[i,k] * b[l, p[k]]  (approximate by ignoring or using current)
    
    # Simpler approach: use the current permutation as a baseline
    # and compute the incremental cost for swapping
    
    unassigned_facs = [f for f in range(n) if f not in fixed_facs]
    unassigned_locs = [l for l in range(n) if l not in fixed_locs]
    
    if len(unassigned_facs) == 0:
        return p
    
    # Build cost matrix
    m = len(unassigned_facs)
    cost_matrix = np.zeros((m, m), dtype=np.float64)
    
    for idx_i, fac in enumerate(unassigned_facs):
        for idx_l, loc in enumerate(unassigned_locs):
            # Cost of placing facility fac at location loc
            # = sum over all other facilities j of a[fac,j] * b[loc, p[j]]
            #   + sum over all other facilities j of a[j,fac] * b[p[j], loc]
            # But we need to exclude j=fac itself and handle the case where j is unassigned
            
            total_cost = 0
            for j in range(n):
                if j == fac:
                    continue
                # If j is fixed, use its current location
                # If j is unassigned, use its current (tentative) location
                loc_j = p[j]
                total_cost += a[fac, j] * b[loc, loc_j]
                total_cost += a[j, fac] * b[loc_j, loc]
            
            cost_matrix[idx_i, idx_l] = total_cost
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    for idx_i, idx_l in zip(row_ind, col_ind):
        p[unassigned_facs[idx_i]] = unassigned_locs[idx_l]
    
    return p

def _iterative_assignment(a, b, p_init, max_time, rng):
    """
    Iterative assignment: repeatedly fix a subset of facilities and solve
    the assignment problem for the rest.
    """
    n = len(a)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Number of facilities to fix in each iteration
    num_fixed = max(32, n // 8)
    
    while time.time() - start_time < max_time:
        # Randomly select facilities to fix
        fixed_facs = list(rng.choice(n, size=num_fixed, replace=False))
        
        # Get their current locations
        fixed_locs = [p[f] for f in fixed_facs]
        
        # Solve the assignment problem for the rest
        p_new = _solve_assignment_subproblem(a, b, set(fixed_facs), set(fixed_locs), p)
        
        cost_new = _qap_cost(a, b, p_new)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p = p_new.copy()
    
    return best_p, best_cost

def _pairwise_swap_improve(a, b, p, max_time):
    """
    Improve permutation using pairwise swaps with exact cost evaluation.
    """
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Precompute for faster swap evaluation
    # For each pair (i, j), compute the cost delta of swapping p[i] and p[j]
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > max_time:
                    break
                
                # Compute cost delta for swapping i and j
                li = p[i]
                lj = p[j]
                
                # New cost contribution for i and j
                # Old: a[i,i]*b[li,li] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj]
                #      + sum_{k!=i,j} a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li]
                #      + sum_{k!=i,j} a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]
                # New: a[i,i]*b[lj,lj] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
                #      + sum_{k!=i,j} a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj]
                #      + sum_{k!=i,j} a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]
                
                # Delta = new - old
                delta = (a[i,i] * (b[lj,lj] - b[li,li]) + 
                        a[i,j] * (b[lj,li] - b[li,lj]) +
                        a[j,i] * (b[li,lj] - b[lj,li]) +
                        a[j,j] * (b[li,li] - b[lj,lj]))
                
                # Add contributions from other facilities
                for k in range(n):
                    if k == i or k == j:
                        continue
                    lk = p[k]
                    delta += a[i,k] * (b[lj,lk] - b[li,lk])
                    delta += a[k,i] * (b[lk,lj] - b[lk,li])
                    delta += a[j,k] * (b[li,lk] - b[lj,lk])
                    delta += a[k,j] * (b[lk,li] - b[lk,lj])
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p, best_cost

def _init_random(a, b, rng):
    """Random initialization."""
    n = len(a)
    p = rng.permutation(n)
    return p

def _init_greedy_central(a, b):
    """Greedy initialization based on centrality."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    dist_centrality = b.sum(axis=0) + b.sum(axis=1)
    
    heavy = np.argsort(-flow_centrality, kind="stable")
    central = np.argsort(dist_centrality, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    max_total_time = 115.0
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy 1: Iterative assignment with different subsets
    # Run for ~40 seconds
    p_init = _init_greedy_central(a, b)
    p_result, cost_result = _iterative_assignment(a, b, p_init, 40.0, rng)
    
    if cost_result < best_cost:
        best_cost = cost_result
        best_p = p_result
    
    # Strategy 2: Pairwise swap improvement on the best solution
    if best_p is not None and time.time() - start_time < max_total_time - 30.0:
        p_improved, cost_improved = _pairwise_swap_improve(a, b, best_p, 30.0)
        if cost_improved < best_cost:
            best_cost = cost_improved
            best_p = p_improved
    
    # Strategy 3: Try another iterative assignment with random init
    if time.time() - start_time < max_total_time - 35.0:
        p_init2 = _init_random(a, b, rng)
        p_result2, cost_result2 = _iterative_assignment(a, b, p_init2, 35.0, rng)
        if cost_result2 < best_cost:
            best_cost = cost_result2
            best_p = p_result2
    
    # Final pairwise swap improvement
    if best_p is not None and time.time() - start_time < max_total_time - 2.0:
        p_final, cost_final = _pairwise_swap_improve(a, b, best_p, max_total_time - time.time() - 1.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
