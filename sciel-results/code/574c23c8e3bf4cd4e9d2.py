"""
QAP solver: Linearized Greedy Construction + Fast Vectorized 2-Opt with
dynamic cost matrix updates and spectral-based multi-start.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])


def greedy_construct(a, b, rng, noise_scale=0.0):
    """
    Constructs a permutation using a linearized greedy approach.
    Maintains a dynamic cost vector that is updated in O(n) per assignment.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    
    # Initial cost vector: cost of placing facility i at location j
    # C[i, j] = sum_k a[i, k] * b[j, ?] -> initially 0, we build up
    # Actually, the greedy cost for placing facility i at location j given current partial assignment
    # is: sum_{k in assigned} a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j]
    # We maintain a vector `cost_vec` of length n for the current facility being placed?
    # No, we iterate facilities.
    
    # A better way:
    # Let P be the set of already assigned facilities.
    # For each unassigned facility i, the cost of assigning it to location j is:
    # c(i, j) = sum_{k in P} (a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j])
    # We can precompute for each location j the "current partial cost" from assigned facilities.
    
    # Let's maintain two vectors:
    # row_cost[j] = sum_{k in P} a[?, k] * b[j, p[k]]  -- wait, this depends on i.
    
    # Let's use the standard greedy update:
    # Start with empty assignment.
    # Maintain an array `delta_cost` of size n, where delta_cost[j] is the incremental cost
    # of assigning the *next* facility to location j, given the current partial assignment.
    # But which facility is next? We pick the one with minimum total cost.
    
    # This is the "greedy" approach where we pick the pair (i, j) with min marginal cost.
    # Marginal cost of assigning i to j given P:
    # MC(i, j) = sum_{k in P} (a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j])
    
    # We can maintain a matrix M of size n x n where M[i, j] = MC(i, j).
    # Initially M is 0.
    # When we assign facility i* to location j*, we update M for all remaining i, j:
    # M[i, j] += a[i, i*]*b[j, j*] + a[i*, i]*b[j*, j]
    
    # This update is O(n^2) per step, total O(n^3).
    # For n=150, n^3 = 3.375M. This is very fast.
    
    M = np.zeros((n, n), dtype=np.float64)
    remaining_facilities = set(range(n))
    remaining_locations = set(range(n))
    
    for _ in range(n):
        if not remaining_facilities:
            break
            
        # Find (i, j) in remaining sets with minimum M[i, j]
        min_val = np.inf
        best_i = -1
        best_j = -1
        
        # To speed this up, we can find min of M restricted to remaining sets
        # But creating a submatrix is O(n^2). Finding min is O(n^2).
        # Since we do this n times, total O(n^3).
        
        # Efficient min finding:
        # We can just iterate or use numpy min on a masked array.
        # For n=150, a simple loop or numpy operation is fine.
        
        # Mask M for remaining
        # Actually, just finding min over the whole M and checking if i,j are remaining is risky
        # if assigned values are low. We should set assigned M[i, :] and M[:, j] to inf.
        
        # Let's maintain M properly.
        # Find min in M
        idx = np.unravel_index(np.argmin(M), M.shape)
        i_star, j_star = idx
        
        # It's possible that i_star or j_star is already assigned if we didn't mask properly.
        # But we will mask them.
        
        p[i_star] = j_star
        remaining_facilities.remove(i_star)
        remaining_locations.remove(j_star)
        
        # Mask out assigned row and column
        M[i_star, :] = np.inf
        M[:, j_star] = np.inf
        
        # Update M for all remaining i, j
        # M[i, j] += a[i, i_star]*b[j, j_star] + a[i_star, i]*b[j_star, j]
        
        # Vectorized update:
        # a_i_star_col = a[:, i_star]  # shape (n,)
        # b_j_star_row = b[j_star, :]  # shape (n,)
        # a_i_star_row = a[i_star, :]  # shape (n,)
        # b_j_star_col = b[:, j_star]  # shape (n,)
        
        # M[i, j] += a[i, i_star]*b[j, j_star] + a[i_star, i]*b[j_star, j]
        # This is an outer product sum:
        # M += np.outer(a[:, i_star], b[j_star, :]) + np.outer(a[i_star, :], b[:, j_star])
        
        M += np.outer(a[:, i_star], b[j_star, :]) + np.outer(a[i_star, :], b[:, j_star])
        
        # Re-mask the newly assigned row/col (the update might have changed inf to inf+val which is inf,
        # but let's be safe)
        M[i_star, :] = np.inf
        M[:, j_star] = np.inf
        
    return p


def local_search(a, b, p, time_limit):
    """
    Fast 2-opt local search with first-improvement.
    Uses O(n) delta computation.
    """
    n = len(p)
    p = p.copy()
    current_cost = compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + time_limit - 1.0
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                x = p[i]
                y = p[j]
                
                # Delta calculation
                # Delta = (a[i,i]-a[j,j])*(b[y,y]-b[x,x]) + (a[i,j]-a[j,i])*(b[y,x]-b[x,y])
                #        + sum_{k!=i,j} (a[i,k]-a[j,k])*(b[y,p[k]]-b[x,p[k]])
                
                # Vectorized sum part
                diff_a = a[i, :] - a[j, :]
                diff_b = b[y, p] - b[x, p]
                
                total_dot = np.dot(diff_a, diff_b)
                
                # Subtract k=i and k=j terms
                term_ik = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
                term_jk = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
                
                sum_part = total_dot - term_ik - term_jk
                
                delta = (a[i, i] - a[j, j]) * (b[y, y] - b[x, x]) \
                      + (a[i, j] - a[j, i]) * (b[y, x] - b[x, y]) \
                      + sum_part
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    
                    # First improvement: break inner loop
                    break
        
        if not improved:
            # Local minimum reached. Perturb.
            if time.time() < deadline - 5:
                # Random swap perturbation
                for _ in range(5):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = compute_cost(a, b, p)
                
                # If perturbation didn't improve best, continue searching from new state
                # (it might find a better basin)
            
            # If we hit deadline, break
            if time.time() > deadline:
                break
        
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy:
    # 1. Generate a few diverse initial solutions using Greedy Construct with noise.
    # 2. Run local search on each.
    # 3. Keep the best.
    
    # We can also try a spectral-based init if time permits.
    
    num_starts = 5
    time_per_start = (deadline - time.time()) / num_starts - 2.0
    
    for k in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial solution
        if k == 0:
            # Plain greedy
            p_init = greedy_construct(a, b, rng)
        else:
            # Add noise to the cost matrix to diversify
            # We can't easily add noise to greedy_construct without modifying it.
            # Instead, we can shuffle the order of facility selection or add random swaps.
            p_init = greedy_construct(a, b, rng)
            # Apply some random swaps to diversify
            for _ in range(10):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    p_init[i], p_init[j] = p_init[j], p_init[i]
        
        # Run local search
        remaining = deadline - time.time() - 2.0
        if remaining < 5:
            break
            
        p_sol, c_sol = local_search(a, b, p_init, remaining)
        
        if c_sol < best_cost:
            best_cost = c_sol
            best_p = p_sol
    
    # If we still have time, do one more long local search on the best solution
    remaining = deadline - time.time() - 2.0
    if remaining > 5 and best_p is not None:
        p_sol, c_sol = local_search(a, b, best_p, remaining)
        if c_sol < best_cost:
            best_cost = c_sol
            best_p = p_sol
    
    if best_p is None:
        # Fallback
        best_p = np.arange(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
