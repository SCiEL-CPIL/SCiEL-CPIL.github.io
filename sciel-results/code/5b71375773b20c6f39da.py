"""QAP Solver using Hungarian Algorithm Global Reassignment and 2-opt Polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    a_sym[i, k] = a[i, k] + a[k, i]
    """
    x = p[i]
    y = p[j]
    
    # diff_a[k] = a_sym[i, k] - a_sym[j, k]
    diff_a = a_sym[i, :] - a_sym[j, :]
    
    # diff_b[k] = b[y, k] - b[x, k]
    diff_b = b[y, :] - b[x, :]
    
    # Calculate sum over all k first: sum_k diff_a[k] * diff_b[p[k]]
    # We need to exclude k=i and k=j from the sum because their positions change
    # and the formula derivation assumes p[k] is constant for k != i, j.
    
    # Vectorized calculation of the full sum
    temp = diff_a * diff_b[p]
    
    # Subtract terms for k=i and k=j
    # For k=i: diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[x]
    # For k=j: diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[y]
    
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _hungarian_step(a, b, p):
    """
    Perform a global reassignment using the Hungarian algorithm.
    We linearize the QAP cost by fixing the current assignment p and calculating
    the marginal cost of assigning each facility i to each location x, assuming
    other facilities remain at their current locations (or their average effect).
    
    Cost(i, x) = sum_j a[i, j] b[x, p[j]] + sum_j a[j, i] b[p[j], x]
    Since a and b are symmetric, this is 2 * sum_j a[i, j] b[x, p[j]].
    This is a linear assignment problem (LAP) relaxation of the QAP.
    """
    n = len(a)
    
    # Precompute the "interaction" with current assignment
    # C[i, x] = sum_j a[i, j] * b[x, p[j]]
    # We can compute this efficiently using matrix multiplication.
    # Let B_p be the matrix where B_p[x, j] = b[x, p[j]].
    # Then C = a @ B_p.T ? No.
    # C[i, x] = sum_j a[i, j] b[x, p[j]]
    # Let M = b[:, p] -> M[x, j] = b[x, p[j]]
    # Then C = a @ M.T ?
    # (a @ M.T)[i, x] = sum_j a[i, j] M.T[j, x] = sum_j a[i, j] M[x, j] = sum_j a[i, j] b[x, p[j]]
    # Yes.
    
    M = b[:, p]  # Shape (n, n), M[x, j] = b[x, p[j]]
    C = a @ M.T  # Shape (n, n), C[i, x] = sum_j a[i, j] b[x, p[j]]
    
    # The full linearized cost for symmetric a,b is 2*C. 
    # Minimizing C is equivalent to minimizing 2*C.
    # We solve the LAP: min sum_i C[i, p_new[i]]
    
    # linear_sum_assignment returns row_ind, col_ind such that row_ind is 0..n-1
    # and col_ind is the permutation.
    # We want p_new such that facility i is at location p_new[i].
    # linear_sum_assignment minimizes sum_i C[i, p_new[i]] if we pass C.
    # It returns (row_ind, col_ind) where row_ind is usually 0..n-1 if we don't need to permute rows?
    # Actually, linear_sum_assignment(cost) returns row_ind, col_ind.
    # If we want facility i assigned to location p[i], we need col_ind[i] = p[i].
    # The function returns indices such that cost[row_ind[k], col_ind[k]] are the matched pairs.
    # Since rows are facilities 0..n-1, row_ind will be 0..n-1 (possibly permuted, but we can map).
    # Actually, standard usage: row_ind, col_ind = linear_sum_assignment(C)
    # Then new_p[row_ind[k]] = col_ind[k].
    # Since we want a permutation of locations for all facilities, and rows are 0..n-1,
    # we can just do:
    row_ind, col_ind = linear_sum_assignment(C)
    
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    
    return new_p

def _local_search_2opt(a, b, a_sym, p, max_iter=1000, time_limit=None, start_time=None):
    """
    Perform 2-opt local search to refine the solution.
    """
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        # To make it faster, we can check a subset of swaps or use a first-improvement strategy
        # For robustness, let's do a full pass but break early if no improvement in a chunk?
        # n=256, n^2/2 = 32768 swaps.
        # Each swap delta is O(n) = 256 ops.
        # 32768 * 256 ~ 8M ops per pass. In NumPy, this is fast.
        
        # We'll iterate through all pairs.
        # To avoid O(n^2) Python loop overhead, we might need to be careful.
        # But 256^2 is small enough that even a Python loop might be okay if we vectorize the delta?
        # No, delta depends on i, j.
        # Let's stick to the previous O(n) delta calculation but in a loop.
        # 32k iterations * ~10-20us (NumPy overhead) = 0.3-0.6s per pass.
        # This is acceptable if we do only a few passes.
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a_sym, b, p, i, j)
                if delta < -1e-9: # Strict improvement
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # We could break here for first-improvement, but full pass is better for convergence
                    # However, full pass is expensive. Let's do first-improvement restart?
                    # No, standard 2-opt does full pass.
                    
    return p, current_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Precompute symmetric a for fast delta
    a_sym = a + a.T
    
    # Initial solution: Rank matching (greedy)
    # This provides a good starting point for the Hungarian step
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Main loop: Alternating Hungarian Global Reassignment and 2-opt Polishing
    # The Hungarian step escapes local minima by globally reassigning based on linearized costs.
    # The 2-opt step fixes the "inconsistencies" introduced by the linearization (which ignores 
    # that other facilities also move).
    
    while time.time() - start_time < time_limit:
        # 1. Hungarian Step
        new_p = _hungarian_step(a, b, p)
        
        # Check if it improved
        new_cost = _qap_cost(a, b, new_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
            p = new_p.copy()
        else:
            # If Hungarian didn't improve directly, maybe it got stuck in a different local min?
            # We can still try to polish the new_p, or stick to p.
            # Let's stick to p if new_p is worse, to avoid drifting to worse regions.
            # However, sometimes a worse intermediate step leads to a better final step.
            # Given the time constraint, let's be greedy: only accept if it improves or is equal.
            # If equal, accept to diversify?
            if new_cost <= best_cost:
                 p = new_p.copy()
            else:
                 # Try to polish the current best p instead?
                 # Or just retry Hungarian with a perturbed p?
                 pass

        # 2. 2-opt Polishing
        # Limit time for polishing to ensure we have time for more Hungarian steps
        polish_start = time.time()
        p, current_cost = _local_search_2opt(a, b, a_sym, p, max_iter=20, time_limit=time_limit - 5.0, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
