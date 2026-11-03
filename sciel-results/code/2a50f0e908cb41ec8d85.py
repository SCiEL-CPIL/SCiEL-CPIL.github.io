"""
Strategy: Iterated Tabu Search with Strategic Oscillation and Dynamic Neighborhood.
Replaces parallel island local search with a single, deep iterated tabu search.
Uses a tabu list to prevent cycling, strategic oscillation to escape deep local minima,
and a dynamic neighborhood that expands when stuck.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym=None):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the facility assigned to location k.
    """
    if a_sym is None:
        a_sym = a + a.T
    
    x = p[i]
    y = p[j]
    
    # Delta = sum_k (a_sym[i,k] - a_sym[j,k]) * (b[y,k] - b[x,k]) 
    # But we need to be careful with the i,j terms themselves.
    # Actually, the standard formula is:
    # delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[x, p[k]] - b[y, p[k]])
    #        + (a[i,j] - a[j,i]) * (b[x, y] - b[y, x])  # This part is tricky
    
    # A more robust vectorized way:
    # Let v = a_sym[i, :] - a_sym[j, :]
    # Let w = b[y, :] - b[x, :]
    # Then delta = sum_k v[k] * w[p[k]] - v[i]*w[x] - v[j]*w[y]
    
    v = a_sym[i, :] - a_sym[j, :]
    w = b[y, :] - b[x, :]
    
    # sum_k v[k] * w[p[k]]
    temp = v * w[p]
    delta = np.sum(temp) - v[i] * w[x] - v[j] * w[y]
    
    return delta

def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute symmetric A
    a_sym = a + a.T
    
    # Initial Solution: Greedy Construction
    # Start with empty permutation, assign facilities one by one to the best available location.
    # To make it faster and better, we can use a heuristic based on row sums.
    # Let's try a simple random start first, then improve.
    # Or better: Sort facilities by total flow (row sum of a) and locations by total distance (row sum of b).
    # Assign highest flow to lowest distance, etc.
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort indices
    fac_order = np.argsort(-a_row_sums)  # Highest flow first
    loc_order = np.argsort(b_row_sums)   # Lowest distance first (assuming we want to minimize)
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Randomize slightly to avoid deterministic local minima
    for _ in range(5):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Tabu Search Parameters
    tabu_tenure = 10
    max_iter_per_phase = 500
    num_phases = 50
    tabu = [None] * n  # tabu[i] stores the iteration count until which swap involving i is tabu
    current_iter = 0
    
    # Strategic Oscillation parameters
    osc_threshold = 50  # If no improvement for this many iterations, trigger oscillation
    
    # Main Loop
    while time.time() - start_time < time_limit:
        current_p = best_p.copy()
        current_cost = best_cost
        no_improve_count = 0
        phase_start_cost = current_cost
        
        for iter_idx in range(max_iter_per_phase):
            if time.time() - start_time > time_limit - 5:
                break
            
            current_iter += 1
            
            # Find best non-tabu move
            best_delta = 0
            best_i, best_j = -1, -1
            best_tabu_expiry = 0
            
            # Sample a subset of moves to speed up search? 
            # With n=256, n^2/2 = 32k moves. Checking all is feasible if delta is fast.
            # _swap_delta is vectorized, so it's fast.
            
            # To allow for some randomness and avoid getting stuck in a specific order,
            # we can shuffle the order of i, j checks or just iterate.
            # Let's iterate all pairs.
            
            for i in range(n):
                if tabu[i] > current_iter:
                    continue
                for j in range(i + 1, n):
                    if tabu[j] > current_iter:
                        continue
                    
                    delta = _swap_delta(a, b, current_p, i, j, a_sym)
                    
                    # Aspiration criterion: accept if it improves global best, even if tabu
                    # (We already skipped tabu moves above, so this is only for non-tabu moves
                    #  but we can also consider tabu moves if they are very good. 
                    #  Let's stick to simple tabu for now.)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                        # Tabu expiry for this move
                        best_tabu_expiry = current_iter + tabu_tenure
            
            if best_i == -1:
                # No non-tabu moves? This shouldn't happen with tenure < n
                break
                
            # Perform the move
            current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
            current_cost += best_delta
            
            # Update Tabu Lists
            tabu[best_i] = best_tabu_expiry
            tabu[best_j] = best_tabu_expiry
            
            # Check for improvement
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Strategic Oscillation
            if no_improve_count >= osc_threshold:
                # Trigger oscillation: perform a random move that is not tabu, or a large perturbation
                # Let's do a random perturbation of 3-5 swaps
                num_perturb = random.randint(3, 5)
                for _ in range(num_perturb):
                    i, j = random.sample(range(n), 2)
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost = _qap_cost(a, b, current_p)
                no_improve_count = 0
                # Clear tabu list to allow fresh search
                for k in range(n):
                    tabu[k] = 0
    
        # After each phase, if we haven't improved, we could try a different initial perturbation
        # But the oscillation inside the phase should handle this.
        
        # Final Polish: 2-opt local search on the best solution found
        if time.time() - start_time < time_limit - 2:
            improved = True
            while improved and time.time() - start_time < time_limit - 2:
                improved = False
                for i in range(n):
                    for j in range(i + 1, n):
                        if time.time() - start_time > time_limit - 2:
                            break
                        delta = _swap_delta(a, b, best_p, i, j, a_sym)
                        if delta < 0:
                            best_p[i], best_p[j] = best_p[j], best_p[i]
                            best_cost += delta
                            improved = True

    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
