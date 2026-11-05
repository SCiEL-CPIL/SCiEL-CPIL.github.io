import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s deadline to be safe within 60s limit
    
    n = 30
    # Precompute unit fractions
    inv = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_sum = None
    best_coeffs = None
    best_abs = None
    
    def update_best(curr, curr_sum):
        nonlocal best_sum, best_coeffs, best_abs
        if curr_sum == 0:
            return
        curr_abs = abs(curr_sum)
        if best_abs is None or curr_abs < best_abs:
            best_sum = curr_sum
            best_coeffs = list(curr)
            best_abs = curr_abs

    def local_search_2opt(coeffs, time_limit):
        """
        Local search using 1-move and 2-move neighborhoods.
        2-move: change two coefficients simultaneously.
        """
        curr = list(coeffs)
        curr_sum = sum(c * inv[i] for i, c in enumerate(curr))
        
        if curr_sum == 0:
            return curr, curr_sum
            
        update_best(curr, curr_sum)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_gain = None
            best_move = None
            best_new_sum = curr_sum
            best_new_abs = abs(curr_sum)
            
            # 1-move neighborhood
            for i in range(n):
                for new_val in (-1, 0, 1):
                    if new_val == curr[i]:
                        continue
                    delta = (new_val - curr[i]) * inv[i]
                    new_sum = curr_sum + delta
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_new_sum = new_sum
                        best_gain = True
                        best_move = (i, new_val)
            
            # 2-move neighborhood (sampled to avoid O(n^2) full scan if too slow, but n=30 is small)
            # Full 2-move scan is 30*29/2 * (3*3 - 1) ~ 435 * 8 ~ 3500 evaluations.
            # Fraction addition is fast enough for this.
            for i in range(n):
                for j in range(i + 1, n):
                    for new_val_i in (-1, 0, 1):
                        if new_val_i == curr[i]:
                            continue
                        for new_val_j in (-1, 0, 1):
                            if new_val_j == curr[j]:
                                continue
                            delta = (new_val_i - curr[i]) * inv[i] + (new_val_j - curr[j]) * inv[j]
                            new_sum = curr_sum + delta
                            if new_sum == 0:
                                continue
                            new_abs = abs(new_sum)
                            if new_abs < best_new_abs:
                                best_new_abs = new_abs
                                best_new_sum = new_sum
                                best_gain = True
                                best_move = (i, new_val_i, j, new_val_j)
            
            if best_move is not None:
                if len(best_move) == 2:
                    idx, val = best_move
                    curr[idx] = val
                else:
                    i, v_i, j, v_j = best_move
                    curr[i] = v_i
                    curr[j] = v_j
                curr_sum = best_new_sum
                improved = True
                update_best(curr, curr_sum)
                
        return curr, curr_sum

    def simulated_annealing(coeffs, time_limit):
        curr = list(coeffs)
        curr_sum = sum(c * inv[i] for i, c in enumerate(curr))
        
        if curr_sum == 0:
            return curr, curr_sum
            
        update_best(curr, curr_sum)
        
        # Start with a temperature based on the current magnitude
        # We want to accept moves that increase |sum| slightly to escape
        T = abs(curr_sum) * 2 if curr_sum != 0 else 0.1
        cooling = 0.99
        min_T = 1e-12
        
        while (time.time() - start_time) < time_limit and T > min_T:
            # Perform a random 1-move or 2-move
            if random.random() < 0.5:
                # 1-move
                i = random.randint(0, n - 1)
                old_val = curr[i]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                delta = (new_val - old_val) * inv[i]
                new_sum = curr_sum + delta
            else:
                # 2-move
                i, j = random.sample(range(n), 2)
                old_i, old_j = curr[i], curr[j]
                new_i = random.choice([-1, 0, 1])
                new_j = random.choice([-1, 0, 1])
                if new_i == old_i and new_j == old_j:
                    continue
                delta = (new_i - old_i) * inv[i] + (new_j - old_j) * inv[j]
                new_sum = curr_sum + delta
            
            if new_sum == 0:
                # Zero is forbidden as a final answer, but during search, 
                # we can treat it as a very "good" state to bounce off, or just reject.
                # Rejecting keeps us away from the forbidden zone.
                continue
            
            curr_abs = abs(curr_sum)
            new_abs = abs(new_sum)
            
            if new_abs < curr_abs:
                # Accept if better
                if len(delta) > 0: # dummy check, delta is fraction
                    pass
                # Apply move
                if random.random() < 0.5:
                    curr[i] = new_val
                else:
                    curr[i] = new_i
                    curr[j] = new_j
                curr_sum = new_sum
                update_best(curr, curr_sum)
            else:
                # Accept if worse with probability exp(-delta/T)
                delta_val = new_abs - curr_abs
                if T > 0:
                    prob = math.exp(-delta_val / T)
                    if random.random() < prob:
                        if random.random() < 0.5:
                            curr[i] = new_val
                        else:
                            curr[i] = new_i
                            curr[j] = new_j
                        curr_sum = new_sum
                        update_best(curr, curr_sum)
            
            T *= cooling
        
        return curr, curr_sum

    random.seed(42)
    
    # 1. Start with a greedy-like initialization
    # Try to pair up terms to cancel out
    # e.g., 1/1 - 1/2 = 1/2, 1/3 - 1/4 = 1/12...
    # A simple random start is often fine, but let's try a few structured ones.
    
    # Initialization 1: Random
    init1 = [random.choice([-1, 0, 1]) for _ in range(n)]
    
    # Initialization 2: Alternating signs for first 30
    init2 = [1 if i % 2 == 0 else -1 for i in range(n)]
    
    # Initialization 3: Block of 1s and -1s
    init3 = [1]*15 + [-1]*15
    
    inits = [init1, init2, init3]
    
    # Phase 1: Local Search on initializations
    for init in inits:
        if (time.time() - start_time) > deadline:
            break
        curr, curr_sum = local_search_2opt(init, deadline)
    
    # Phase 2: Simulated Annealing with restarts
    # We use the best found so far as the starting point for SA, or random restarts
    # To be safe, we'll do a few SA runs starting from the best known solution
    # and some random ones.
    
    if best_coeffs is None:
        best_coeffs = init1
        best_sum = sum(c * inv[i] for i, c in enumerate(best_coeffs))
        best_abs = abs(best_sum)
        
    # Run SA starting from best_coeffs
    if (time.time() - start_time) < deadline:
        curr, curr_sum = simulated_annealing(best_coeffs, deadline)
        
    # Run SA from random restarts
    while (time.time() - start_time) < deadline:
        restart_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        # Quick local search first to get to a basin
        curr, curr_sum = local_search_2opt(restart_coeffs, start_time + 1.0) # 1s budget for local search
        # Then SA for remaining time in this iteration
        if (time.time() - start_time) < deadline:
            curr, curr_sum = simulated_annealing(curr, deadline)

    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
