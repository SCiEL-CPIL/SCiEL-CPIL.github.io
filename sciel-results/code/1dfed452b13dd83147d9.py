import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_abs = None
    best_coeffs = None
    
    n = 30
    
    # Precompute denominators as Fractions
    invs = [Fraction(1, i) for i in range(1, n+1)]
    
    # Initial best from seed
    seed_coeffs = [0] * 28 + [1, -1]
    seed_sum = Fraction(0)
    for i in range(n):
        if seed_coeffs[i] != 0:
            seed_sum += seed_coeffs[i] * invs[i]
    if seed_sum != 0:
        best_abs = abs(seed_sum)
        best_coeffs = seed_coeffs[:]
    else:
        best_abs = Fraction(1, 1) # Fallback
        best_coeffs = [1] + [0]*(n-1)

    # Local Search
    def evaluate(coeffs):
        s = Fraction(0)
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * invs[i]
        return s

    def local_search(coeffs, current_sum, time_limit):
        nonlocal best_abs, best_coeffs
        
        # Ensure current_sum is accurate
        # It's better to recompute if we're not sure, but we can track it.
        # Let's just recompute at start of each restart to be safe, 
        # but inside the loop, update incrementally.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_local_change = None
            best_local_sum = current_sum
            best_local_abs = abs(current_sum)
            
            # Try changing each coefficient
            for i in range(n):
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    
                    # Calculate new sum
                    # new_sum = current_sum + (new_val - old_val) * invs[i]
                    delta = (new_val - old_val) * invs[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        continue
                    
                    new_abs = abs(new_sum)
                    
                    if new_abs < best_local_abs:
                        best_local_abs = new_abs
                        best_local_sum = new_sum
                        best_local_change = (i, new_val)
            
            if best_local_change:
                i, new_val = best_local_change
                coeffs[i] = new_val
                current_sum = best_local_sum
                improved = True
                
                # Update global best
                if best_local_abs < best_abs:
                    best_abs = best_local_abs
                    best_coeffs = coeffs[:]
            else:
                # Local minimum reached
                break
                
        return coeffs, current_sum

    # Perform restarts
    # We can start with the seed, then random restarts
    # Random restarts might find better basins.
    
    # 1. Start with seed
    c = best_coeffs[:]
    s = evaluate(c)
    if s != 0:
        local_search(c, s, deadline)
    else:
        # If seed was somehow 0 (it isn't), fix it
        c = [1] + [0]*(n-1)
        s = evaluate(c)
        local_search(c, s, deadline)

    # 2. Random restarts
    while time.time() < deadline:
        c = [random.choice([-1, 0, 1]) for _ in range(n)]
        s = evaluate(c)
        
        # If s is 0, we need to move away from 0
        if s == 0:
             # Just pick a random non-zero move or restart
             # Actually, if s==0, we can't use it. 
             # We can force one coefficient to be non-zero if all are zero, 
             # but random choice ensures some are non-zero usually.
             # If s==0, we just skip this restart or make a small perturbation.
             # Let's just skip to next random config.
             continue

        local_search(c, s, deadline)
        
        # If we found a very small value, we might stop early? 
        # No, keep searching until deadline.

    # Ensure best_coeffs is valid and non-zero sum
    # If best_coeffs is None, return seed
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
