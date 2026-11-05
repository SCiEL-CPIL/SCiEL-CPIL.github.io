import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 50.0  # 50 seconds to be safe
    
    n = 30
    # Precompute the terms
    terms = [1.0 / i for i in range(1, n + 1)]
    
    best_sum = None
    best_coeff = None
    best_abs = float('inf')
    
    # Helper to compute sum
    def compute_sum(coeff):
        s = 0.0
        for i in range(n):
            if coeff[i] != 0:
                s += coeff[i] * terms[i]
        return s
    
    # Helper to try local search
    def local_search(coeff):
        nonlocal best_sum, best_coeff, best_abs
        current_sum = compute_sum(coeff)
        current_abs = abs(current_sum)
        if current_abs == 0:
            return coeff, current_sum, current_abs
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            best_local = None
            best_local_abs = current_abs
            
            for i in range(n):
                for new_val in [-1, 0, 1]:
                    if new_val == coeff[i]:
                        continue
                    old_val = coeff[i]
                    coeff[i] = new_val
                    new_sum = current_sum + (new_val - old_val) * terms[i]
                    new_abs = abs(new_sum)
                    if new_abs > 0 and new_abs < best_local_abs:
                        best_local_abs = new_abs
                        best_local = i
                        best_local_val = new_val
                    coeff[i] = old_val
            
            if best_local is not None:
                coeff[best_local] = best_local_val
                current_sum = compute_sum(coeff)
                current_abs = best_local_abs
                improved = True
        
        # Update global best
        if current_abs < best_abs:
            best_abs = current_abs
            best_sum = current_sum
            best_coeff = coeff[:]
        
        return coeff, current_sum, current_abs
    
    # Run multiple random restarts
    num_restarts = 100
    for r in range(num_restarts):
        if (time.time() - start_time) >= deadline:
            break
        
        # Random initialization
        coeff = [random.choice([-1, 0, 1]) for _ in range(n)]
        local_search(coeff)
    
    # Also try a greedy initialization
    # Start with all zeros, then add terms in order of 1/i (largest first) to minimize sum
    coeff = [0] * n
    current_sum = 0.0
    for i in range(n - 1, -1, -1):  # From 1/30 to 1/1
        for val in [-1, 1]:
            new_sum = current_sum + val * terms[i]
            if abs(new_sum) < abs(current_sum) or current_sum == 0:
                if abs(new_sum) > 0:
                    coeff[i] = val
                    current_sum = new_sum
                    break
    local_search(coeff)
    
    # Also try the seed solution as a starting point
    coeff = [0] * 28 + [1, -1]
    local_search(coeff)
    
    # Return the best found
    if best_coeff is None:
        # Fallback to seed
        return [0] * 28 + [1, -1]
    
    return best_coeff
# EVOLVE-BLOCK-END
