"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # Safety margin
    N = 30
    L = 2329089562800  # lcm(1..30)
    
    # Weights w_i = L / i
    weights = [L // i for i in range(1, N + 1)]
    
    best_coeffs = None
    best_abs_val = float('inf')
    
    # Helper to calculate sum
    def calc_sum(coeffs):
        s = 0
        for i in range(N):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i]
        return s

    # Local Search (Hill Climbing)
    def local_search(initial_coeffs, deadline):
        nonlocal best_abs_val, best_coeffs
        
        curr_coeffs = initial_coeffs[:]
        curr_sum = calc_sum(curr_coeffs)
        curr_abs = abs(curr_sum)
        
        if curr_abs > 0 and curr_abs < best_abs_val:
            best_abs_val = curr_abs
            best_coeffs = curr_coeffs[:]
            
        # If we found the minimum possible non-zero value (1), we can stop
        if best_abs_val == 1:
            return
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid deterministic traps
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    return
                
                old_val = curr_coeffs[i]
                # Try all 3 possible values for coefficient i
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    # Update sum incrementally
                    delta = (new_val - old_val) * weights[i]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum)
                    
                    if new_abs == 0:
                        continue # Must be non-zero
                    
                    if new_abs < curr_abs:
                        # Accept move
                        curr_coeffs[i] = new_val
                        curr_sum = new_sum
                        curr_abs = new_abs
                        improved = True
                        
                        if curr_abs < best_abs_val:
                            best_abs_val = curr_abs
                            best_coeffs = curr_coeffs[:]
                            
                            if best_abs_val == 1:
                                return
                        break # Restart loop or continue? Usually continue to next index or restart. Let's continue to next index.

    # Greedy Initialization Strategy
    # Start with all zeros, then add terms to minimize sum? 
    # Better: Random initialization with some structure.
    # Or: Greedy construction.
    
    # Strategy 1: Random Restarts with Hill Climbing
    max_restarts = 50
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Generate initial state
        # We can use a "greedy" initialization:
        # Start with 0s. Pick a random index, set to +/-1 to reduce sum?
        # Sum is 0 initially. Any move increases sum.
        # So random initialization is better.
        # To help convergence, maybe initialize with pairs that cancel?
        # c_i = 1, c_j = -1 where i, j are close?
        
        init_coeffs = [0] * N
        
        # Heuristic: Try to make a large partial sum, then cancel?
        # Or just random.
        # Let's try a mix: 50% random, 50% "structured"
        if restart_count % 2 == 0:
            # Random
            for i in range(N):
                init_coeffs[i] = random.choice([-1, 0, 1])
        else:
            # Structured: Pick random pairs (i, j) and set c_i=1, c_j=-1
            # This creates a sum of L/i - L/j = L*(j-i)/(ij)
            # This is often small.
            used = set()
            for _ in range(10): # 10 pairs
                i = random.randint(0, N-1)
                j = random.randint(0, N-1)
                if i == j or i in used or j in used:
                    continue
                init_coeffs[i] = 1
                init_coeffs[j] = -1
                used.add(i)
                used.add(j)
        
        local_search(init_coeffs, deadline)
        
        if best_abs_val == 1:
            break

    # Fallback if no valid non-zero sum was found (highly unlikely)
    if best_coeffs is None:
        # Return a simple valid non-zero sum
        # e.g., c_1 = 1, others 0 -> Sum = L/1 = L
        best_coeffs = [0] * N
        best_coeffs[0] = 1
        best_abs_val = weights[0]

    return best_coeffs
# EVOLVE-BLOCK-END
