"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 30
    # Precompute denominators
    # We work with fractions to maintain exactness, or use a common denominator.
    # LCM of 1..30 is large, but Python handles big integers natively.
    # Let's use the LCM as the common denominator to compare magnitudes easily.
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Convert each term 1/i to L/i
    # weights[i] = L // i
    weights = [L // i for i in range(1, n + 1)]
    
    # Target: Minimize |sum(coeff[i] * weights[i])| subject to sum != 0
    
    best_val = None
    best_abs_val = float('inf')
    best_coeffs = [0] * n
    
    # Strategy: Random restarts with local search (hill climbing)
    # State is a list of -1, 0, 1.
    # Value is sum(c_i * w_i).
    
    def get_val(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * weights[i]
        return s

    def local_search(coeffs, current_val):
        nonlocal best_val, best_abs_val, best_coeffs
        improved = True
        while improved:
            improved = False
            # Try flipping each index
            for i in range(n):
                # Try changing coeffs[i] to -1, 0, 1 (if different)
                for new_c in [-1, 0, 1]:
                    if coeffs[i] == new_c:
                        continue
                    old_c = coeffs[i]
                    delta = (new_c - old_c) * weights[i]
                    new_val = current_val + delta
                    if new_val == 0:
                        continue # Exclude zero
                    abs_new_val = abs(new_val)
                    if abs_new_val < best_abs_val:
                        # Update best
                        best_abs_val = abs_new_val
                        best_val = new_val
                        best_coeffs = coeffs.copy()
                        
                    # Accept move if it improves local objective (reduces abs val)
                    # But be careful not to get stuck in 0.
                    if abs_new_val < abs(current_val):
                        coeffs[i] = new_c
                        current_val = new_val
                        improved = True
                        break # Restart loop to re-evaluate all i with new state
                if improved:
                    break
        return current_val

    # Run several random restarts
    num_restarts = 0
    while time.time() < deadline:
        # Random initialization
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        val = get_val(coeffs)
        
        if val == 0:
            # If zero, flip a random small weight term to break it
            # Or just let local search handle it, but local search skips 0.
            # If we start at 0, we can't improve. So pick a random non-zero start or force a move.
            # Let's just pick a random index and set it to 1 if it was 0, or flip if non-zero.
            idx = random.randint(0, n-1)
            if coeffs[idx] == 0:
                coeffs[idx] = 1
            else:
                coeffs[idx] = -coeffs[idx]
            val = get_val(coeffs)
            if val == 0:
                continue # Very unlikely, but skip if still 0

        # Perform local search
        local_search(coeffs, val)
        
        num_restarts += 1
        # If we found the theoretical minimum (1/L), we can stop?
        # The min non-zero sum is 1/L. In our integer representation, that is 1.
        if best_abs_val == 1:
            break

    # If no non-zero value found (highly unlikely), return the seed
    if best_val is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
