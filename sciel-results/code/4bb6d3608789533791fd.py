"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    n = 30
    # LCM(1..30)
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    # Weights: w[i] = LCM / (i+1) for i in 0..29 (since coefficient[i-1]/i means coeff[0]/1, coeff[1]/2, ...)
    # Actually coefficient[i-1]/i: index 0 -> /1, index 1 -> /2, ..., index 29 -> /30
    # So weight for position j (0-indexed) is LCM / (j+1)
    weights = [lcm_val // (j + 1) for j in range(n)]
    
    # Each coefficient is in {-1, 0, 1}
    # Sum = sum(coeff[j] * weights[j])
    # We want smallest nonzero absolute value of this sum
    
    # Meet in the middle: split into two halves
    half = n // 2
    first_half_weights = weights[:half]
    second_half_weights = weights[half:]
    
    # Generate all possible sums for first half: 3^15 = 14348907
    # This is manageable
    first_sums = [0]
    for w in first_half_weights:
        new_sums = []
        for s in first_sums:
            new_sums.append(s - w)
            new_sums.append(s)
            new_sums.append(s + w)
        first_sums = new_sums
    
    # For second half, generate all sums and sort
    second_sums = [0]
    for w in second_half_weights:
        new_sums = []
        for s in second_sums:
            new_sums.append(s - w)
            new_sums.append(s)
            new_sums.append(s + w)
        second_sums = new_sums
    
    second_sums.sort()
    
    # For each sum in first_sums, find if there's a sum in second_sums such that
    # first_sum + second_sum = 0 (i.e., second_sum = -first_sum)
    # If so, total sum = 0, which is excluded.
    # We want smallest nonzero |first_sum + second_sum|
    
    # Use binary search in second_sums to find closest to -first_sum
    from bisect import bisect_left, bisect_right
    
    best_abs = None
    best_coeff = None
    
    for fs in first_sums:
        target = -fs
        idx = bisect_left(second_sums, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(second_sums):
            candidates.append(second_sums[idx])
        if idx > 0:
            candidates.append(second_sums[idx - 1])
        
        for ss in candidates:
            total = fs + ss
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                # Need to recover coefficients - skip for now, do local search
                best_coeff = None
    
    # Also check if total=0 is achievable (it is, all zeros)
    # We need smallest nonzero
    
    # Now we need to find actual coefficients that achieve best_abs
    # Use random search + local optimization
    
    # First, try to find a solution with the best_abs found
    # Use random restarts with local search
    
    def random_coeffs():
        return [random.choice([-1, 0, 1]) for _ in range(n)]
    
    def compute_sum(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    best_overall_abs = best_abs
    best_overall_coeffs = None
    
    # Try to find coefficients achieving best_abs via random search + local search
    max_iterations = 2000000
    it = 0
    while it < max_iterations and time.time() < deadline:
        it += 1
        coeffs = random_coeffs()
        s = compute_sum(coeffs)
        
        if s == 0:
            continue
        
        abs_s = abs(s)
        
        # Local search: try single flips
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for new_val in [-1, 0, 1]:
                    if new_val == coeffs[i]:
                        continue
                    old_val = coeffs[i]
                    coeffs[i] = new_val
                    new_s = s + (new_val - old_val) * weights[i]
                    if new_s == 0:
                        coeffs[i] = old_val
                        continue
                    new_abs = abs(new_s)
                    if new_abs < abs_s:
                        s = new_s
                        abs_s = new_abs
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    break
        
        if abs_s < best_overall_abs:
            best_overall_abs = abs_s
            best_overall_coeffs = coeffs[:]
        
        # If we found |sum| = 1, that's the minimum possible
        if best_overall_abs == 1:
            break
    
    if best_overall_coeffs is not None:
        return best_overall_coeffs
    else:
        # Fallback: the seed solution
        return [0] * 28 + [1, -1]
# EVOLVE-BLOCK-END
