"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that sum(c[i]/(i+1)) is nonzero
    and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 30
    # Precompute denominators
    denominators = [i + 1 for i in range(n)]
    
    # We want to minimize |sum(c[i] / (i+1))| subject to sum != 0
    # This is equivalent to finding a subset (with signs) that sums close to 0 but not exactly 0.
    
    # Strategy: Local search / Hill Climbing
    # 1. Start with a random configuration
    # 2. Iteratively flip a coefficient or change it to 0/1/-1 to reduce absolute sum
    # 3. If stuck at 0, try to move away to the smallest non-zero value? 
    #    Actually, if we hit 0, we must move to the closest non-zero.
    #    But hitting 0 is "good" for minimization, but invalid for the final answer.
    #    So we want to find the smallest non-zero sum achievable.
    
    best_val = None
    best_coeffs = None
    best_abs = float('inf')
    
    # Helper to calculate sum
    def calc_sum(coeffs):
        s = Fraction(0, 1)
        for c, d in zip(coeffs, denominators):
            if c != 0:
                s += Fraction(c, d)
        return s
    
    # Initialize with a random guess
    # A good start might be alternating or random
    current = [random.choice([-1, 0, 1]) for _ in range(n)]
    
    # Ensure it's not zero initially? Doesn't matter, we'll handle it.
    curr_sum = calc_sum(current)
    
    # If initial is 0, we need a non-zero start for local search to make sense
    # Or just proceed, if curr_sum is 0, any move is an improvement in "distance from 0" 
    # but we want the *smallest* non-zero.
    # Actually, let's define the objective:
    # We want to minimize |S| for S != 0.
    # If S == 0, we are in a "trap". We should move to the nearest non-zero.
    
    def objective(coeffs):
        s = calc_sum(coeffs)
        if s == 0:
            return float('inf') # Or a large number, but we need to distinguish
        return abs(s)
    
    # Local Search
    # We will try to minimize the absolute value.
    # If we reach 0, we need to accept the smallest non-zero neighbor.
    
    # Let's keep track of the best non-zero solution found so far.
    
    # To speed up, we can use integer arithmetic with LCM?
    # LCM of 1..30 is huge (2329089562800).
    # Fractions are fine for 30 terms.
    
    # Improvement: Use a list of (value, coeff) pairs? No, just recompute.
    # Optimization: Keep a running sum.
    
    def get_sum(coeffs):
        s = Fraction(0, 1)
        for i in range(n):
            if coeffs[i] != 0:
                s += Fraction(coeffs[i], denominators[i])
        return s
    
    # Start with a few random restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Bias towards 0? Or uniform?
        # Uniform is fine.
        current = [random.choice([-1, 0, 1]) for _ in range(n)]
        curr_sum = get_sum(current)
        
        # If current sum is 0, we need to find a non-zero neighbor.
        # If current sum is non-zero, we try to reduce |curr_sum|.
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
            
            # Try flipping/changing each coefficient
            for i in range(n):
                if time.time() > deadline:
                    break
                
                old_val = current[i]
                # Possible new values: {-1, 0, 1} excluding old_val
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    # Calculate new sum efficiently
                    # delta = (new_val - old_val) / (i+1)
                    delta = Fraction(new_val - old_val, denominators[i])
                    new_sum = curr_sum + delta
                    
                    if new_sum == 0:
                        # We found a zero sum. This is not a valid final answer,
                        # but it's a good "center". 
                        # We should record the neighbors of this zero sum as candidates?
                        # Or just accept that we hit 0, and then we MUST move to a non-zero.
                        # For now, let's say this is "good" for search but invalid for return.
                        # We'll handle the "best non-zero" logic separately.
                        # Actually, if we hit 0, the next best is the minimum |delta| from 0.
                        # Let's just mark this as a "zero hit" and break out to find neighbors?
                        # Simpler: If new_sum == 0, we don't accept it as the *current* state 
                        # if we are tracking best_non_zero. 
                        # But for local search dynamics, moving to 0 is good.
                        # Let's allow moving to 0, but keep track of best_non_zero.
                        
                        # If we move to 0, we update current to this state.
                        # Then in the next iteration, curr_sum will be 0.
                        # We need to be careful with objective function.
                        
                        # Let's update current to this zero state.
                        current[i] = new_val
                        curr_sum = new_sum
                        improved = True
                        # We found a zero sum! This is the global minimum of |sum|.
                        # The answer must be the smallest non-zero sum reachable.
                        # We can break out of the inner loop and do a "neighborhood scan" from 0.
                        # Or just continue, the next step will move away from 0.
                        break
                    else:
                        new_abs = abs(new_sum)
                        # Accept if it reduces absolute value
                        # Special case: if curr_sum is 0, any non-zero is an improvement 
                        # over "infinity" (if we treated 0 as invalid).
                        # But we want the *smallest* non-zero.
                        # So if curr_sum == 0, we should accept the new_sum with the smallest |new_sum|?
                        # No, local search usually accepts any improvement.
                        # If curr_sum == 0, all non-zero neighbors are "improvements" in the sense that they are valid.
                        # But we want the *best* (smallest) one.
                        # So if curr_sum == 0, we should pick the neighbor with the smallest |new_sum|.
                        
                        if curr_sum == 0:
                            # We are at 0. We want to move to the smallest non-zero neighbor.
                            # We should check all neighbors and pick the best one.
                            # For now, just accept if it's smaller than the best found so far?
                            # Or just accept any non-zero?
                            # Let's accept any non-zero, but we track the global best.
                            if new_abs < best_abs:
                                best_abs = new_abs
                                best_val = new_sum
                                best_coeffs = current[:]
                                current[i] = new_val
                                curr_sum = new_sum
                                improved = True
                                break
                        else:
                            # Normal case: curr_sum != 0
                            if new_abs < curr_abs:
                                # Update global best if this is better
                                if new_abs < best_abs:
                                    best_abs = new_abs
                                    best_val = new_sum
                                    best_coeffs = current[:]
                                
                                current[i] = new_val
                                curr_sum = new_sum
                                improved = True
                                break
                # End for new_val
            # End for i
        # End while improved
        
        # After local search converges (or times out), if we are at 0, we need to find the best non-zero neighbor.
        if curr_sum == 0:
            # Scan all neighbors of this zero state
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = current[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = Fraction(new_val - old_val, denominators[i])
                    new_sum = delta # since curr_sum is 0
                    if new_sum != 0:
                        new_abs = abs(new_sum)
                        if new_abs < best_abs:
                            best_abs = new_abs
                            best_val = new_sum
                            # Reconstruct coeffs for this best
                            temp_coeffs = current[:]
                            temp_coeffs[i] = new_val
                            best_coeffs = temp_coeffs
        else:
            # We are at a non-zero local minimum
            curr_abs = abs(curr_sum)
            if curr_abs < best_abs:
                best_abs = curr_abs
                best_val = curr_sum
                best_coeffs = current[:]
        
        # If we found a very small value, we might stop early?
        # The theoretical minimum is likely 1/LCM or similar.
        # 1/2329089562800 is extremely small.
        # We can check if best_abs <= 1/2329089562800.
        # If so, we found the global minimum (assuming 1 is the smallest numerator).
        # Actually, the sum is P/Q. Min |P/Q|. Q divides LCM.
        # So min |P|/Q. P is integer. Min non-zero |P| is 1.
        # So min value is 1/LCM? Not necessarily, because P is constrained by the coefficients.
        # But 1/LCM is a lower bound.
        
        # Let's check if we hit the lower bound.
        # LCM of 1..30
        import math
        lcm_val = 1
        for i in range(1, 31):
            lcm_val = lcm_val * i // math.gcd(lcm_val, i)
        
        if best_val is not None and best_abs <= Fraction(1, lcm_val):
            break

    if best_coeffs is None:
        # Fallback: return a valid non-zero solution
        # e.g. [1, 0, ..., 0] -> 1/1 = 1
        # Or [0, 1, 0, ..., 0] -> 1/2
        # Let's try to find a small one quickly
        # Just return [1, -1, 0, ...] -> 1 - 1/2 = 1/2
        return [1, -1] + [0] * 28

    return best_coeffs
# EVOLVE-BLOCK-END
