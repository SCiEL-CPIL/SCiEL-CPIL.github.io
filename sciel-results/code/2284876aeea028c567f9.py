"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product

def _compute_lcm_1_to_30():
    result = 1
    for i in range(1, 31):
        result = result * i // gcd(result, i)
    return result

_LCM = _compute_lcm_1_to_30()

def _weights():
    # weight[i] corresponds to term (i+1) in 1..30, scaled by LCM
    # coefficient[i-1] multiplies 1/i, so scaled contribution is coeff[i-1] * LCM // i
    return [_LCM // i for i in range(1, 31)]

def _meet_in_middle(deadline):
    """
    Meet-in-the-middle: split 30 coefficients into two halves of 15.
    For each half, enumerate all 3^15 = 14348909 combinations... that's too many.
    
    Wait, 3^15 = 14348907. That's about 14 million. With numpy we can do this.
    Actually, let's split into 10 and 20? No, 3^15 is manageable with numpy.
    
    Actually 3^15 = 14,348,907. That's a lot for numpy arrays but doable.
    Let me think of a better split: 15 and 15.
    
    For each half, compute all possible sums. Then for each sum in first half,
    find the closest sum in second half that makes total close to 0.
    """
    w = _weights()
    n = 30
    half = 15
    
    # First half: indices 0..14, weights w[0]..w[14]
    # Second half: indices 15..29, weights w[15]..w[29]
    
    # Generate all 3^15 combinations for first half
    # Use iterative approach to avoid memory issues
    import numpy as np
    
    # We'll generate all sums for the first half
    # 3^15 = 14348907 - this is large but numpy can handle it
    # Actually, let's be more careful. 14M integers is about 112MB. Should be fine.
    
    # Generate all coefficient combinations for first half
    # Instead of generating all at once, let's do it iteratively
    
    # Actually, let me reconsider. 3^15 is 14.3M. Generating all of them as a numpy array
    # of shape (14348907, 15) would be 14348907 * 15 * 8 bytes = ~1.7GB. Too much.
    
    # Better approach: generate sums directly without storing all coefficient vectors.
    # We just need the sum for each combination.
    
    # Let's use a different strategy: split into 10 and 20.
    # 3^10 = 59049, 3^20 = 3,486,784,401 - too big for second half.
    
    # Split into 12 and 18? 3^12 = 531441, 3^18 = 387,420,489. Still big.
    
    # Let's try a different approach: use the fact that we want sum close to 0.
    # Use a greedy + local search approach, but smarter.
    
    # Actually, let me try the meet-in-the-middle with 15/15 but only store the sums.
    # Generate all 3^15 sums for first half (14.3M values), sort them.
    # Then for each of 3^15 sums in second half, binary search in first half sums.
    
    # 14.3M integers is about 115MB. That's manageable.
    
    # Generate first half sums
    w1 = w[:half]
    w2 = w[half:]
    
    # Generate all 3^half coefficient combinations for first half and compute sums
    # Use iterative method
    n1 = half
    total1 = 3 ** n1
    
    # Build sums array
    sums1 = np.empty(total1, dtype=np.int64)
    
    # Iterative generation
    # For index i (0-based), the coefficient is (i // 3^(n1-1-j)) % 3 - 1 for position j
    # This is complex. Let's use a recursive/iterative approach.
    
    # Alternative: use numpy broadcasting to generate all combinations
    # This would require 3^15 * 15 array which is too big.
    
    # Let's do it incrementally:
    # Start with sums = [0]
    # For each weight, new_sums = old_sums * [1, 0, -1] expanded
    
    # This uses O(3^k) memory at step k, which peaks at 3^15.
    
    current_sums = np.array([0], dtype=np.int64)
    for j in range(n1):
        wj = w1[j]
        # For each existing sum, create three new sums: sum + wj, sum, sum - wj
        new_sums = np.empty(current_sums.size * 3, dtype=np.int64)
        new_sums[0::3] = current_sums + wj
        new_sums[1::3] = current_sums
        new_sums[2::3] = current_sums - wj
        current_sums = new_sums
        if time.time() > deadline:
            break
    
    # Now current_sums has all 3^15 possible sums for first half
    # Sort it for binary search
    current_sums.sort()
    
    # Now do the same for second half, but for each sum, binary search in current_sums
    # to find the closest value that makes total close to 0.
    
    # Actually, we need to track which coefficient combination gives which sum,
    # because we need to return the coefficients, not just the sum.
    
    # This is a problem. We need to recover the coefficients.
    
    # Alternative approach: use a different strategy.
    # Since we just need to minimize |sum|, and the sum is an integer,
    # the minimum possible non-zero absolute value is 1 (if achievable).
    
    # Let's try to find if sum = 1 or sum = -1 is achievable.
    # If not, try 2, -2, etc.
    
    # But checking achievability is still NP-hard in general.
    
    # Let me try a different approach: use local search with multiple restarts
    # and a better initialization.
    
    return None  # Fallback to local search

def _local_search(deadline):
    """Local search with multiple restarts."""
    import random
    import numpy as np
    
    w = _weights()
    n = 30
    best_val = float('inf')
    best_coeffs = None
    
    # Greedy initialization
    def _greedy_init():
        coeffs = [0] * n
        current_sum = 0
        # Try to add terms in order of largest weight first
        order = sorted(range(n), key=lambda i: -w[i])
        for idx in order:
            # Try -1, 0, 1 and pick the one that minimizes |current_sum + coeff * w[idx]|
            best_c = 0
            best_abs = abs(current_sum)
            for c in [-1, 1]:
                new_abs = abs(current_sum + c * w[idx])
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_c = c
            coeffs[idx] = best_c
            current_sum += best_c * w[idx]
        return coeffs, current_sum
    
    def _evaluate(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            s += c * w[i]
        return s
    
    def _local_search_once(coeffs, deadline):
        nonlocal best_val, best_coeffs
        current_sum = _evaluate(coeffs)
        current_abs = abs(current_sum)
        
        if current_abs == 0:
            # Need to perturb to get non-zero
            # Flip one coefficient
            for i in range(n):
                old = coeffs[i]
                for new in [-1, 0, 1]:
                    if new == old:
                        continue
                    coeffs[i] = new
                    new_sum = current_sum + (new - old) * w[i]
                    if new_sum != 0 and abs(new_sum) < current_abs:
                        current_sum = new_sum
                        current_abs = abs(new_sum)
                        break
                if current_abs < best_val:
                    best_val = current_abs
                    best_coeffs = list(coeffs)
                coeffs[i] = old
            return coeffs
        
        if current_abs < best_val:
            best_val = current_abs
            best_coeffs = list(coeffs)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                old = coeffs[i]
                for new in [-1, 0, 1]:
                    if new == old:
                        continue
                    new_sum = current_sum + (new - old) * w[i]
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < current_abs:
                        coeffs[i] = new
                        current_sum = new_sum
                        current_abs = new_abs
                        improved = True
                        if current_abs < best_val:
                            best_val = current_abs
                            best_coeffs = list(coeffs)
                        break  # Try next i
        return coeffs
    
    # Multiple restarts
    random.seed(42)
    num_restarts = 50
    restart_deadline = deadline - 5  # Leave 5 seconds margin
    
    # First, try greedy init
    coeffs, _ = _greedy_init()
    _local_search_once(coeffs, restart_deadline)
    
    # Then random restarts
    for r in range(num_restarts):
        if time.time() > restart_deadline:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        _local_search_once(coeffs, restart_deadline)
    
    if best_coeffs is None:
        # Fallback: simple non-zero solution
        return [0] * 28 + [1, -1]
    
    return best_coeffs

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline, 5 second safety margin
    
    # Try local search
    result = _local_search(deadline)
    
    # Verify the result is valid
    w = _weights()
    s = sum(c * w[i] for i, c in enumerate(result))
    if s == 0:
        # Invalid, use fallback
        result = [0] * 28 + [1, -1]
    
    return result
# EVOLVE-BLOCK-END
