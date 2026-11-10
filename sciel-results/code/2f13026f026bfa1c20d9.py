"""Program for small nonzero signed harmonic subsums."""
import time
from fractions import Fraction
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    Uses optimized Meet-in-the-Middle with Two-Pointer technique.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit

    n = 30

    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)

    # weights[i] = L // (i+1) for coefficient index i (0-based), so term is c_i * L/(i+1)
    weights = [L // (i + 1) for i in range(n)]

    # Split into two halves: first 15 and last 15
    half = n // 2  # 15
    left_weights = weights[:half]
    right_weights = weights[half:]

    # Enumerate all 3^15 sums for the left half
    left_sums = [0]
    for w in left_weights:
        new_sums = []
        for s in left_sums:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        left_sums = new_sums

    # Enumerate all 3^15 sums for the right half
    right_sums = [0]
    for w in right_weights:
        new_sums = []
        for s in right_sums:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        right_sums = new_sums

    # Sort both lists
    left_sums.sort()
    right_sums.sort()

    # Use Two-Pointer technique to find min nonzero |l + r|
    # l is from left_sums, r is from right_sums
    # We want l + r close to 0 but not 0.
    i = 0
    j = len(right_sums) - 1
    best_int_sum = None
    best_left_sum = None
    best_right_sum = None
    
    # Optional: Track if we found a 0 sum, though not required for result
    # Just finding min nonzero is enough.
    
    while i < len(left_sums) and j >= 0:
        if time.time() > deadline:
            break
            
        current_sum = left_sums[i] + right_sums[j]
        
        if current_sum != 0:
            abs_sum = abs(current_sum)
            if best_int_sum is None or abs_sum < best_int_sum:
                best_int_sum = abs_sum
                best_left_sum = left_sums[i]
                best_right_sum = right_sums[j]
                
                # If we found 1, it's the minimum possible nonzero integer value
                # (since weights are integers, sum is integer, min nonzero is 1)
                # Note: The actual harmonic sum is 1/L.
                # If we find 1, we can stop?
                # Yes, 1 is the smallest positive integer.
                # But we need to be sure we found a valid combo.
                # Since we are iterating, if we find 1, it's the global min.
                if best_int_sum == 1:
                    break
            
        if current_sum > 0:
            # Sum is too positive, decrease it by reducing j (right sum is sorted asc)
            # Wait, if right_sums is sorted asc, right_sums[j] is the largest.
            # Decreasing j decreases right_sums[j], thus decreases total sum.
            j -= 1
        else:
            # Sum is too negative (or zero, but we handled zero separately? No, if 0 we skip update)
            # If current_sum < 0, increase i to increase left_sums[i]
            i += 1
            
    if best_int_sum is None:
        # Fallback if no nonzero sum found (should not happen for N=30)
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    def reconstruct(weights, target):
        """Backtracking to find coefficients in {-1,0,1} summing to target."""
        coeffs = [0] * len(weights)
        
        def bt(idx, cur_sum):
            if idx == len(weights):
                return cur_sum == target
            w = weights[idx]
            # Pruning: if remaining weights cannot bridge the gap, return False
            # max_remaining = sum(abs(weights[k]) for k in range(idx, len(weights)))
            # if abs(target - cur_sum) > max_remaining: return False
            # But calculating max_remaining every step is slow. Precompute suffix sums?
            # For 15 depth, simple DFS is likely fast enough.
            
            for val in (0, 1, -1):
                coeffs[idx] = val
                if bt(idx + 1, cur_sum + val * w):
                    return True
            coeffs[idx] = 0
            return False
        
        if bt(0, 0):
            return coeffs
        return None

    left_coeffs = reconstruct(left_weights, best_left_sum)
    right_coeffs = reconstruct(right_weights, best_right_sum)

    if left_coeffs is None or right_coeffs is None:
        # Fallback to seed-like behavior if reconstruction fails
        # This should rarely happen
        return [0] * 28 + [1, -1]

    result = left_coeffs + right_coeffs

    # Verify
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(result) if c != 0)
    if total_frac == 0:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]

    return result

# EVOLVE-BLOCK-END
