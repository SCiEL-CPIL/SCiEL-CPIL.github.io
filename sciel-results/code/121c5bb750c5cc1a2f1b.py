"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s hard limit
    
    # LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    
    # Precompute weights: L / i for i = 1..30
    weights = [L // i for i in range(1, 31)]
    
    # Split into two halves of 15
    half = 15
    left_weights = weights[:half]      # L/1, L/2, ..., L/15
    right_weights = weights[half:]     # L/16, L/17, ..., L/30
    
    def enumerate_states(weights_half):
        """Enumerate all 3^n states, return dict: sum -> list of coefficient tuples"""
        n = len(weights_half)
        states = {}
        
        # Iterative enumeration
        # Start with one state: all zeros, sum=0
        # Each state is (sum_value, coeff_tuple)
        # We'll build iteratively
        
        # For efficiency, use a list of (sum, coeffs) and extend
        # But 3^15 = 14348907 which is large. Let's use a different approach.
        
        # Actually, let's just do it directly with a dictionary mapping sum to list of coeffs
        # To save memory, we can store just the best (smallest absolute sum won't help here)
        # We need all states because we're looking for minimal nonzero difference.
        
        # Let's use a list-based approach
        # State: (current_sum, coeffs_so_far)
        # Build up iteratively
        
        current_states = [(0, ())]
        for w in weights_half:
            new_states = []
            for s, coeffs in current_states:
                # Option 1: coefficient = -1
                new_states.append((s - w, coeffs + (-1,)))
                # Option 2: coefficient = 0
                new_states.append((s, coeffs + (0,)))
                # Option 3: coefficient = 1
                new_states.append((s + w, coeffs + (1,)))
            current_states = new_states
        
        return current_states
    
    # Enumerate left half states
    left_states = enumerate_states(left_weights)
    
    # Check time
    if time.time() > deadline:
        # Fallback
        return [0]*28 + [1, -1]
    
    # For the right half, we need to find for each left state the right state
    # that minimizes |left_sum - right_sum| where left_sum - right_sum != 0
    
    # Strategy: enumerate right states, sort by sum value, then for each left state
    # do a binary search to find the closest right sum
    
    right_states = enumerate_states(right_weights)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Sort right states by their sum
    # right_states is a list of (sum, coeffs)
    right_sorted = sorted(right_states, key=lambda x: x[0])
    right_sums = [s for s, _ in right_sorted]
    
    import bisect
    
    best_diff = None
    best_left_coeffs = None
    best_right_coeffs = None
    
    # For each left state, find the right state that minimizes |left_sum - right_sum|
    for left_sum, left_coeffs in left_states:
        # We want right_sum such that left_sum - right_sum is small but nonzero
        # i.e., right_sum close to left_sum but not equal
        
        # Find position where left_sum would be inserted in right_sums
        pos = bisect.bisect_left(right_sums, left_sum)
        
        # Check candidates around pos
        candidates = []
        for p in [pos - 2, pos - 1, pos, pos + 1, pos + 2]:
            if 0 <= p < len(right_sums):
                candidates.append(p)
        
        for p in candidates:
            right_sum = right_sums[p]
            diff = left_sum - right_sum
            if diff != 0:
                if best_diff is None or abs(diff) < abs(best_diff):
                    best_diff = diff
                    best_left_coeffs = left_coeffs
                    best_right_coeffs = right_sorted[p][1]
    
    # Convert back to coefficients
    if best_left_coeffs is not None:
        coeffs = list(best_left_coeffs) + list(best_right_coeffs)
        return coeffs
    else:
        # Fallback
        return [0]*28 + [1, -1]
# EVOLVE-BLOCK-END
