import time
import math
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums_with_history(weights):
        """Generate all possible sums and their coefficient assignments."""
        # sums_dict maps sum_value -> list of coefficient tuples (one per path)
        # To save memory, we store only one path per sum
        sums_dict = {0: [0] * len(weights)}
        for idx, wi in enumerate(weights):
            new_sums = {}
            for s, coeffs in sums_dict.items():
                # +wi
                new_s = s + wi
                new_coeffs = coeffs[:idx] + [1] + coeffs[idx+1:]
                if new_s not in new_sums:
                    new_sums[new_s] = new_coeffs
                # -wi
                new_s = s - wi
                new_coeffs = coeffs[:idx] + [-1] + coeffs[idx+1:]
                if new_s not in new_sums:
                    new_sums[new_s] = new_coeffs
                # 0
                if s not in new_sums:
                    new_sums[s] = coeffs
            sums_dict = new_sums
        return sums_dict
    
    def generate_sums_simple(weights):
        """Generate just the set of possible sums (faster, less memory)."""
        sums = {0}
        for wi in weights:
            new_sums = set()
            for s in sums:
                new_sums.add(s + wi)
                new_sums.add(s)
                new_sums.add(s - wi)
            sums = new_sums
        return sums
    
    # For the first half, we need to be able to reconstruct coefficients
    # For the second half, we just need the sums for the two-pointer search
    
    sums1_dict = generate_sums_with_history(w1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums2_set = generate_sums_simple(w2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sorted_sums1 = sorted(sums1_dict.keys())
    sorted_sums2 = sorted(sums2_set)
    
    # Two-pointer to find pair closest to 0 (excluding 0)
    i = 0
    j = len(sorted_sums2) - 1
    best_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    while i < len(sorted_sums1) and j >= 0:
        if time.time() > deadline - 5:
            break
        total = sorted_sums1[i] + sorted_sums2[j]
        if total != 0 and abs(total) < best_val:
            best_val = abs(total)
            best_s1 = sorted_sums1[i]
            best_s2 = sorted_sums2[j]
        if total < 0:
            i += 1
        else:
            j -= 1
    
    if time.time() > deadline or best_s1 is None:
        return [0] * 28 + [1, -1]
    
    # Get coefficients for best_s1
    coeffs1 = sums1_dict[best_s1]
    
    # Now we need to find coefficients for best_s2 in the second half
    # Use backtracking
    def backtrack_coeffs(weights, target_sum):
        """Backtrack to find coefficients that achieve target_sum."""
        coeffs = [0] * len(weights)
        # Build history of achievable sums as sets for O(1) lookup
        history = [set([0])]
        curr = {0}
        for wi in weights:
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            history.append(curr)
        
        current_sum = target_sum
        for i in range(len(weights) - 1, -1, -1):
            wi = weights[i]
            if current_sum - wi in history[i]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in history[i]:
                coeffs[i] = 0
            elif current_sum + wi in history[i]:
                coeffs[i] = -1
                current_sum += wi
            else:
                coeffs[i] = 0
        return coeffs
    
    coeffs2 = backtrack_coeffs(w2, best_s2)
    
    result = coeffs1 + coeffs2
    
    # Verify the result
    total = Fraction(0, 1)
    for i, c in enumerate(result):
        if c != 0:
            total += Fraction(c, i + 1)
    
    if total == 0:
        # Fallback
        return [0] * 28 + [1, -1]
    
    return result
