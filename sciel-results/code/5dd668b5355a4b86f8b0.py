"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    def is_valid(subset):
        """Check if all reciprocal subset sums of subset are distinct using integer scaling."""
        n = len(subset)
        if n == 0:
            return True
        # Compute LCM of all elements in subset
        L = 1
        for x in subset:
            L = lcm(L, x)
        if L > 10**18:  # Too large, fall back to Fraction check
            from fractions import Fraction
            sums = [Fraction(0)] * (1 << n)
            for mask in range(1, 1 << n):
                lsb = mask & (-mask)
                bit_idx = lsb.bit_length() - 1
                prev = mask ^ lsb
                sums[mask] = sums[prev] + Fraction(1, subset[bit_idx])
            return len(set(sums)) == (1 << n)
        
        # Scale: 1/x becomes L/x which is an integer
        weights = [L // x for x in subset]
        
        # Compute all subset sums using dynamic programming
        sums = [0] * (1 << n)
        for mask in range(1, 1 << n):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            sums[mask] = sums[prev] + weights[bit_idx]
        
        return len(set(sums)) == (1 << n)
    
    best = []
    
    # Try multiple greedy strategies with different orderings
    orderings = [
        list(range(1, 23)),       # ascending
        list(range(22, 0, -1)),   # descending
        sorted(range(1, 23), key=lambda x: 1/x),  # by reciprocal ascending
        sorted(range(1, 23), key=lambda x: -1/x), # by reciprocal descending
        # Prime-heavy sets tend to work well
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 9, 10, 14, 15, 16, 18, 20, 21, 22],
        # Mix of small and large
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
    ]
    
    # Add more random orderings
    random.seed(42)
    for _ in range(200):
        perm = list(range(1, 23))
        random.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = []
        for candidate in order:
            trial = result + [candidate]
            n = len(trial)
            if n <= 18:  # 2^18 = 262144, manageable with integers
                if is_valid(trial):
                    result = trial
        if len(result) > len(best):
            best = result
            if len(best) >= 14:
                # Found a good result, try to improve further
                pass
    
    if best:
        return best
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
