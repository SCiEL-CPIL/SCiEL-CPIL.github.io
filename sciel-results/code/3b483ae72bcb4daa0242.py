"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    def is_valid(subset):
        """Check if all reciprocal subset sums of subset are distinct."""
        n = len(subset)
        if n == 0:
            return True
        sums = [Fraction(0)] * (1 << n)
        for mask in range(1, 1 << n):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            sums[mask] = sums[prev] + Fraction(1, subset[bit_idx])
        return len(set(sums)) == (1 << n)
    
    best = []
    
    # Try multiple greedy strategies with different orderings
    orderings = [
        list(range(1, 23)),       # ascending
        list(range(22, 0, -1)),   # descending
        sorted(range(1, 23), key=lambda x: 1/x),  # by reciprocal ascending
        sorted(range(1, 23), key=lambda x: -1/x), # by reciprocal descending
    ]
    
    # Add some random orderings
    random.seed(42)
    for _ in range(50):
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
            if n <= 15:  # 2^15 = 32768, still manageable with Fraction
                if is_valid(trial):
                    result = trial
        if len(result) > len(best):
            best = result
            if len(best) >= 12:  # Good result, keep searching but don't spend too long
                pass
    
    if best:
        return best
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
