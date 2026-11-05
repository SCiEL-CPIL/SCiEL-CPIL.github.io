"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time

def _check_violation(elem, others):
    """Check if 1/elem equals sum of 1/x for some subset of others."""
    target = 1.0 / elem
    m = len(others)
    if m == 0:
        return False
    # Meet in the middle for efficiency when m > 10
    if m <= 10:
        for r in range(1, m + 1):
            for combo in combinations(others, r):
                total = sum(1.0 / x for x in combo)
                if abs(total - target) < 1e-10:
                    return True
    else:
        # Split others into two halves
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        
        # Generate all subset sums for left
        left_sums = [0.0]
        for x in left:
            inv = 1.0 / x
            left_sums = left_sums + [s + inv for s in left_sums]
        
        # Generate all subset sums for right
        right_sums = [0.0]
        for x in right:
            inv = 1.0 / x
            right_sums = right_sums + [s + inv for s in right_sums]
        
        # Sort right_sums for binary search
        right_sums_sorted = sorted(right_sums)
        
        for ls in left_sums:
            if ls == 0.0:
                continue
            # Check if there's an rs such that ls + rs == target
            # i.e., rs == target - ls
            needed = target - ls
            if needed < -1e-10:
                continue
            # Binary search in right_sums_sorted
            import bisect
            idx = bisect.bisect_left(right_sums_sorted, needed - 1e-10)
            if idx < len(right_sums_sorted):
                if abs(right_sums_sorted[idx] - needed) < 1e-10:
                    return True
            if idx + 1 < len(right_sums_sorted):
                if abs(right_sums_sorted[idx + 1] - needed) < 1e-10:
                    return True
    return False

def _is_valid(subset):
    s = list(subset)
    for i, elem in enumerate(s):
        others = [x for j, x in enumerate(s) if j != i]
        if _check_violation(elem, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Start with the seed: 13-24 (12 elements)
    current = list(range(13, 25))
    
    # Try to add elements 1-12 one by one
    for elem in range(1, 13):
        if time.time() > deadline:
            break
        candidate = current + [elem]
        if _is_valid(candidate):
            current = candidate
    
    # Try to add in different orders
    for order in [range(1, 13), range(12, 0, -1), [1,2,3,4,5,6,7,8,9,10,11,12]]:
        if time.time() > deadline:
            break
        trial = list(range(13, 25))
        for elem in order:
            if time.time() > deadline:
                break
            candidate = trial + [elem]
            if _is_valid(candidate):
                trial = candidate
        if len(trial) > len(current):
            current = trial
    
    return sorted(current)

# EVOLVE-BLOCK-END
