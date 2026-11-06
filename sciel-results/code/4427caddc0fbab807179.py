"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_superincreasing(seq):
    """Check if sequence is superincreasing (each element > sum of all previous)."""
    if not seq:
        return True
    s = 0
    for x in seq:
        if x <= s:
            return False
        s += x
    return True

def _all_subset_sums_distinct(seq):
    """Check that all 2^n subset sums are distinct.
    For positive integers, this is equivalent to the sequence being superincreasing."""
    n = len(seq)
    if n <= 1:
        return True
    # Superincreasing is necessary and sufficient for distinct subset sums with positive ints
    return _is_superincreasing(seq)

def _random_superincreasing(max_val, n):
    """Generate a random superincreasing sequence of length n with max element <= max_val."""
    # We need a1 < a2 < ... < an where ai > sum(a1..a_{i-1})
    # So a1 >= 1, a2 >= 2, a3 >= 4, a4 >= 8, ...
    # Minimum possible sequence is 1, 2, 4, 8, ..., 2^(n-1)
    # Max element for minimum is 2^(n-1)
    
    # We want to find sequences where the largest element is minimized.
    # Strategy: try to make elements as small as possible while maintaining superincreasing property.
    
    # Let's use a different approach: start with minimums and try to reduce the largest.
    # Actually, the minimum largest element for n=12 is 2^11 = 2048 (powers of 2).
    # But we can do better! The superincreasing condition is ai > sum(a1..a_{i-1}).
    # The tightest packing gives ai = sum(a1..a_{i-1}) + 1.
    # This gives 1, 2, 4, 8, ..., 2^(n-1). Largest = 2^(n-1).
    
    # Wait, can we do better? No, because if ai = sum(a1..a_{i-1}) + 1, that's the minimum.
    # And 1, 2, 4, ..., 2^(n-1) has largest 2^(n-1) = 2048 for n=12.
    
    # But wait - the problem says "distinct subset sums". For positive integers, 
    # distinct subset sums is equivalent to superincreasing.
    # The minimum possible largest element is indeed 2^(n-1) = 2048.
    
    # Let me verify: n=3, can we do better than 1,2,4 (largest=4)?
    # 1,2,3: subset sums: 0,1,2,3,1+2=3 (duplicate with 3!). No.
    # 1,3,4: 0,1,3,4,1+3=4 (dup). No.
    # 2,3,5: 0,2,3,5,2+3=5 (dup). No.
    # So 1,2,4 is optimal for n=3, largest=4=2^2.
    
    # For n=12, the answer should be 2^11 = 2048.
    # The seed program returns [1,2,4,8,...,2048] which has largest 2048.
    # So the seed is already optimal!
    
    # Let me double-check by trying to find a sequence with largest < 2048.
    # If largest < 2048, then we need 12 distinct positive ints all < 2048 that are superincreasing.
    # a1 >= 1, a2 >= 2, a3 >= 4, ..., a12 >= 2048. Contradiction.
    
    # So the minimum largest element is 2048, achieved by powers of 2.
    
    return [1 << i for i in range(n)]

def construct():
    """Construct exactly 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest integer. Returns the list of 12 integers."""
    n = 12
    # The optimal solution is powers of 2: 1, 2, 4, 8, ..., 2048
    # Largest = 2048 = 2^11
    result = [1 << i for i in range(n)]
    # Verify
    assert len(result) == n
    assert all(x > 0 for x in result)
    assert all(x <= 1048576 for x in result)
    assert len(set(result)) == n  # distinct
    assert _all_subset_sums_distinct(result)
    return result

# EVOLVE-BLOCK-END
