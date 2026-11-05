"""Optimal construction for distinct subset sums with minimal largest element.

Theoretical lower bound: for n positive integers with all 2^n subset sums
distinct, sorting a_1 < ... < a_n forces a_i > sum_{j<i} a_j, hence
a_i >= 2^{i-1}, so a_n >= 2^{n-1}. Powers of two attain this bound, giving
the provably optimal largest element 2^{n-1} = 2048 for n = 12.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset
    sums are distinct, minimizing the largest integer.
    
    The optimal set is the powers of 2: [1, 2, 4, ..., 2048].
    This is the unique set achieving the minimum possible maximum element.
    """
    n = 12
    # Provably optimal: powers of two. Largest = 2**(n-1) = 2048.
    result = [1 << i for i in range(n)]
    
    # Fast correctness verification (4096 subset sums, trivial cost).
    # We verify that all subset sums are distinct.
    seen = set()
    seen.add(0)
    for a in result:
        new_sums = {s + a for s in seen}
        # For powers of 2, new_sums and seen are disjoint.
        # If they intersect, it's an error in construction (impossible here).
        if new_sums & seen:
            # This should never happen for powers of two.
            pass
        seen.update(new_sums)
        
    # All 4096 sums must be distinct
    assert len(seen) == (1 << n), "Subset sums are not distinct"
    
    # Verify distinctness of the integers themselves
    assert len(set(result)) == n, "Integers are not distinct"
    
    # Verify all are positive and <= 1048576
    assert all(x > 0 for x in result), "Integers must be positive"
    assert all(x <= 1048576 for x in result), "Integers exceed bound"

    return result
# EVOLVE-BLOCK-END
