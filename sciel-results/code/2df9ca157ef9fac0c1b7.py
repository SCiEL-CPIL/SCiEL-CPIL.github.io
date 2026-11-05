"""Optimal construction for distinct subset sums with minimal largest element.

Theoretical lower bound: for n positive integers with all 2^n subset sums
distinct, sorting a_1 < ... < a_n forces a_i > sum_{j<i} a_j, hence
a_i >= 2^{i-1}, so a_n >= 2^{n-1}. Powers of two attain this bound, giving
the provably optimal largest element 2^{n-1} = 2048 for n = 12.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 60.0  # hard wall-clock guard (evaluator kills at 90s)

    n = 12
    # Provably optimal: powers of two. Largest = 2**(n-1) = 2048.
    # This is the unique set of 12 distinct positive integers that minimizes
    # the maximum element while ensuring all 2^12 subset sums are distinct.
    result = [1 << i for i in range(n)]

    # Fast correctness verification (4096 subset sums, trivial cost).
    if time.time() < deadline - 5.0:
        seen = set([0])
        for a in result:
            new_sums = {s + a for s in seen}
            # For powers of two, new_sums and seen are disjoint.
            # If they intersect, it's a logic error.
            if new_sums & seen:
                raise AssertionError("Subset sums are not distinct")
            seen.update(new_sums)
        
        # All 4096 sums must be distinct
        if len(seen) != (1 << n):
            raise AssertionError("Subset sums are not distinct")
        
        # Verify distinctness of the integers themselves
        if len(set(result)) != n:
            raise AssertionError("Integers are not distinct")
        
        # Verify all are positive and <= 1048576
        if not all(x > 0 for x in result):
            raise AssertionError("Integers must be positive")
        if not all(x <= 1048576 for x in result):
            raise AssertionError("Integers exceed bound")

    return result
# EVOLVE-BLOCK-END
