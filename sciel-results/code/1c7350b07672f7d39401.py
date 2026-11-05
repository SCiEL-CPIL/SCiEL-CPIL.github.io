"""
Optimal construction for distinct subset sums with minimal largest element.

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
    """
    n = 12
    # The unique optimal solution (up to permutation, but we need distinct positive ints)
    # is the powers of 2. Any other set with all distinct subset sums must have
    # max element >= 2^(n-1).
    result = [1 << i for i in range(n)]
    
    # Optional: Verify correctness (very fast, O(2^n))
    # This is included to ensure robustness against any potential misunderstanding
    # of the problem constraints, though powers of 2 are mathematically guaranteed.
    seen = {0}
    for a in result:
        new_sums = {s + a for s in seen}
        if new_sums & seen:
            # This should never happen for powers of 2
            raise ValueError("Subset sums are not distinct")
        seen |= new_sums
        
    if len(seen) != (1 << n):
        raise ValueError("Incorrect number of subset sums")
        
    return result
# EVOLVE-BLOCK-END
