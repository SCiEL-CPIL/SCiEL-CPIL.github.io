"""
Constructs a set of n=12 distinct positive integers with distinct subset sums,
minimizing the largest element.

The theoretical minimum for the largest element $a_n$ in a set of $n$ positive
integers with $2^n$ distinct subset sums is $2^{n-1}$. This is achieved by the
set of powers of 2: $\{1, 2, 4, \dots, 2^{n-1}\}$.

For n=12, the optimal set is [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048].
The largest element is 2048.
"""

import time

def _verify_distinct_subset_sums(vals):
    """
    Verify that all subset sums of vals are distinct.
    Returns True if distinct, False otherwise.
    """
    n = len(vals)
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        # Check for collisions
        if new_sums & sums:
            return False
        sums.update(new_sums)
    return len(sums) == (1 << n)

def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    """
    n = 12
    # The optimal solution is the powers of 2
    vals = [1 << i for i in range(n)]
    
    # Optional: Verify the solution (very fast for n=12)
    # if not _verify_distinct_subset_sums(vals):
    #     # Fallback to greedy if verification fails (should not happen)
    #     # Greedy also produces powers of 2
    #     pass
    
    return vals
