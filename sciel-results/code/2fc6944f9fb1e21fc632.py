"""
Optimized program for OneMax task.
Constructs a 100-bit binary string maximizing the number of ones.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0
SAFETY_MARGIN_SECONDS = 5.0

# Precompute the optimal solution once at module load time.
# This is the direct construction: all bits set to '1'.
# Using simple string multiplication for zero overhead and maximum portability.
_OPTIMAL = "1" * N


def construct():
    """
    Return a binary string of length N maximizing the number of ones.

    For the OneMax problem with N=100, the global optimum is the string
    of 100 ones. This function returns that precomputed string.

    The function executes in constant time with minimal overhead,
    directly returning the precomputed optimal solution while maintaining
    an explicit deadline guard as required by the task contract.
    """
    # Track start time for explicit deadline guard
    t0 = time.time()

    # The optimal solution for OneMax is trivially the all-ones string.
    # No search, optimization, or iterative refinement can improve upon this.
    # The precomputed constant is returned directly.
    result = _OPTIMAL

    # Deadline guard: ensure we return well before the 150 s kill limit.
    # Although this function is O(1), we maintain the guard structure
    # to satisfy the directive for an explicit deadline check.
    elapsed = time.time() - t0
    if elapsed + SAFETY_MARGIN_SECONDS >= DEADLINE_SECONDS:
        # We are too close to the deadline; return the optimal result now.
        return result

    return result
