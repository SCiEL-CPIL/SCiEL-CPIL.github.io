import time
import os

N = 100

# Use monotonic clock for reliable elapsed-time measurement
_START = time.monotonic()
# Safety margin: the evaluator kills at 150s; we reserve 10s of slack
_DEADLINE = _START + 140.0

# Precompute the optimal solution once at import time
# The OneMax objective is maximized by all-ones; this is the unique global optimum.
_OPTIMAL = "1" * N


# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal 100-bit binary string for the OneMax objective.

    The objective counts the number of '1' characters in a binary string of
    length 100. The maximum achievable value is 100, attained uniquely by the
    string consisting entirely of '1's. This function returns that string
    directly, with a deadline guard to ensure safe termination within the
    evaluator's 150-second wall-clock limit.
    """
    # Deadline guard: if we are past the safety threshold, still return the
    # known optimal string (which is computed in O(N) and will not hang).
    if time.monotonic() > _DEADLINE:
        return _OPTIMAL

    return _OPTIMAL
# EVOLVE-BLOCK-END
