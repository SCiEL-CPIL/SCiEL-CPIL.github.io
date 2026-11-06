"""OneMax direct binary construction (100 bits).

Constructs the optimal 100-bit string of all ones. The score is number of ones,
so the maximum possible value is 100. This implementation returns the optimal
string using a simple, deterministic construction path.
"""
import time

N = 100
_DEADLINE_SECONDS = 145.0  # safety margin before the 150s kill
_T_START = time.monotonic()

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (optimal for OneMax).

    The OneMax objective is the count of '1' characters in a binary string.
    For a fixed length of 100, the maximum is 100, achieved uniquely by the
    string of 100 '1's. This function returns that optimal string directly.
    """
    # Deadline guard: if we somehow exceed the budget, still return the
    # known optimum (construction is O(N) and completes in microseconds).
    if time.monotonic() - _T_START > _DEADLINE_SECONDS:
        return "1" * N

    # Direct optimal construction: a string of N '1' characters.
    return "1" * N
# EVOLVE-BLOCK-END
