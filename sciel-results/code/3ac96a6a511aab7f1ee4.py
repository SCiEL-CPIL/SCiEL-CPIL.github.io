"""OneMax direct binary construction (100 bits).

Constructs a 100-bit binary string maximizing ones.
The global optimum is all ones, giving a score of 100.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns a string of N '1' characters, which is the unique global optimum
    for the OneMax objective with N bits.

    Includes an explicit deadline guard per task contract, though the
    computation is O(1) and completes in microseconds.
    """
    t0 = time.monotonic()
    # Safety margin: return well before the 150s wall-clock kill.
    deadline = t0 + 140.0

    # Direct construction of the optimal solution: all bits set to 1.
    # This is the unique maximizer of sum(x_i) over x_i in {0, 1}, x in {0,1}^N.
    optimal = '1' * N

    # Verify interface compliance: length must be exactly N.
    assert len(optimal) == N, f"Expected length {N}, got {len(optimal)}"

    # Verify all characters are valid binary digits.
    assert all(c in ('0', '1') for c in optimal), "Non-binary character in result"

    # Deadline guard: return the result if within time budget.
    # For this O(1) solution, the time check is always satisfied.
    if time.monotonic() < deadline:
        return optimal

    # Fallback (unreachable for O(1) work, but satisfies contract):
    # Return the best result found so far (which is already optimal).
    return optimal
# EVOLVE-BLOCK-END
