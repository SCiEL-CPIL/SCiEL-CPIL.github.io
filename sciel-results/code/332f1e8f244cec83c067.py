"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the global optimum for OneMax(100).
    The objective is to maximize the count of '1's in a 100-bit binary string.
    The unique global optimum is a string of 100 '1's, yielding a score of 100.
    Since this is a direct construction task, we return the known optimum.
    Uses bytearray initialization as the construction mechanism.
    """
    # Deadline guard: track start time and set a limit slightly below the 150s kill switch.
    # For this static construction, the time check is trivial but ensures compliance
    # with the directive to keep an explicit deadline guard.
    t0 = time.monotonic()
    deadline = t0 + 149.0  # Safety margin before 150s

    # Construct the optimal solution: 100 '1's
    # Using bytearray for mutable buffer allocation before decoding to string.
    # This is O(N) and extremely fast.
    buf = bytearray(b'1' * N)
    optimal_string = buf.decode('ascii')

    # Verify we are still within the time limit before returning.
    # In a real search, we would return the best found so far if time ran out.
    # Here, the construction is instantaneous, so this is just a formal guard.
    if time.monotonic() < deadline:
        return optimal_string
    
    # Fallback (should never be reached for this static construction)
    return optimal_string
# EVOLVE-BLOCK-END
