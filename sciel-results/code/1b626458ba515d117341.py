"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all '1's.
    Uses Python's arbitrary-precision integer arithmetic to construct
    the all-ones string via format(val, '0100b').
    """
    # Deadline guard: track elapsed time and return best result with safety margin
    start_time = time.time()
    deadline = start_time + 140  # 10-second safety margin before 150s limit
    
    # Direct construction of the known global optimum (all ones)
    # Using Python's big-integer type: (1 << N) - 1 gives N bits of 1
    # format(val, '0100b') converts to binary string with zero-padding
    try:
        val = (1 << N) - 1
        result = format(val, '0100b')
    except Exception:
        # Fallback to standard Python string multiplication
        result = "1" * N

    # Verify length constraint
    if len(result) != N:
        # Fallback: ensure correct length (should never happen)
        result = "1" * N
    
    # Check deadline (should never be triggered for this simple construction)
    if time.time() > deadline:
        # Return the best result found (which is the optimal solution)
        return result
    
    return result
# EVOLVE-BLOCK-END
