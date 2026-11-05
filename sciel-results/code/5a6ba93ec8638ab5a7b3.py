"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective with no constraints has a trivial optimum:
    all bits set to 1. This returns that string directly.
    
    Implementation detail: We model the construction as a bitwise OR between
    a zero-initialized integer and a mask of 2^N - 1, leveraging Python's
    arbitrary-precision integer arithmetic. The result is converted to a
    binary string via bin(), ensuring the output is strictly derived from
    the integer representation.
    """
    # Deadline guard for safety (150s limit, use 145s as hard stop)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Construct the all-ones state using integer arithmetic
    # Zero-initialized integer
    val = 0
    
    # Mask of 2^N - 1 (all N bits set to 1)
    mask = (1 << N) - 1
    
    # Bitwise OR to produce the all-ones integer
    optimal_int = val | mask
    
    # Convert to binary string, stripping the '0b' prefix
    result_str = bin(optimal_int)[2:]
    
    # Safety check: ensure length is correct
    # If for some reason the string is shorter (should not happen for N=100), pad
    if len(result_str) < N:
        result_str = result_str.zfill(N)
        
    # Final deadline check (should be instant)
    if time.time() > deadline:
        # Fallback to direct string construction if somehow we timed out
        return "1" * N
        
    return result_str
# EVOLVE-BLOCK-END
