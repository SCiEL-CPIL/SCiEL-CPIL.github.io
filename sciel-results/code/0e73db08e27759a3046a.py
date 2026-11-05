"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    This function constructs the global optimum for the OneMax problem, which is
    a string of N '1's. It uses integer bit manipulation to generate the pattern
    and includes an explicit deadline guard to ensure compliance with time limits,
    although the computation is constant time.
    """
    start_time = time.time()
    # Set a safety margin before the 150s limit. 
    # The operation is O(1) or O(N) with N=100, so this is just a formality.
    deadline = start_time + 145.0 

    # Generate the all-ones integer mask: 2^N - 1
    # This is equivalent to having N bits set to 1.
    mask = (1 << N) - 1
    
    # Convert the integer to a binary string and remove the '0b' prefix
    binary_string = bin(mask)[2:]
    
    # Ensure the string length is exactly N.
    # For mask = 2^N - 1, len(bin(mask)[2:]) is exactly N, so padding/truncation
    # is not strictly necessary but ensures robustness.
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
        
    # Explicit deadline guard
    if time.time() > deadline:
        # If we somehow exceed the deadline (extremely unlikely), return the 
        # valid all-ones string directly to avoid timeout.
        return "1" * N
        
    return binary_string
# EVOLVE-BLOCK-END
