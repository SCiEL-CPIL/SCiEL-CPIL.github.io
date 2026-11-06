"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard to ensure we finish well within the 150s limit
    t0 = time.perf_counter()
    limit = t0 + 145.0
    
    # Strategy: Represent the optimal 100-bit string as a Python bytearray
    # initialized with 100 ASCII '1' characters, then decode it to a string.
    # This leverages mutable buffer allocation as a distinct construction path.
    
    # Create a bytearray of size N filled with the ASCII code for '1' (49)
    # bytearray(b'1' * N) is efficient.
    # Alternatively, bytearray(N) creates null bytes, so we need to fill it.
    # bytearray(b'1' * N) is the most direct way to initialize with '1's.
    ba = bytearray(b'1' * N)
    
    # Decode to string
    optimal_str = ba.decode('ascii')
    
    # Safety check
    if time.perf_counter() > limit:
        return optimal_str
        
    return optimal_str
# EVOLVE-BLOCK-END
