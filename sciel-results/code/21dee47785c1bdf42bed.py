"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for the OneMax problem.
    The optimal solution is a string of all '1's, yielding a raw score of 100.
    """
    # Deadline guard (safety margin before 150s limit)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Construct using bytearray for mutable buffer allocation path
    # Initialize with ASCII '1' (49) repeated N times
    buf = bytearray([49] * N)
    
    # Safety check (will never trigger for O(N) construction)
    if time.time() > deadline:
        return buf.decode('ascii')
    
    return buf.decode('ascii')
# EVOLVE-BLOCK-END
