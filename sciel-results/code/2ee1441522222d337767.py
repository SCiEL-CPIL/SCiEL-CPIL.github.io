"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The optimal solution for OneMax is the string of all ones.
    # Construct using bytearray for a distinct mutable buffer path.
    # Initialize with ASCII '1' (49) repeated N times.
    buf = bytearray(b'1' * N)
    
    # Convert to string
    result = buf.decode('ascii')
    
    # Check if we are still within the deadline
    if time.time() < deadline:
        return result
    else:
        # Fallback (should never happen for N=100)
        return "1" * N
# EVOLVE-BLOCK-END
