"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE = 150  # Seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction: OneMax on 100 bits.
    The optimal solution is a string of 100 '1's.
    Includes a deadline guard as per directive.
    """
    # Check time elapsed
    if time.time() - START_TIME > DEADLINE - 10:
        # Return best known result (which is optimal)
        return "1" * N
    
    # Construct optimal solution
    return "1" * N
# EVOLVE-BLOCK-END
