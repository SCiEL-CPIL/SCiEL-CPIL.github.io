"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 330s kill

def priority(vector, n):
    # Prioritize vectors with more 1s (central values in {0,1,2})
    # This heuristic tends to produce larger cap sets
    count_ones = sum(1 for v in vector if v == 1)
    # Also add a small tie-breaker based on sum of digits
    digit_sum = sum(vector)
    return count_ones * 10 + (n - abs(digit_sum - n))
# EVOLVE-BLOCK-END
