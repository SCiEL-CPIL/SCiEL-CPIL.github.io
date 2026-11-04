import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try a Hanning window first
    # This minimizes sidelobes significantly compared to rectangular (ones)
    x = np.hanning(N)
    
    # If we need to ensure "signed" (both + and -), we could subtract the mean,
    # but that reduces the integral (denominator).
    # Keeping it positive maximizes the integral.
    # The term "signed" in the task description "signed step heights"
    # likely just means the heights can be any real number (positive or negative),
    # not that they must be +/- 1.
    
    # We can try to optimize the window slightly using a few iterations of local search
    # on the continuous variables to minimize the max sidelobe.
    # But Hanning is already very good.
    
    # Check time
    if time.time() > deadline:
        return x
        
    # Optional: Simple refinement
    # We can try to adjust the alpha of a Kaiser window or similar,
    # but Hanning is a fixed Kaiser.
    
    return x
