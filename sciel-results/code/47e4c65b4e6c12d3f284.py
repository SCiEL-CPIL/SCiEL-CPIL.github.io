"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def construct():
    deadline = time.time() + 110  # safety margin
    
    # Use golden ratio for Kronecker sequence
    alpha = (1 + math.sqrt(5)) / 2.0
    
    # Generate binary sequence based on fractional parts of i*alpha
    h = np.zeros(N)
    for i in range(N):
        frac = (i * alpha) % 1.0
        h[i] = 1.0 if frac < 0.5 else 0.0
    
    # Ensure exactly N/2 ones
    num_ones = int(np.sum(h))
    if num_ones < N // 2:
        # Need to flip some 0s to 1s
        zeros = np.where(h == 0)[0]
        needed = N // 2 - num_ones
        # Flip the zeros with largest fractional parts (closest to 0.5 from below)
        fracs = np.array([(i * alpha) % 1.0 for i in zeros])
        # We want to flip those with fracs closest to 0.5
        indices_to_flip = zeros[np.argsort(-fracs)[:needed]]
        h[indices_to_flip] = 1.0
    elif num_ones > N // 2:
        ones = np.where(h == 1)[0]
        excess = num_ones - N // 2
        fracs = np.array([(i * alpha) % 1.0 for i in ones])
        # Flip ones with fracs closest to 0.5 from above (i.e., largest fracs < 0.5... wait)
        # Actually flip ones with smallest fracs (farthest from 0.5)
        indices_to_flip = ones[np.argsort(fracs)[:excess]]
        h[indices_to_flip] = 0.0
    
    return h
# EVOLVE-BLOCK-END
