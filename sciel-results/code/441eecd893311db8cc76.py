"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    # Use a sum of multiple harmonics with random phases to flatten the autocorrelation.
    # This minimizes the maximum overlap by making the side lobes of the autocorrelation
    # (and thus the variation in overlap) smaller than a single-tone or low-frequency approach.
    
    # Number of harmonics to use
    M = N // 2
    
    # Generate random phases for each harmonic
    # Using a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, M)
    
    i = np.arange(N)
    x = np.zeros(N)
    
    # Sum of cosines with random phases
    for m in range(1, M + 1):
        x += np.cos(2 * np.pi * m * i / N + phases[m-1])
        
    # Normalize x to fit within [-0.5, 0.5]
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs * 0.5
    else:
        x = np.zeros(N)
        
    h = 0.5 + x
    
    # Ensure exact sum is N/2 and values are in [0, 1]
    # Due to floating point, sum might not be exactly N/2
    # Also, clipping might be needed if numerical errors push it out of bounds
    h = np.clip(h, 0, 1)
    
    # Adjust to ensure sum is exactly N/2
    # The sum of h is N/2 + sum(x). Since sum(x) is close to 0 but not exactly 0 due to finite precision and random phases,
    # we might need a small correction.
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    
    if diff != 0:
        # Distribute the difference evenly or adjust a few elements
        # A simple way is to add diff/N to all, then re-clip and iterate, 
        # but for N=100, adding diff/N is small enough to not violate bounds significantly.
        h += diff / N
        h = np.clip(h, 0, 1)
        
    return h

# EVOLVE-BLOCK-END
