import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses the zeros of the Laguerre polynomial L_12(x) scaled to match the magnitude
    of high-performing arithmetic configurations.
    """
    t0 = time.time()
    deadline = 590.0  # Safety margin
    
    # Get the zeros of the Laguerre polynomial L_12(x)
    # These are the natural nodes for the construction
    try:
        lag_zeros = roots_laguerre(12)[0]
        lag_zeros = np.sort(lag_zeros)
    except Exception:
        # Fallback to arithmetic if scipy fails, scaled by 10
        return [40.0 * (i + 1) for i in range(12)]

    # The seed program used arithmetic roots [4, 8, ..., 48].
    # Scaling by 10 gave [40, ..., 480] with score ~33.
    # Scaling by 1000 gave [4000, ..., 48000] with score ~33.3.
    
    # The Laguerre zeros have a max of ~19.8.
    # Let's scale them to match the "scale 10" arithmetic max (480) first.
    # If the structural benefit is real, this should outperform arithmetic scale 10.
    # If the magnitude benefit dominates, we might need to scale higher.
    # Given the stagnation between scale 10 and 1000 for arithmetic, 
    # the structural difference is likely the key.
    
    # Let's try scaling the Laguerre zeros to have a max of 480.
    scale_factor = 480.0 / lag_zeros[-1]
    scaled_lag_zeros = lag_zeros * scale_factor
    
    # Check if time is running out
    if time.time() - t0 > deadline:
        return list(scaled_lag_zeros)
        
    # Return the scaled Laguerre zeros
    return list(scaled_lag_zeros)
