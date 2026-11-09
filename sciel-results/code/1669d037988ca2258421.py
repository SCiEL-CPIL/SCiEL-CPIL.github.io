"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Use a low-frequency, smooth perturbation (Gaussian) to ensure
    # positive autocorrelation R(k) for all lags k.
    # Since S(k) = N/4 - R(k), minimizing max S(k) requires maximizing min R(k).
    # A Gaussian's autocorrelation is a Gaussian, which is always positive.
    
    t = np.arange(N)
    # Center the Gaussian
    center = N / 2.0
    # Width parameter: smaller sigma means higher frequency, larger sigma means lower.
    # We want low frequency to keep R(k) positive over a wide range of k.
    # A very wide Gaussian (low freq) will have R(k) close to constant (positive).
    sigma = N / 4.0 
    
    # Generate Gaussian
    x = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # Make zero mean
    x = x - np.mean(x)
    
    # Scale to fit within [-0.5, 0.5]
    # Current range is roughly [-max_neg, max_pos]
    # We want h in [0,1], so x in [-0.5, 0.5]
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs * 0.5
    else:
        x = np.zeros(N)
        
    h = 0.5 + x
    
    # Ensure strict bounds [0, 1] and sum constraint
    # The scaling ensures bounds if max_abs was the limit.
    # But floating point might cause slight overflow.
    h = np.clip(h, 0, 1)
    
    # Adjust sum to be exactly N/2
    # Current sum is N*0.5 + sum(x). sum(x) is 0 by construction, but clipping might change it.
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    
    if diff != 0:
        # Distribute the difference uniformly
        h += diff / N
        
        # Re-clip and hope the sum stays close (diff is usually very small)
        h = np.clip(h, 0, 1)
        
        # If clipping broke the sum significantly, do a final correction
        # on the elements with the most margin
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if diff > 0:
            # Add to elements closest to 1
            order = np.argsort(1 - h)
            h[order[:int(abs(diff)*N)]] += diff / int(abs(diff)*N) if int(abs(diff)*N)>0 else 0
        elif diff < 0:
            # Subtract from elements closest to 0
            order = np.argsort(h)
            h[order[:int(abs(diff)*N)]] += diff / int(abs(diff)*N) if int(abs(diff)*N)>0 else 0
            
    return h
# EVOLVE-BLOCK-END
