"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _fast_overlap_score(h):
    """
    Computes the maximum overlap of h with shifted 1-h over linear shifts.
    Uses np.correlate for speed.
    """
    one_minus_h = 1.0 - h
    # np.correlate(h, one_minus_h, 'valid') computes sum_{i=0}^{N-1-k} h[i] * one_minus_h[i+k]
    # This matches the definition in the original code.
    overlaps = np.correlate(h, one_minus_h, mode='valid')
    return np.max(overlaps)

def _spectral_construct():
    """
    Construct h as a sinusoidal perturbation around 0.5 to minimize max overlap.
    h[i] = 0.5 + A * sin(2*pi*f*i/N)
    We need sum(h) = N/2, so the mean must be 0.5.
    The overlap at shift k is:
    sum_i h[i]*(1-h[i+k]) = sum_i h[i] - sum_i h[i]*h[i+k]
    = N/2 - (sum_i h[i]^2 + cross terms)
    
    For a pure sinusoid with integer frequency, the cross-correlation sum_i h[i]*h[i+k]
    is constant for all k (due to orthogonality), which minimizes the maximum.
    
    We search over frequency f and amplitude A to minimize the max overlap.
    """
    best_h = None
    best_score = float('inf')
    
    # Try different frequencies (integer cycles in N samples)
    for f in range(1, N//2 + 1):
        # Try different amplitudes
        # A must be such that h[i] in [0,1], so A <= 0.5
        # We also want to minimize the max overlap
        # For a sinusoid, the overlap at shift k is approximately:
        # N/2 - A^2/2 (for large N, the cross-correlation is A^2/2 for all k)
        # But this is an approximation. Let's compute exactly.
        
        # Try a few amplitudes
        for A in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            i = np.arange(N)
            h = 0.5 + A * np.sin(2 * np.pi * f * i / N)
            
            # Clip to [0,1] and renormalize to have sum = N/2
            h = np.clip(h, 0, 1)
            
            # Renormalize: we need sum(h) = N/2 = 50
            current_sum = np.sum(h)
            if current_sum > 0:
                h = h * (N / 2.0) / current_sum
            
            # Clip again after scaling
            h = np.clip(h, 0, 1)
            
            # Adjust to ensure exact sum = N/2
            diff = N / 2.0 - np.sum(h)
            if abs(diff) > 1e-10:
                # Distribute the difference uniformly
                h += diff / N
                h = np.clip(h, 0, 1)
                
                # If still not exact, do a final correction
                diff2 = N / 2.0 - np.sum(h)
                if abs(diff2) > 1e-10:
                    # Add to the element with most room
                    if diff2 > 0:
                        idx = np.argmax(1.0 - h)
                        h[idx] += diff2
                    else:
                        idx = np.argmin(h)
                        h[idx] += diff2
                    h = np.clip(h, 0, 1)
            
            score = _fast_overlap_score(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_h, best_score

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly len/2.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Use spectral construction
    h_spectral, score_spectral = _spectral_construct()
    
    if h_spectral is not None:
        return h_spectral
    
    # Fallback: simple alternating pattern
    h = np.zeros(N)
    h[::2] = 1.0
    return h

# EVOLVE-BLOCK-END
