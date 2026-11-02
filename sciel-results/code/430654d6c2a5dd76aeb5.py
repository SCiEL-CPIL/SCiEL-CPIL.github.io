"""
Third autocorrelation inequality: choose a signed step function with 400 equal intervals on [-1/4, 1/4]
to improve an autoconvolution upper bound.

We model the step function as a binary sign vector derived from the zero crossings
of a frequency-modulated sine wave (chirp), optimizing over center frequency
and chirp rate to minimize the peak side-lobe of the autoconvolution.
"""

import numpy as np

N = 400

def _compute_peak_side_lobe(signs):
    """
    Compute the peak side-lobe magnitude of the aperiodic autocorrelation.
    This is a robust proxy for minimizing the autoconvolution upper bound,
    as it directly controls the worst-case correlation deviation.
    """
    signs = np.asarray(signs, dtype=np.float64)
    n = len(signs)
    
    # Compute aperiodic autocorrelation using FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    f = np.fft.rfft(signs, n=size)
    autocorr = np.fft.irfft(f * np.conj(f), n=size)[:2*n-1]
    
    # Exclude lag 0 (autocorr[0] and autocorr[2n-2] are the same, autocorr[n] is lag 0 in full circle? No)
    # autocorr[0] is R(0) = N
    # autocorr[1..n-1] are R(1)..R(n-1)
    # autocorr[n..2n-2] are R(-1)..R(-(n-1))
    # We want max abs of R(k) for k != 0.
    # Due to symmetry, we can just look at 1..n-1.
    if n > 1:
        side_lobes = np.abs(autocorr[1:n])
        peak = np.max(side_lobes)
    else:
        peak = 0.0
        
    return peak

def _local_optimize(signs, max_iters=2000, patience=50):
    """
    Local search by flipping individual bits to reduce peak side-lobe.
    """
    signs = np.array(signs, dtype=np.float64)
    current_score = _compute_peak_side_lobe(signs)
    best_score = current_score
    best_signs = signs.copy()
    
    no_improvement_count = 0
    
    for _ in range(max_iters):
        improved = False
        # Try flipping each bit
        for i in range(N):
            signs[i] = -signs[i]
            new_score = _compute_peak_side_lobe(signs)
            if new_score < best_score - 1e-9:
                best_score = new_score
                best_signs = signs.copy()
                improved = True
                no_improvement_count = 0
            else:
                signs[i] = -signs[i] # revert
            
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > patience:
                break
        else:
            # If we improved, restart the scan from beginning? Or continue?
            # Restarting is simpler and often works well for small N.
            signs = best_signs.copy()
            
    return best_signs, best_score

def construct():
    """
    Construct a signed step function with N equal intervals.
    Returns a sequence of signed step heights (values in {-1, +1}).
    
    Strategy: Use low-frequency sine waves to create smooth sign blocks.
    We search over frequency and phase, then refine with local bit-flipping.
    """
    x = np.arange(N, dtype=np.float64)
    t = x / N
    
    best_signs = np.ones(N, dtype=np.float64)
    best_score = _compute_peak_side_lobe(best_signs)
    
    # Search over low frequencies and phases
    # Frequency f is in cycles over the interval [0, 1]
    # Low frequencies mean few sign changes (blocks)
    # f=0.5 -> 1 cycle -> 2 blocks (+ then -)
    # f=1.0 -> 2 cycles -> 4 blocks
    # ...
    
    freqs = np.linspace(0.5, 10.0, 100)
    phases = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    
    for f in freqs:
        for p in phases:
            sine_vals = np.sin(2 * np.pi * (f * t + p / (2 * np.pi)))
            signs = np.where(sine_vals >= 0, 1.0, -1.0)
            
            score = _compute_peak_side_lobe(signs)
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
                
    # Refine the best candidate found with local search
    if best_score < 1000: # Avoid refining if it's terrible
        refined_signs, refined_score = _local_optimize(best_signs)
        if refined_score < best_score:
            best_signs = refined_signs
            best_score = refined_score
            
    return best_signs
