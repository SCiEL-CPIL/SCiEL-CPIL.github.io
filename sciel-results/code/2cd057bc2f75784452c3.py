"""Minimize normalized max autoconvolution of nonnegative step function on [-1/4,1/4]."""
import time
import numpy as np

N = 600
T_START = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - T_START)

def _autoconv_peak(heights):
    """Compute max |autoconv| / (max(h) * sum(h)) (assumed metric)."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    total = h.sum()
    max_h = h.max()
    if total < 1e-15 or max_h < 1e-15:
        return 1e10
    # Linear autoconvolution
    ac = np.convolve(h, h, mode='full')
    # The continuous autoconv peak is approximated by the discrete peak scaled by w^2?
    # But the ratio is scale invariant for the shape.
    # The discrete peak is max(ac).
    # The normalization in the discrete domain:
    # Mass M = sum(h) * w. Max H = max(h).
    # Peak G = max(ac) * w^2.
    # Ratio = (max(ac) * w^2) / (max(h) * sum(h) * w) = max(ac) * w / (max(h) * sum(h)).
    # w is constant, so we minimize max(ac) / (max(h) * sum(h)).
    return max(ac) / (max_h * total)

def construct():
    t0 = time.time()
    
    # Generate a Triangle function
    # h[i] is proportional to distance from center, or 1 - distance from center?
    # Triangle peaked at center: 1 - |x|
    # Triangle V-shape: |x|
    # Both have ratio 0.66.
    # Let's use the one peaked at center (standard triangle).
    
    x = np.linspace(-1, 1, N)
    h_tri = 1.0 - np.abs(x)
    h_tri = np.maximum(h_tri, 0)
    
    # Also try V-shape
    h_v = np.abs(x)
    
    # Also try Uniform for comparison
    h_uni = np.ones(N)
    
    # Evaluate
    val_tri = _autoconv_peak(h_tri)
    val_v = _autoconv_peak(h_v)
    val_uni = _autoconv_peak(h_uni)
    
    best_h = h_tri
    best_val = val_tri
    
    if val_v < best_val:
        best_val = val_v
        best_h = h_v
        
    if val_uni < best_val:
        best_val = val_uni
        best_h = h_uni
        
    # Check time
    if _time_left() < 5:
        return list(np.maximum(best_h, 0))
        
    # Try to optimize the shape slightly?
    # Maybe a power law: (1 - |x|^p)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        h_pow = np.maximum(1.0 - np.abs(x)**p, 0)
        val_pow = _autoconv_peak(h_pow)
        if val_pow < best_val:
            best_val = val_pow
            best_h = h_pow
            
    return list(np.maximum(best_h, 0))
