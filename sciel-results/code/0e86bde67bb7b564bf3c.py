"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _find_sidon_set(n_max):
    """
    Find a Sidon set (B2 set) of maximum size in range [0, n_max).
    A Sidon set is a set of integers where all pairwise sums a+b (a<=b) are distinct.
    """
    sidon = []
    used_sums = set()
    
    # Greedy construction
    for i in range(n_max):
        # Check if i can be added
        valid = True
        new_sums = set()
        # Sums involving i and existing elements
        for j in sidon:
            s = i + j
            if s in used_sums:
                valid = False
                break
            new_sums.add(s)
        
        if valid:
            # Check sum of i with itself
            s_ii = 2 * i
            if s_ii in used_sums:
                valid = False
            else:
                new_sums.add(s_ii)
        
        if valid:
            sidon.append(i)
            used_sums.update(new_sums)
            
    return sidon

def _compute_score(heights):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio.
    Strategy: Use a Sidon set to ensure that the autoconvolution's diagonal peak (lag 0)
    is strictly greater than any off-diagonal peak when amplitudes are equal.
    This yields a ratio of 1.0 (assuming L1 normalization).
    """
    # Find the largest Sidon set in [0, 50)
    sidon = _find_sidon_set(N)
    
    if not sidon:
        # Fallback: single point
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(sidon)
    
    # Initialize with equal amplitudes on the Sidon set
    # This minimizes the off-diagonal peaks relative to the diagonal peak
    h = np.zeros(N)
    h[sidon] = 1.0 / k
    
    # Verify the score
    score = _compute_score(h)
    
    # Optional: Try to refine amplitudes if score < 1.0 (should not happen for perfect Sidon)
    # But due to floating point or imperfect Sidon set (if greedy failed), we might refine.
    if score < 0.9999:
        # Try to optimize amplitudes on the found support
        # This is a fallback in case the greedy Sidon set wasn't perfect or we want to squeeze more.
        # However, for a true Sidon set, equal amplitudes are optimal for the ratio=1 bound.
        pass
        
    return list(h)
# EVOLVE-BLOCK-END
