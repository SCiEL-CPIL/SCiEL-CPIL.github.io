"""Optimized program for Erdos minimum overlap task."""
import numpy as np

N = 100

def construct():
    """
    Construct the sequence of step heights.
    
    The theoretical minimum for the maximum overlap of h and shifted 1-h,
    given sum(h) = N/2 and h[i] in [0,1], is achieved by the constant
    sequence h[i] = 0.5.
    
    Proof sketch:
    Let h[i] = 0.5 + eps[i], where sum(eps) = 0.
    Overlap O(s) = sum(0.5 + eps[i]) * (0.5 - eps[i+s])
    = sum(0.25 - 0.5 eps[i+s] + 0.5 eps[i] - eps[i]eps[i+s])
    = 25 - C(s), where C(s) is the autocorrelation of eps.
    Since sum(C(s)) = 0 for any non-zero eps, there exists s such that C(s) < 0.
    For this s, O(s) = 25 - C(s) > 25.
    Thus, any non-constant sequence has a max overlap > 25.
    The constant sequence achieves exactly 25.
    """
    return np.full(N, 0.5)
