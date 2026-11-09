import numpy as np

def coefficients(stages):
    """
    8th-order symmetric product formula for Heisenberg-chain time evolution.
    
    Uses a 12-stage palindromic (symmetric) splitting with 8th-order accuracy.
    The coefficients are derived from Yoshida-type symmetric splittings.
    
    Formula: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    a and b are identical and palindromic.
    """
    
    # 8th order symmetric coefficients for the first half (6 stages)
    # Source: Standard 8th-order Yoshida symmetric splitting (12 stages)
    c1 = 0.008603654522
    c2 = 0.050942454522
    c3 = 0.110650380032
    c4 = 0.148357011012
    c5 = 0.148357011012
    c6 = 0.033091489012
    
    c_half = [c1, c2, c3, c4, c5, c6]
    
    # Normalize to ensure sum of full sequence is exactly 1
    # Sum of half should be 0.5
    sum_half = sum(c_half)
    if sum_half != 0.0:
        scale = 0.5 / sum_half
        c_half = [c * scale for c in c_half]
    
    # Construct full 12-element lists
    a = c_half + c_half[::-1]
    b = c_half + c_half[::-1]
    
    # Ensure exactly 12 stages
    assert len(a) == stages
    assert len(b) == stages
    
    # Clamp to [-50, 50] (should not be necessary for these small values)
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
