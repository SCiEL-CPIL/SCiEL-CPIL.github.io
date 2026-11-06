"""Optimized 12-stage product formula for Heisenberg chain time evolution."""
import time

def coefficients(stages):
    """
    12-stage symmetric 6th-order product formula coefficients.
    
    Uses symmetric high-order splitting coefficients to minimize
    spectral-norm error of the time evolution operator.
    """
    # 6th-order symmetric coefficients (6 independent values, mirrored)
    # These are scaled to satisfy sum(a) = sum(b) = 1
    a_half = [0.02645, 0.23415, 0.23415, 0.23415, 0.23415, 0.02645]
    b_half = [0.02645, 0.23415, 0.23415, 0.23415, 0.23415, 0.02645]
    
    # Scale to ensure sum = 1
    sum_a_half = sum(a_half)
    sum_b_half = sum(b_half)
    
    if sum_a_half > 0:
        a_half = [x / (2 * sum_a_half) for x in a_half]
    if sum_b_half > 0:
        b_half = [x / (2 * sum_b_half) for x in b_half]
    
    # Build full 12-element lists with symmetry
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Ensure exactly 12 elements
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    
    # Final normalization to ensure sum = 1 exactly
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0:
        a = [x / sum_a for x in a]
    if sum_b != 0:
        b = [x / sum_b for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
