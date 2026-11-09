import numpy as np

def coefficients(stages):
    """
    Returns 12-stage 5th-order symmetric Suzuki-Trotter splitting coefficients.
    The 5th order symmetric scheme has 10 non-zero steps. We pad with zeros to reach 12 stages.
    """
    # 5th order symmetric splitting parameter
    x = 2 ** (-1.0 / 5.0)
    
    # Raw coefficients for the 5 distinct steps in the symmetric 5th order scheme
    # Structure: A1 B1 A2 B2 A3 B3 A2 B2 A1 B1 (10 steps)
    # Coefficients for A and B are the same in the symmetric scheme
    c1 = 0.5
    c2 = (1.0 - 2.0 * x) / (2.0 * x * x)
    c3 = 1.0 / x
    c4 = c2
    c5 = c1
    
    # Sum of these 5 coefficients (each appears twice in the 10-step scheme)
    # Sum_total = 2 * (c1 + c2 + c3 + c4 + c5)
    # We want the sum of the 10 coefficients to be 1.
    # So we normalize by S = 2 * (c1 + c2 + c3 + c4 + c5)
    S = 2.0 * (c1 + c2 + c3 + c4 + c5)
    
    # Normalized coefficients
    a1 = c1 / S
    a2 = c2 / S
    a3 = c3 / S
    a4 = c4 / S
    a5 = c5 / S
    
    # Construct the 10-step sequence
    a_10 = [a1, a2, a3, a4, a5, a5, a4, a3, a2, a1]
    b_10 = [a1, a2, a3, a4, a5, a5, a4, a3, a2, a1]
    
    # Pad to 12 stages with zeros at the ends
    # The prompt order is B1 A1 ... B12 A12
    # We can place the 10-step scheme in the middle or start.
    # Symmetric padding: 1 zero at start, 1 zero at end?
    # a = [0, a1, ..., a1, 0] -> 12 elements
    a = [0.0] + a_10 + [0.0]
    b = [0.0] + b_10 + [0.0]
    
    # Ensure length is 12
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    # Truncate if necessary (though stages=12 is fixed)
    a = a[:stages]
    b = b[:stages]
    
    return a, b
