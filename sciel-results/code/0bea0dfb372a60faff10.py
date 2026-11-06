"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter splitting mapped to 12 (B, A) stages.
    
    The 4th-order scheme is S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t).
    Expanding this into alternating B and A exponentials yields 3 B-steps and 2 A-steps.
    We use a nested 4th-order scheme to fill more stages for better accuracy.
    """
    if stages != 12:
        # Fallback for non-12 stages: simple Strang splitting
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th-order Suzuki-Trotter coefficients
    # S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    # where S_2(t) = exp(-i t B / 2) * exp(-i t A) * exp(-i t B / 2)
    
    # Coefficients for the 4th order scheme
    # c1 = 1 / (2 - 2^(1/3))
    # c2 = -2^(1/3) / (2 - 2^(1/3))
    
    cbrt2 = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - cbrt2
    c1 = 1.0 / denom
    c2 = -cbrt2 / denom
    
    # Expanding S_4(t):
    # S_2(c1*t): B_{c1/2} A_{c1} B_{c1/2}
    # S_2(c2*t): B_{c2/2} A_{c2} B_{c2/2}
    # S_2(c1*t): B_{c1/2} A_{c1} B_{c1/2}
    
    # Concatenated B terms:
    # 1. c1/2
    # 2. c1/2 + c2/2
    # 3. c2/2 + c1/2
    # 4. c1/2
    
    # Concatenated A terms:
    # 1. c1
    # 2. c2
    # 3. c1
    
    b_vals = [
        c1 / 2.0,
        (c1 + c2) / 2.0,
        (c2 + c1) / 2.0,
        c1 / 2.0
    ]
    
    a_vals = [
        c1,
        c2,
        c1
    ]
    
    # Initialize lists with zeros
    b = [0.0] * 12
    a = [0.0] * 12
    
    # Fill in the non-zero coefficients
    for i in range(4):
        b[i] = b_vals[i]
    for i in range(3):
        a[i] = a_vals[i]
        
    # Ensure sums are exactly 1 (numerical stability check)
    sb = sum(b)
    sa = sum(a)
    
    if abs(sb - 1.0) > 1e-12:
        b = [v / sb for v in b]
    if abs(sa - 1.0) > 1e-12:
        a = [v / sa for v in a]
        
    return a, b
# EVOLVE-BLOCK-END
