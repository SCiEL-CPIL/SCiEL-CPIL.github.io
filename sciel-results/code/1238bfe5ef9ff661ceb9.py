"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def coefficients(stages):
    """
    6th-order symmetric Suzuki-Trotter splitting mapped to 12 (B, A) stages.
    
    The 6th-order scheme is S_6(t) = S_2(d1*t) S_2(d2*t) S_2(d3*t) S_2(d2*t) S_2(d1*t).
    Expanding this into alternating B and A exponentials yields 6 B-steps and 5 A-steps.
    We map these to the first 6 and 5 slots of the 12-stage format, padding the rest with zeros.
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

    # 6th-order Suzuki-Trotter coefficients
    # S_6(t) = S_2(d1*t) * S_2(d2*t) * S_2(d3*t) * S_2(d2*t) * S_2(d1*t)
    # S_2(t) = exp(-i t B / 2) * exp(-i t A) * exp(-i t B / 2)
    
    # Coefficients for the 6th order scheme
    # d1 = 1 / (2 * (2 + 2^(1/3)))
    # d2 = -2^(1/3) / (2 * (2 + 2^(1/3)))
    # d3 = 1 - 2*d1 - 2*d2
    
    cbrt2 = 2.0 ** (1.0 / 3.0)
    denom = 2.0 * (2.0 + cbrt2)
    d1 = 1.0 / denom
    d2 = -cbrt2 / denom
    d3 = 1.0 - 2.0 * d1 - 2.0 * d2
    
    # Expanding S_6(t):
    # S_2(d1*t): B_{d1/2} A_{d1} B_{d1/2}
    # S_2(d2*t): B_{d2/2} A_{d2} B_{d2/2}
    # S_2(d3*t): B_{d3/2} A_{d3} B_{d3/2}
    # S_2(d2*t): B_{d2/2} A_{d2} B_{d2/2}
    # S_2(d1*t): B_{d1/2} A_{d1} B_{d1/2}
    
    # Concatenated B terms:
    # 1. d1/2
    # 2. d1/2 + d2/2
    # 3. d2/2 + d3/2
    # 4. d3/2 + d2/2
    # 5. d2/2 + d1/2
    # 6. d1/2
    
    # Concatenated A terms:
    # 1. d1
    # 2. d2
    # 3. d3
    # 4. d2
    # 5. d1
    
    b_vals = [
        d1 / 2.0,
        (d1 + d2) / 2.0,
        (d2 + d3) / 2.0,
        (d3 + d2) / 2.0,
        (d2 + d1) / 2.0,
        d1 / 2.0
    ]
    
    a_vals = [
        d1,
        d2,
        d3,
        d2,
        d1
    ]
    
    # Initialize lists with zeros
    b = [0.0] * 12
    a = [0.0] * 12
    
    # Fill in the non-zero coefficients
    for i in range(6):
        b[i] = b_vals[i]
    for i in range(5):
        a[i] = a_vals[i]
        
    # Ensure sums are exactly 1 (numerical stability check)
    # The sums should be 1 by construction, but let's verify and correct if needed
    sb = sum(b)
    sa = sum(a)
    
    if abs(sb - 1.0) > 1e-12:
        b = [v / sb for v in b]
    if abs(sa - 1.0) > 1e-12:
        a = [v / sa for v in a]
        
    return a, b
# EVOLVE-BLOCK-END
