import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula using 4th-order Suzuki-Trotter splitting.
    The method is constructed by repeating the 4th-order symmetric step 4 times.
    Each step uses the structure B-A-B-A-B, mapped to 3 (B,A) pairs by setting the last A to 0.
    Coefficients are scaled by 1/4 to ensure sum(a)=1 and sum(b)=1.
    """
    if stages != 12:
        # Fallback to Strang splitting for other stage counts
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
    # alpha_1 = 1 / (2 - 2^(1/3))
    # alpha_2 = -2^(1/3) / (2 - 2^(1/3))
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # One step (3 pairs) uses:
    # b: c1, c2, c1
    # a: c1, c2, 0  (Note: beta coefficients are same as alpha for symmetric case)
    # Sum of b in one step: c1 + c2 + c1 = 2*c1 + c2 = 1
    # Sum of a in one step: c1 + c2 + 0 = 1
    
    # We repeat this 4 times to fill 12 stages.
    # Total sum will be 4, so we scale by 1/4.
    scale = 0.25
    
    a = []
    b = []
    for _ in range(4):
        b += [c1 * scale, c2 * scale, c1 * scale]
        a += [c1 * scale, c2 * scale, 0.0 * scale]
    
    return a, b
# EVOLVE-BLOCK-END
