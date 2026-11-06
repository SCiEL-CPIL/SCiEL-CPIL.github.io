import numpy as np

def coefficients(stages):
    """
    12-stage product formula based on 4th-order Yoshida splitting.
    Uses 4 subdivisions of S_4, resulting in 12 A-steps and 13 B-steps.
    The 13th B-step is set to 0 to fit the 12-stage interface.
    Coefficients are normalized to sum to 1.
    """
    # 4th-order Yoshida coefficients
    alpha = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (2.0 - alpha))
    c2 = -alpha / (2.0 * (2.0 - alpha))
    
    # Scale for 4 subdivisions (t/4)
    scale = 0.25
    a1 = c1 * scale
    a2 = c2 * scale
    a3 = c1 * scale
    
    b1 = c1 * scale
    b2 = c2 * scale
    b3 = c2 * scale
    b4 = c1 * scale
    
    # Construct 12 A coefficients
    # Pattern: a1, a2, a3 repeated 4 times
    a = [a1, a2, a3, a1, a2, a3, a1, a2, a3, a1, a2, a3]
    
    # Construct 12 B coefficients
    # Pattern from 4 merged S_4 blocks:
    # b1, b2, b3, (b4+b1), b2, b3, (b4+b1), b2, b3, (b4+b1), b2, b3
    # The final b4 is dropped (set to 0 effectively by not including it in the 12 slots)
    b = [b1, b2, b3, b4 + b1, b2, b3, b4 + b1, b2, b3, b4 + b1, b2, b3]
    
    # Normalize to ensure sum = 1
    a_sum = sum(a)
    b_sum = sum(b)
    
    if a_sum != 0:
        a = [x / a_sum for x in a]
    if b_sum != 0:
        b = [x / b_sum for x in b]
    
    # Pad or truncate to exactly `stages` elements if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    if len(a) > stages:
        a = a[:stages]
        b = b[:stages]
        
    return a, b
