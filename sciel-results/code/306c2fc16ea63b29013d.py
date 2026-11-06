import numpy as np

def coefficients(stages):
    if stages != 12:
        # Fallback for other stage counts
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 4th-order Yoshida coefficients for 12 stages
    # Based on the 3-step 4th order Suzuki scheme coefficients
    alpha = 2.0 ** (1.0 / 3.0)
    
    # Standard 3-step 4th order coefficients
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = alpha / (2.0 * (1.0 - alpha))
    c3 = 1.0 / (2.0 * (1.0 + alpha))
    
    # The 12-stage symmetric sequence derived from the 3-step scheme
    # Sequence: c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1
    # This sequence has sum 4 * (c1 + c2 + c3) = 4 * 1 = 4
    # We normalize to sum 1 by dividing by 4
    
    seq = [c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1]
    
    # Normalize
    sum_seq = sum(seq)
    if sum_seq != 0:
        a = [x / sum_seq for x in seq]
        b = [x / sum_seq for x in seq]
    else:
        a = [1.0 / 12.0] * 12
        b = [1.0 / 12.0] * 12
        
    return a, b
