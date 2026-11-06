import math

def coefficients(stages):
    """
    Implements a 4th-order Yoshida splitting using 12 stages.
    The 4th-order method uses 5 sub-steps of the 2nd-order Strang method.
    The remaining stages are set to zero.
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    a = [0.0] * stages
    b = [0.0] * stages

    # Yoshida 4th-order coefficients
    # c1 = 1 / (4 - 2^(1/3))
    # c2 = -2^(1/3) / (4 - 2^(1/3))
    # c3 = 1 - 2*c1
    c1 = 1.0 / (4.0 - 2.0 ** (1.0 / 3.0))
    c2 = -2.0 ** (1.0 / 3.0) / (4.0 - 2.0 ** (1.0 / 3.0))
    c3 = 1.0 - 2.0 * c1

    # The 5 Strang steps have weights: c1, c2, c3, c2, c1
    weights = [c1, c2, c3, c2, c1]

    # Map to 12 stages
    # Sequence: B(w1/2) A(w1) B((w1+w2)/2) A(w2) B((w2+w3)/2) A(w3) B((w3+w2)/2) A(w2) B((w2+w1)/2) A(w1) B(w1/2)
    
    # Stage 1
    b[0] = weights[0] / 2.0
    a[0] = weights[0]
    
    # Stage 2
    b[1] = (weights[0] + weights[1]) / 2.0
    a[1] = weights[1]
    
    # Stage 3
    b[2] = (weights[1] + weights[2]) / 2.0
    a[2] = weights[2]
    
    # Stage 4
    b[3] = (weights[2] + weights[3]) / 2.0
    a[3] = weights[3]
    
    # Stage 5
    b[4] = (weights[3] + weights[4]) / 2.0
    a[4] = weights[4]
    
    # Stage 6
    b[5] = weights[4] / 2.0
    a[5] = 0.0
    
    # Stages 7-12 are zero
    for k in range(6, stages):
        a[k] = 0.0
        b[k] = 0.0

    return a, b
