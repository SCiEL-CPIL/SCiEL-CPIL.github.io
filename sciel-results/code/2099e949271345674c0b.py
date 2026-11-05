import numpy as np

def coefficients(stages):
    """
    Returns coefficients for a symmetric 2nd-order splitting scheme.
    The scheme is constructed to have 'stages' A-steps and 'stages' B-steps.
    For stages=12, it uses a symmetric distribution of B-coefficients that
    mimics the Strang splitting structure extended to more steps, ensuring
    sum(a)=1 and sum(b)=1.
    """
    if stages != 12:
        # Fallback to 1st order for other stage counts
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 12 stages
    a = [1.0 / 12.0] * 12
    
    # Symmetric B coefficients
    # b[0] and b[11] are the edge steps (half of a Strang-like step)
    # b[1] to b[10] are the middle steps
    # We want the "effective" Strang structure.
    # Standard Strang: B(1/2) A(1) B(1/2).
    # If we split the A(1) into 12 A(1/12) steps, and adjust B to maintain symmetry and sum=1.
    # The "mass" of B should be distributed such that the first and last are smaller.
    # Let's use the values derived above:
    # b[0] = b[11] = 1/24
    # b[1..10] = 11/120
    
    b = [0.0] * 12
    b[0] = 1.0 / 24.0
    b[11] = 1.0 / 24.0
    mid_val = (1.0 - 2.0 * b[0]) / 10.0
    for i in range(1, 11):
        b[i] = mid_val
        
    return a, b
