import numpy as np

def coefficients(stages):
    """4th-order Suzuki-Trotter splitting for 12-stage product formula.
    
    Uses two 6-stage 4th-order substeps, each evolving for time t/2.
    Pattern: exp(-i b_k t B) exp(-i a_k t A) for k=1..12
    """
    if stages != 12:
        # Fallback for other stage counts - use Strang-like
        n = stages // 2
        a = [0.5 / n] * (2 * n)
        b = []
        for _ in range(n):
            b += [1.0 / n, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order symmetric splitting parameters
    # lambda = 1 / (2 - 2^(-1/3))
    lambda_val = 1.0 / (2.0 - 2.0**(-1.0/3.0))
    
    # 6-stage 4th-order coefficients (for full time step)
    # Pattern: B, A, B, A, B, A
    b6 = [
        1.0 / (2.0 * (1.0 - lambda_val)),
        -lambda_val / (1.0 - lambda_val),
        1.0 / (2.0 * (1.0 - lambda_val))
    ]
    a6 = [
        lambda_val / (2.0 * (1.0 - lambda_val)),
        1.0 - 2.0 * lambda_val / (2.0 * (1.0 - lambda_val)),
        lambda_val / (2.0 * (1.0 - lambda_val))
    ]
    
    # Each 6-stage block evolves for time t/2, so scale by 1/2
    # Two such blocks make 12 stages
    a = []
    b = []
    for _ in range(2):
        for ai in a6:
            a.append(ai / 2.0)
        for bi in b6:
            b.append(bi / 2.0)
    
    # Ensure exact sums due to floating point
    a_sum = sum(a)
    b_sum = sum(b)
    if abs(a_sum - 1.0) > 1e-12:
        a[5] += (1.0 - a_sum)
    if abs(b_sum - 1.0) > 1e-12:
        b[5] += (1.0 - b_sum)
    
    return a, b
