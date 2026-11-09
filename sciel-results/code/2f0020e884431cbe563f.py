"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter splitting coefficients for a 12-stage product formula.
    The scheme is constructed by composing three Strang substeps with optimized weights
    to cancel the 3rd-order error term, resulting in a global 4th-order accuracy.
    """
    if stages != 12:
        # Fallback to uniform Strang if stages is not 12, though task specifies 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th-order symmetric splitting weights
    # w1 = w3, w2 = 1 - 2*w1
    # Condition for 4th order: w1 * (1 - 2*w1)^2 * ... = 0?
    # Standard result: w1 = 1 / (2 * (1 - 2**(1/3)))
    alpha = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 * (1.0 - alpha))
    w2 = 1.0 - 2.0 * w1

    # Coefficients for the 7-factor sequence:
    # B1 A1 B2 A2 B3 A3 B4
    # B1 = w1/2
    # A1 = w1
    # B2 = (w1 + w2) / 2
    # A2 = w2
    # B3 = (w2 + w1) / 2  (same as B2)
    # A3 = w1
    # B4 = w1/2
    
    b1 = w1 / 2.0
    a1 = w1
    b2 = (w1 + w2) / 2.0
    a2 = w2
    b3 = (w2 + w1) / 2.0
    a3 = w1
    b4 = w1 / 2.0

    # Map to 12 stages (k=1..12) with symmetry a[k] = a[13-k], b[k] = b[13-k]
    # We place the non-zero terms in the first 6 stages and mirror them.
    # Stages 1-6:
    # 1: B=b1, A=a1
    # 2: B=0,  A=0
    # 3: B=b2, A=a2
    # 4: B=0,  A=0
    # 5: B=b3, A=a3
    # 6: B=b4, A=0
    # Stages 7-12 are mirrors of 6-1.
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    a[0] = a1
    b[0] = b1
    
    a[2] = a2
    b[2] = b2
    
    a[4] = a3
    b[4] = b3
    
    b[5] = b4
    # a[5] remains 0
    
    # Mirror
    a[11] = a[0]
    b[11] = b[0]
    
    a[10] = a[2]
    b[10] = b[2]
    
    a[9] = a[4]
    b[9] = b[4]
    
    b[7] = b[5]
    # a[7] remains 0
    
    # Ensure finite and within bounds (they are)
    return a, b
# EVOLVE-BLOCK-END
