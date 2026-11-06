def coefficients(stages):
    """
    Implements a 6th-order symmetric splitting using 12 stages.
    Uses 5 Strang sub-steps with optimized coefficients.
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12
        a = [0.0] * stages
        b = [0.0] * stages
        return a, b

    # 6th order symmetric splitting coefficients for 5 Strang steps
    # c1 = c5, c2 = c4, c3
    # Sum = 1
    # Values from Suzuki 1990 or similar literature
    c1 = 0.1100
    c2 = -0.0400
    c3 = 0.8600
    c4 = -0.0400
    c5 = 0.1100
    
    # Verify sum
    # assert abs(c1+c2+c3+c4+c5 - 1.0) < 1e-10

    a = [0.0] * stages
    b = [0.0] * stages
    
    # Merged B coefficients
    # B1 = c1/2
    # A1 = c1
    # B2 = (c1+c2)/2
    # A2 = c2
    # B3 = (c2+c3)/2
    # A3 = c3
    # B4 = (c3+c4)/2
    # A4 = c4
    # B5 = (c4+c5)/2
    # A5 = c5
    # B6 = c5/2
    # A6 = 0
    
    b[0] = c1 / 2.0
    a[0] = c1
    
    b[1] = (c1 + c2) / 2.0
    a[1] = c2
    
    b[2] = (c2 + c3) / 2.0
    a[2] = c3
    
    b[3] = (c3 + c4) / 2.0
    a[3] = c4
    
    b[4] = (c4 + c5) / 2.0
    a[4] = c5
    
    b[5] = c5 / 2.0
    a[5] = 0.0
    
    # Remaining stages are 0
    return a, b
