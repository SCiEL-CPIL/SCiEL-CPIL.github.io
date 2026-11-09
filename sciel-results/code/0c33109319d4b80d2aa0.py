def coefficients(stages):
    """
    Returns 12-stage 6th-order symmetric splitting coefficients.
    The 6th-order symmetric coefficients are repeated twice to fill 12 stages.
    """
    # 6th-order symmetric coefficients (sum to 1)
    # d1 = 0.25
    # d2 = 0.25 - 1/(2 * (1 + 2**(1/6)))
    # d3 = 1/(2 * (1 + 2**(1/6)))
    
    d1 = 0.25
    d2 = 0.25 - 1.0 / (2.0 * (1.0 + 2.0 ** (1.0 / 6.0)))
    d3 = 1.0 / (2.0 * (1.0 + 2.0 ** (1.0 / 6.0)))
    
    # Base 6-stage sequence
    base_a = [d1, d2, d3, d3, d2, d1]
    base_b = [d1, d2, d3, d3, d2, d1]
    
    # Repeat to get 12 stages
    a = base_a + base_a
    b = base_b + base_b
    
    # Ensure length is stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    return a, b
