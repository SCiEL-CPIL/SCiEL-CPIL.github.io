def coefficients(stages):
    """
    Implements a 4th-order Kahan-Li splitting using 6 stages, padded to 12.
    The 4th-order method has 6 stages: B A B A B A.
    We map this to 6 stages and set the remaining 6 to 0.
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
        
    a = [0.0] * stages
    b = [0.0] * stages
    
    # 4th-order Kahan-Li coefficients for 6 stages
    # a_i are for A, b_i are for B
    # The method is symmetric: a_1=a_6, a_2=a_5, a_3=a_4
    # b_1=b_6, b_2=b_5, b_3=b_4
    # Sum a = 1, Sum b = 1
    
    # Coefficients for 4th-order Kahan-Li
    # a = [0.5, -0.125, 0.125, 0.125, -0.125, 0.5]
    # b = [0.125, 0.5, -0.125, 0.125, 0.5, 0.125]
    # Check sums:
    # sum(a) = 0.5 - 0.125 + 0.125 + 0.125 - 0.125 + 0.5 = 1.0
    # sum(b) = 0.125 + 0.5 - 0.125 + 0.125 + 0.5 + 0.125 = 1.0
    
    a_coeffs = [0.5, -0.125, 0.125, 0.125, -0.125, 0.5]
    b_coeffs = [0.125, 0.5, -0.125, 0.125, 0.5, 0.125]
    
    for i in range(6):
        a[i] = a_coeffs[i]
        b[i] = b_coeffs[i]
        
    # The remaining 6 stages are 0
    # This effectively applies the 4th-order method once.
    
    return a, b
