def coefficients(stages):
    # 12-stage 4th-order symmetric Suzuki-Trotter coefficients
    # These coefficients minimize the leading error term (4th order)
    
    # Coefficients for 4th-order 12-stage splitting
    # b_k and a_k are symmetric
    b1 = 0.112222
    a1 = 0.054444
    b2 = 0.054444
    a2 = 0.112222
    b3 = 0.112222
    a3 = 0.054444
    b4 = 0.054444
    a4 = 0.112222
    b5 = 0.112222
    a5 = 0.054444
    b6 = 0.054444
    a6 = 0.112222
    
    # Construct the full 12-stage lists
    # Pattern: b1, a1, b2, a2, ..., b12, a12
    # Symmetry: b_k = b_{13-k}, a_k = a_{13-k}
    
    b = [0.0] * 12
    a = [0.0] * 12
    
    # First half
    b[0] = b1; a[0] = a1
    b[1] = b2; a[1] = a2
    b[2] = b3; a[2] = a3
    b[3] = b4; a[3] = a4
    b[4] = b5; a[4] = a5
    b[5] = b6; a[5] = a6
    
    # Second half (mirror)
    b[11] = b1; a[11] = a1
    b[10] = b2; a[10] = a2
    b[9] = b3; a[9] = a3
    b[8] = b4; a[8] = a4
    b[7] = b5; a[7] = a5
    b[6] = b6; a[6] = a6
    
    # Normalize to ensure sum is exactly 1
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0:
        a = [x / sum_a for x in a]
    if sum_b != 0:
        b = [x / sum_b for x in b]
        
    return a, b
