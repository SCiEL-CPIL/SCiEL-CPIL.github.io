import numpy as np

def test_2x2x3_rank8():
    # A is 2x2, B is 2x3, C = A@B is 2x3
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[0,2] = a*i + b*j
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    # C[1,2] = c*i + d*j
    
    # Rank 8 decomposition:
    # M1 = (a+c)*e
    # M2 = (a-c)*g
    # M3 = (b+d)*f
    # M4 = (b-d)*h
    # M5 = (a+b)*i
    # M6 = (a-b)*j
    # M7 = (c+d)*i
    # M8 = (c-d)*j
    
    # C[0,0] = (M1+M2)/2
    # C[1,0] = (M1-M2)/2
    # C[0,1] = (M3+M4)/2
    # C[1,1] = (M3-M4)/2
    # C[0,2] = (M5+M7)/2 + ... 
    
    # Wait, C[0,2] should be a*i + b*j
    # M5 = (a+b)*i, M7 = (c+d)*i
    # M6 = (a-b)*j, M8 = (c-d)*j
    
    # C[0,2] = a*i + b*j
    # From M5: a*i + b*i
    # From M6: a*j - b*j
    # (M5+M6)/2 = (a*i + b*i + a*j - b*j)/2 - not quite
    
    # Let me try:
    # C[0,2] = (M5 + M6)/2 where M5=(a+b)*i, M6=(a-b)*j? No.
    
    # Actually:
    # (a+b)*i + (a-b)*j = a*i + b*i + a*j - b*j
    # We want a*i + b*j
    # Difference: b*i - b*j = b*(i-j)
    
    # This isn't working cleanly. Let me just verify numerically.
    
    np.random.seed(42)
    A = np.random.randn(2,2)
    B = np.random.randn(2,3)
    C = A @ B
    
    a,b,c,d = A[0,0], A[0,1], A[1,0], A[1,1]
    e,f,g,h,i,j = B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2]
    
    M1 = (a+c)*e
    M2 = (a-c)*g
    M3 = (b+d)*f
    M4 = (b-d)*h
    M5 = (a+b)*i
    M6 = (a-b)*j
    M7 = (c+d)*i
    M8 = (c-d)*j
    
    C00 = (M1+M2)/2
    C10 = (M1-M2)/2
    C01 = (M3+M4)/2
    C11 = (M3-M4)/2
    C02 = (M5+M6)/2  # ?
    C12 = (M7+M8)/2  # ?
    
    C_est = np.array([[C00, C01, C02], [C10, C11, C12]])
    
    print("C:", C)
    print("C_est:", C_est)
    print("Match:", np.allclose(C, C_est))
    
    # Check what C02 and C12 actually compute
    print("C[0,2] actual:", C[0,2], "est:", C02)
    print("C[1,2] actual:", C[1,2], "est:", C12)

test_2x2x3_rank8()
