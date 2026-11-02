import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products definitions
    # P1: (X00+X11)(Y00+Y11)
    # P2: (X10+X11)Y00
    # P3: X00(Y01-Y11)
    # P4: X11(Y10-Y00)
    # P5: (X00+X01)Y11
    # P6: (X10-X00)(Y00+Y01)
    # P7: (X01-X11)(Y10+Y11)
    
    strassen_a = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    strassen_b = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # C contributions: [coeff for Z00, coeff for Z01, coeff for Z10, coeff for Z11]
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    strassen_c = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # Define the 4 groups
    # Group 1: A1, B11 -> C1
    # A1 indices in U (i*m+j):
    # A00: 0*4+0=0, A01: 0*4+1=1, A10: 1*4+0=4, A11: 1*4+1=5
    a1_idx = [0, 1, 4, 5]
    # B11 indices in V (j*p+k):
    # B00: 0*4+0=0, B01: 0*4+1=1, B10: 1*4+0=4, B11: 1*4+1=5
    b11_idx = [0, 1, 4, 5]
    # C1 indices in W (k*n+i):
    # C00: 0*2+0=0, C01: 1*2+0=2, C10: 0*2+1=1, C11: 1*2+1=3
    c1_idx = [0, 2, 1, 3]
    
    # Group 2: A2, B21 -> C1
    # A2 indices in U:
    # A02: 0*4+2=2, A03: 0*4+3=3, A12: 1*4+2=6, A13: 1*4+3=7
    a2_idx = [2, 3, 6, 7]
    # B21 indices in V:
    # B20: 2*4+0=8, B21: 2*4+1=9, B30: 3*4+0=12, B31: 3*4+1=13
    b21_idx = [8, 9, 12, 13]
    # C1 indices in W: same as above
    c1_idx_same = c1_idx
    
    # Group 3: A1, B12 -> C2
    # A1 indices in U: same as above
    a1_idx_same = a1_idx
    # B12 indices in V:
    # B02: 0*4+2=2, B03: 0*4+3=3, B12: 1*4+2=6, B13: 1*4+3=7
    b12_idx = [2, 3, 6, 7]
    # C2 indices in W:
    # C02: 2*2+0=4, C03: 3*2+0=6, C12: 2*2+1=5, C13: 3*2+1=7
    c2_idx = [4, 6, 5, 7]
    
    # Group 4: A2, B22 -> C2
    # A2 indices in U: same as above
    a2_idx_same = a2_idx
    # B22 indices in V:
    # B22: 2*4+2=10, B23: 2*4+3=11, B32: 3*4+2=14, B33: 3*4+3=15
    b22_idx = [10, 11, 14, 15]
    # C2 indices in W: same as above
    c2_idx_same = c2_idx
    
    groups = [
        (a1_idx, b11_idx, c1_idx),
        (a2_idx, b21_idx, c1_idx_same),
        (a1_idx_same, b12_idx, c2_idx),
        (a2_idx_same, b22_idx, c2_idx_same)
    ]
    
    r = 0
    for a_idx, b_idx, c_idx in groups:
        for t in range(7):
            # Fill U
            for k in range(4):
                if strassen_a[t][k] != 0:
                    U[a_idx[k]][r] += strassen_a[t][k]
            # Fill V
            for k in range(4):
                if strassen_b[t][k] != 0:
                    V[b_idx[k]][r] += strassen_b[t][k]
            # Fill W
            for k in range(4):
                if strassen_c[t][k] != 0:
                    W[c_idx[k]][r] += strassen_c[t][k]
            r += 1
            
    return {"U": U, "V": V, "W": W}
