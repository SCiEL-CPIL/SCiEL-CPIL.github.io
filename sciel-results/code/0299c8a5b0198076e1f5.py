import numpy as np
import time

def construct():
    # Rank-7 decomposition of 2x2x2 (Strassen)
    # u, v, w are vectors of length 4
    # Order: (00, 01, 10, 11) for A and C, (00, 01, 10, 11) for B
    # Note: B is 2x2, so indices are (row, col) -> 00, 01, 10, 11
    # Wait, for 2x2x2, B is 2x2.
    # For 2x2x3, B is 2x3.
    
    # Strassen 2x2x2 terms
    # u, v, w
    # 1. (1,0,0,1) (1,0,0,1) (1,0,0,1) -> C00, C11 get +1?
    # Let's use the standard Strassen formulas for the coefficients.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define u, v, w for each M
    # u is coeff of A, v is coeff of B, w is coeff of C
    
    s_u = []
    s_v = []
    s_w = []
    
    # M1
    s_u.append([1, 0, 0, 1])
    s_v.append([1, 0, 0, 1])
    s_w.append([1, 0, 0, 1]) # C00, C01, C10, C11 -> 1, 0, 0, 1
    
    # M2
    s_u.append([0, 0, 1, 1])
    s_v.append([1, 0, 0, 0])
    s_w.append([0, 0, 1, -1]) # C10 +1, C11 -1
    
    # M3
    s_u.append([1, 0, 0, 0])
    s_v.append([0, 0, 1, -1])
    s_w.append([0, 1, 0, 1]) # C01 +1, C11 +1
    
    # M4
    s_u.append([0, 0, 0, 1])
    s_v.append([0, 1, -1, 0])
    s_w.append([0, 0, 1, 0]) # C10 +1
    
    # M5
    s_u.append([1, 1, 0, 0])
    s_v.append([0, 0, 0, 1])
    s_w.append([-1, 1, 0, 0]) # C00 -1, C01 +1
    
    # M6
    s_u.append([0, 0, 1, -1])
    s_v.append([1, 1, 0, 0])
    s_w.append([0, 0, 0, 1]) # C11 +1
    
    # M7
    s_u.append([0, 1, 0, -1])
    s_v.append([0, 0, 1, 1])
    s_w.append([1, 0, 0, 0]) # C00 +1
    
    # Rank-12 decomposition of 2x2x3 (Schoolbook)
    # A: 4 entries (00, 01, 10, 11)
    # B: 6 entries (00, 01, 02, 10, 11, 12)
    # C: 6 entries (00, 01, 02, 10, 11, 12)
    
    b_u = []
    b_v = []
    b_w = []
    
    # C00 = A00 B00 + A01 B10
    b_u.append([1, 0, 0, 0]); b_v.append([1, 0, 0, 0, 0, 0]); b_w.append([1, 0, 0, 0, 0, 0])
    b_u.append([0, 1, 0, 0]); b_v.append([0, 0, 0, 1, 0, 0]); b_w.append([1, 0, 0, 0, 0, 0])
    
    # C01 = A00 B01 + A01 B11
    b_u.append([1, 0, 0, 0]); b_v.append([0, 1, 0, 0, 0, 0]); b_w.append([0, 1, 0, 0, 0, 0])
    b_u.append([0, 1, 0, 0]); b_v.append([0, 0, 0, 0, 1, 0]); b_w.append([0, 1, 0, 0, 0, 0])
    
    # C02 = A00 B02 + A01 B12
    b_u.append([1, 0, 0, 0]); b_v.append([0, 0, 1, 0, 0, 0]); b_w.append([0, 0, 1, 0, 0, 0])
    b_u.append([0, 1, 0, 0]); b_v.append([0, 0, 0, 0, 0, 1]); b_w.append([0, 0, 1, 0, 0, 0])
    
    # C10 = A10 B00 + A11 B10
    b_u.append([0, 0, 1, 0]); b_v.append([1, 0, 0, 0, 0, 0]); b_w.append([0, 0, 0, 1, 0, 0])
    b_u.append([0, 0, 0, 1]); b_v.append([0, 0, 0, 1, 0, 0]); b_w.append([0, 0, 0, 1, 0, 0])
    
    # C11 = A10 B01 + A11 B11
    b_u.append([0, 0, 1, 0]); b_v.append([0, 1, 0, 0, 0, 0]); b_w.append([0, 0, 0, 0, 1, 0])
    b_u.append([0, 0, 0, 1]); b_v.append([0, 0, 0, 0, 1, 0]); b_w.append([0, 0, 0, 0, 1, 0])
    
    # C12 = A10 B02 + A11 B12
    b_u.append([0, 0, 1, 0]); b_v.append([0, 0, 1, 0, 0, 0]); b_w.append([0, 0, 0, 0, 0, 1])
    b_u.append([0, 0, 0, 1]); b_v.append([0, 0, 0, 0, 0, 1]); b_w.append([0, 0, 0, 0, 0, 1])
    
    # Kronecker Product
    # U: 16 x Rank
    # V: 24 x Rank
    # W: 24 x Rank
    
    rank = 7 * 12
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    for i in range(7):
        for j in range(12):
            # u_kronecker = s_u[i] x b_u[j]
            # v_kronecker = s_v[i] x b_v[j]
            # w_kronecker = s_w[i] x b_w[j]
            
            # s_u[i] is 4x1, b_u[j] is 4x1 -> 16x1
            for a in range(4):
                for b in range(4):
                    U[a * 4 + b][col] = s_u[i][a] * b_u[j][b]
                    
            # s_v[i] is 4x1, b_v[j] is 6x1 -> 24x1
            for a in range(4):
                for b in range(6):
                    V[a * 6 + b][col] = s_v[i][a] * b_v[j][b]
                    
            # s_w[i] is 4x1, b_w[j] is 6x1 -> 24x1
            for a in range(4):
                for b in range(6):
                    W[a * 6 + b][col] = s_w[i][a] * b_w[j][b]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
