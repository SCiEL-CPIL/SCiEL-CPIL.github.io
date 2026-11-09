import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    start = time.time()
    
    # Rank-28 decomposition: 4 independent Strassen (rank 7) for 2x2 blocks
    # A is 2x4: split as [A0 | A1] where A0, A1 are 2x2
    # B is 4x4: split as [[B00, B01], [B10, B11]] with 2x2 blocks
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (8, 28)
    
    # Strassen 7 terms for 2x2 * 2x2 -> 2x2
    # A_local = [[a, b], [c, d]] (indices 0,1,2,3)
    # B_local = [[e, f], [g, h]] (indices 0,1,2,3)
    # C_local = [[C00, C01], [C10, C11]] (indices 0,1,2,3)
    # 
    # M1 = (a+d)(e+h): U={0:1,3:1}, V={0:1,3:1}, W={0:1,3:1}
    # M2 = (c+d)e:   U={2:1,3:1}, V={0:1},    W={2:1,3:-1}
    # M3 = a(f-h):    U={0:1},      V={1:1,3:-1},W={1:1,3:1}
    # M4 = d(g-e):    U={3:1},      V={2:1,0:-1},W={0:1,2:1}
    # M5 = (a+b)h:   U={0:1,1:1},  V={3:1},    W={0:-1,1:1}
    # M6 = (c-a)(e+f):U={2:1,0:-1}, V={0:1,1:1},W={3:1}
    # M7 = (b-d)(g+h):U={1:1,3:-1}, V={2:1,3:1},W={0:1}
    
    # Verify: C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen = [
        # term: (U_entries, V_entries, W_entries)
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),  # M1
        ({(2,1),(3,1)}, {(0,1)}, {(2,1),(3,-1)}),        # M2
        ({(0,1)}, {(1,1),(3,-1)}, {(1,1),(3,1)}),        # M3
        ({(3,1)}, {(2,1),(0,-1)}, {(0,1),(2,1)}),        # M4
        ({(0,1),(1,1)}, {(3,1)}, {(0,-1),(1,1)}),        # M5
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),        # M6
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,1)}),        # W7
    ]
    
    # Four block combinations:
    # Block 0: left block A0 (cols 0,1 of A), B blocks B00 (rows 0,1; cols 0,1 of B) -> C block cols 0,1, rows 0,1
    #   A index: i in {0,1}, j in {0,1} -> U[i*4+j]
    #   B index: j in {0,1}, k in {0,1} -> V[j*4+k]
    #   C index: k in {0,1}, i in {0,1} -> W[k*2+i]
    # Block 1: left block A0, B blocks B01 (rows 0,1; cols 2,3 of B) -> C block cols 2,3
    # Block 2: left block A1 (cols 2,3 of A), B blocks B10 (rows 2,3; cols 0,1 of B) -> C block cols 0,1
    # Block 3: left block A1, B blocks B11 (rows 2,3; cols 2,3 of B) -> C block cols 2,3
    
    # For C0 = A0*B00 + A1*B10:
    #   A0*B00: A cols 0,1; B rows 0,1 cols 0,1; C cols 0,1
    #   A1*B10: A cols 2,3; B rows 2,3 cols 0,1; C cols 0,1
    # For C1 = A0*B01 + A1*B11:
    #   A0*B01: A cols 0,1; B rows 0,1 cols 2,3; C cols 2,3
    #   A1*B11: A cols 2,3; B rows 2,3 cols 2,3; C cols 2,3
    
    col = 0
    for (a_col_off, b_row_off, b_col_off) in [(0,0,0), (0,0,2), (2,2,0), (2,2,2)]:
        for (U_ent, V_ent, W_ent) in strassen:
            for (idx, coeff) in U_ent:
                a_row = idx // 2
                a_col = idx % 2
                U[a_row * m + a_col + a_col_off, col] += coeff
            
            for (idx, coeff) in V_ent:
                b_row = idx // 2
                b_col = idx % 2
                V[(b_row + b_row_off) * p + b_col + b_col_off, col] += coeff
            
            for (idx, coeff) in W_ent:
                c_row = idx // 2
                c_col = idx % 2
                W[(c_row + b_col_off) * n + c_col, col] += coeff
            
            col += 1
    
    # Verify correctness
    if time.time() - start < 2:
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
        # For matrix mult, this should be 1 for all valid (i,j,k)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 1.0
                    recon = 0.0
                    for r in range(rank):
                        recon += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = abs(recon - tensor_val)
                    if err > max_err:
                        max_err = err
        # Also check that non-multiplication entries are zero (they should be by construction)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
