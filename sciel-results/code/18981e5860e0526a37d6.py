import numpy as np

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n
    rank = 56
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen forms for 2x2
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    
    # L_s(A) coefficients
    # 1: A00 + A11
    # 2: A10 + A11
    # 3: A00
    # 4: A11
    # 5: A00 + A01
    # 6: A10 - A00
    # 7: A01 - A11
    
    L_A = [
        [(0, 1), (1, 1)], # 1
        [(1, 1), (1, 1)], # 2 (A10+A11) -> (1,0) is A10? No.
        # Let's use (i, l)
        # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
    ]
    
    # Correct L_A
    L_A = [
        { (0,0): 1, (1,1): 1 }, # 1
        { (1,0): 1, (1,1): 1 }, # 2
        { (0,0): 1 },            # 3
        { (1,1): 1 },            # 4
        { (0,0): 1, (0,1): 1 }, # 5
        { (1,0): 1, (0,0): -1 },# 6
        { (0,1): 1, (1,1): -1 } # 7
    ]
    
    # M_s(B) coefficients
    # B00: (0,0), B01: (0,1), B10: (1,0), B11: (1,1)
    M_B = [
        { (0,0): 1, (1,1): 1 }, # 1
        { (0,0): 1 },            # 2
        { (0,1): 1, (1,1): -1 },# 3
        { (1,0): 1, (0,0): -1 },# 4
        { (1,1): 1 },            # 5
        { (0,0): 1, (0,1): 1 }, # 6
        { (1,0): 1, (1,1): 1 }  # 7
    ]
    
    # Reconstruction coefficients W_s(i, k_local)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    W_rec = [
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }, # 1
        { (0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0 }, # 2
        { (0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0 }, # 3
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }, # 4
        { (0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0 },# 5
        { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1 }, # 6
        { (0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0 }  # 7
    ]
    # Check C11: M1 - M2 + M3 + M6
    # My W_rec for C11:
    # 1: 1
    # 2: -1
    # 3: 1
    # 6: 1
    # Others 0.
    
    # Let's re-verify W_rec
    # C00: 1, 4, -5, -7
    # C01: 3, 5
    # C10: 2, 7
    # C11: 1, -2, 3, 6
    
    W_rec = [
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }, # 1
        { (0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1 },# 2
        { (0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1 }, # 3
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }, # 4
        { (0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0 },# 5
        { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1 }, # 6
        { (0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0 }  # 7
    ]
    
    col = 0
    for j in range(4):
        for s in range(7):
            # A1 B1j
            # U
            for (ai, al), val in L_A[s].items():
                u_idx = ai * m + al
                U[u_idx, col] = val
            # V
            for (bl, bk), val in M_B[s].items():
                v_idx = bl * p + (2 * j + bk)
                V[v_idx, col] = val
            # W
            for (ci, ck), val in W_rec[s].items():
                w_idx = (2 * j + ck) * n + ci
                W[w_idx, col] = val
            col += 1
            
            # A2 B2j
            # U
            for (ai, al), val in L_A[s].items():
                u_idx = ai * m + (2 + al)
                U[u_idx, col] = val
            # V
            for (bl, bk), val in M_B[s].items():
                v_idx = (2 + bl) * p + (2 * j + bk)
                V[v_idx, col] = val
            # W
            for (ci, ck), val in W_rec[s].items():
                w_idx = (2 * j + ck) * n + ci
                W[w_idx, col] = val
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
