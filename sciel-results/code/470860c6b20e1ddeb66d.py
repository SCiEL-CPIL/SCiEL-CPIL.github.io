import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    # Strassen decomposition for 2x2 * 2x2
    # Local indices for A, B, C (2x2 matrices):
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Each entry in strassen_cols is a dict mapping U_idx, V_idx, C_idx to list of (local_idx, coeff)
    # Format: {0: [(local_A_idx, coeff), ...], 1: [(local_B_idx, coeff), ...], 2: [(local_C_idx, coeff), ...]}
    
    strassen_cols = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        {0: [(0,1), (3,1)], 1: [(0,1), (3,1)], 2: [(0,1), (3,1)]},
        # M2: (A10+A11)B00 -> C10-C11
        {0: [(2,1), (3,1)], 1: [(0,1)], 2: [(2,1), (3,-1)]},
        # M3: A00(B01-B11) -> C01+C11
        {0: [(0,1)], 1: [(1,1), (3,-1)], 2: [(1,1), (3,1)]},
        # M4: A11(B10-B00) -> C00+C10
        {0: [(3,1)], 1: [(2,1), (0,-1)], 2: [(0,1), (2,1)]},
        # M5: (A00+A01)B11 -> -C00+C01
        {0: [(0,1), (1,1)], 1: [(3,1)], 2: [(0,-1), (1,1)]},
        # M6: (A10-A00)(B00+B01) -> C11
        {0: [(2,1), (0,-1)], 1: [(0,1), (1,1)], 2: [(3,1)]},
        # M7: (A01-A11)(B10+B11) -> C00
        {0: [(1,1), (3,-1)], 1: [(2,1), (3,1)], 2: [(0,1)]}
    ]
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n*m, rank), dtype=int)  # (8, 28)
    V = np.zeros((m*p, rank), dtype=int)  # (16, 28)
    W = np.zeros((p*n, rank), dtype=int)  # (8, 28)
    
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3)
    # B is 4x4. Split into 2x2 blocks:
    #   B_TL (rows 0-1, cols 0-1), B_TR (rows 0-1, cols 2-3)
    #   B_BL (rows 2-3, cols 0-1), B_BR (rows 2-3, cols 2-3)
    # C is 2x4. Output convention: W[k*n+i] where k=col, i=row
    # C_L = A_L * B_TL + A_R * B_BL  (cols 0,1)
    # C_R = A_L * B_TR + A_R * B_BR  (cols 2,3)
    
    def map_A(local_idx, offset):
        # local_idx 0-3 -> 2x2 block
        # offset 0 -> A_L (cols 0,1), offset 1 -> A_R (cols 2,3)
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (offset * 2 + c)
    
    def map_B(local_idx, row_off, col_off):
        # local_idx 0-3 -> 2x2 block
        # row_off 0 -> rows 0-1, row_off 1 -> rows 2-3
        # col_off 0 -> cols 0-1, col_off 1 -> cols 2-3
        r = local_idx // 2
        c = local_idx % 2
        return (row_off * 2 + r) * 4 + (col_off * 2 + c)
    
    def map_C(local_idx, col_off):
        # Output C is 2x4, W[k*n+i] where k=col, i=row
        # local_idx 0-3 -> 2x2 output block
        # col_off 0 -> cols 0,1, col_off 1 -> cols 2,3
        r = local_idx // 2
        c = local_idx % 2
        return (col_off * 2 + c) * 2 + r
    
    # Block 1: A_L * B_TL -> C_L part 1 (cols 0-1 of C)
    for j, col_def in enumerate(strassen_cols):
        col_idx = j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 0, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 0), col_idx] += coeff
    
    # Block 2: A_R * B_BL -> C_L part 2 (cols 0-1 of C)
    for j, col_def in enumerate(strassen_cols):
        col_idx = 7 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 1, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 0), col_idx] += coeff
    
    # Block 3: A_L * B_TR -> C_R part 1 (cols 2-3 of C)
    for j, col_def in enumerate(strassen_cols):
        col_idx = 14 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 0, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 1), col_idx] += coeff
    
    # Block 4: A_R * B_BR -> C_R part 2 (cols 2-3 of C)
    for j, col_def in enumerate(strassen_cols):
        col_idx = 21 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 1, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 1), col_idx] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
