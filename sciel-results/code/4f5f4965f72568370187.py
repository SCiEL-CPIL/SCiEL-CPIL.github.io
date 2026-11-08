import time
import numpy as np

N, M, P = 2, 4, 4
U_DIM, V_DIM, W_DIM = N * M, M * P, P * N
RANK = 28

# The 7 Strassen multiplication formulas for 2x2 matrices.
# Each is a triple (u, v, w) where u, v, w are 4-element vectors (sparse dicts here).
# u corresponds to A entries, v to B entries, w to C entries.
# Indices 0..3 correspond to positions (0,0),(0,1),(1,0),(1,1).
_STRASSEN = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ({2: 1, 3: 1}, {0: 1},          {2: 1, 3: -1}),
    ({0: 1},          {1: 1, 3: -1}, {1: 1, 3: 1}),
    ({3: 1},          {2: 1, 0: -1}, {0: 1, 2: 1}),
    ({0: 1, 1: 1}, {3: 1},          {0: -1, 1: 1}),
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
]

def construct():
    U = np.zeros((U_DIM, RANK), dtype=float)
    V = np.zeros((V_DIM, RANK), dtype=float)
    W = np.zeros((W_DIM, RANK), dtype=float)

    # Four independent 2x2x2 Strassen blocks:
    # Block b corresponds to (a_row_off, b_row_off, b_col_off, c_col_off)
    # A has 2 rows, each of size 2x2 (since M=4 split into two 2-col halves, N=2 means 2 such blocks in A's row dim).
    # Actually, for 2x4 * 4x4:
    # A is 2x4. We can view it as 2 rows, each row is a 1x4, but for Strassen we need 2x2 blocks.
    # The current seed treats it as: A is split into 2 blocks of 2x2 vertically? No, A is 2x4.
    # Let's look at the indices:
    # U index: i*m + j. i in [0,1], j in [0,3].
    # If we map local (r, c) in 2x2 to global:
    # A block (a_row_off): local row r -> global row i = a_row_off?
    # Wait, A has N=2 rows. So a_row_off must be 0 or 1.
    # Local col c -> global col j = a_col_off * 2 + c.
    # So A is split into 4 blocks of 2x2? No, N=2, M=4.
    # A[i, j]. i=0,1. j=0,1,2,3.
    # Block 0: i=0, j=0,1. (Row 0, Cols 0-1)
    # Block 1: i=0, j=2,3. (Row 0, Cols 2-3)
    # Block 2: i=1, j=0,1. (Row 1, Cols 0-1)
    # Block 3: i=1, j=2,3. (Row 1, Cols 2-3)
    # But Strassen needs 2x2. So we need 2 rows in A.
    # The seed code uses:
    # apply_strassen(0, 0, 0, 0, 0)
    # apply_strassen(1, 1, 0, 0, 7)
    # apply_strassen(0, 0, 1, 1, 14)
    # apply_strassen(1, 1, 1, 1, 21)
    
    # Let's decode the mapping in the seed:
    # A local (r, c) -> global i, j.
    # u_global[g_idx] where g_idx = g_r * 4 + g_c.
    # g_r = r (local row).
    # g_c = a_offset * 2 + c.
    # So if a_offset=0: g_c = c (0,1). If a_offset=1: g_c = 2+c (2,3).
    # But wait, A has only 2 rows (i=0,1).
    # In the seed, `r` from local 2x2 is 0 or 1. So g_r is 0 or 1.
    # This means the local 2x2 block uses BOTH rows of A.
    # So the 2x2 block is the entire row i (0 or 1) and 2 columns.
    # But Strassen multiplies 2x2 by 2x2.
    # Here A is 2x2 (using both rows i=0,1 and 2 cols).
    # B is 2x2 (using 2 rows j and 2 cols k).
    # C is 2x2 (using 2 rows i and 2 cols k).
    
    # So the 4 blocks correspond to:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]  (Wait, seed uses a_offset=1 for second?)
    
    # Let's re-read seed mapping for Block 2 (start_col 7):
    # apply_strassen(1, 1, 0, 0, 7)
    # a_offset=1 -> A cols 2,3.
    # b_row_off=1 -> B rows 2,3.
    # b_col_off=0 -> B cols 0,1.
    # c_col_off=0 -> C cols 0,1.
    # This computes A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2].
    
    # Block 3 (start_col 14):
    # apply_strassen(0, 0, 1, 1, 14)
    # a_offset=0 -> A cols 0,1.
    # b_row_off=0 -> B rows 0,1.
    # b_col_off=1 -> B cols 2,3.
    # c_col_off=1 -> C cols 2,3.
    # This computes A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4].
    
    # Block 4 (start_col 21):
    # apply_strassen(1, 1, 1, 1, 21)
    # a_offset=1 -> A cols 2,3.
    # b_row_off=1 -> B rows 2,3.
    # b_col_off=1 -> B cols 2,3.
    # c_col_off=1 -> C cols 2,3.
    # This computes A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4].
    
    # So the 4 blocks are:
    # 1. A_L * B_TL -> C_L
    # 2. A_R * B_BL -> C_L  (Note: A_R uses cols 2-3, B_BL uses rows 2-3, cols 0-1)
    #    Wait, A is 2x4. A_L is 2x2 (cols 0-1). A_R is 2x2 (cols 2-3).
    #    B is 4x4. B_TL is 2x2 (rows 0-1, cols 0-1). B_TR is 2x2 (rows 0-1, cols 2-3).
    #    B_BL is 2x2 (rows 2-3, cols 0-1). B_BR is 2x2 (rows 2-3, cols 2-3).
    #    C is 2x4. C_L is 2x2 (cols 0-1). C_R is 2x2 (cols 2-3).
    
    # C = A * B = [A_L * B_L + A_R * B_R] where B_L is 4x2, B_R is 4x2.
    # C_L = A_L * B_TL + A_R * B_BL.
    # C_R = A_L * B_TR + A_R * B_BR.
    
    # The seed implements:
    # Block 1: A_L * B_TL -> contributes to C_L
    # Block 2: A_R * B_BL -> contributes to C_L
    # Block 3: A_L * B_TR -> contributes to C_R
    # Block 4: A_R * B_BR -> contributes to C_R
    
    # This is correct.
    
    blocks = [
        (0, 0, 0, 0, 0),   # a_off, b_row_off, b_col_off, c_col_off, start_col
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]
    
    for a_off, b_r_off, b_c_off, c_c_off, start_col in blocks:
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(_STRASSEN):
            col = start_col + t_idx
            
            # Fill U
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                # Local (r, c) in 2x2 A block
                # Global A[i, j]: i=r, j=a_off*2 + c
                g_i = r
                g_j = a_off * 2 + c
                u_idx = g_i * M + g_j
                U[u_idx, col] += coeff
                
            # Fill V
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                # Local (r, c) in 2x2 B block
                # Global B[j, k]: j=b_r_off*2 + r, k=b_c_off*2 + c
                g_j = b_r_off * 2 + r
                g_k = b_c_off * 2 + c
                v_idx = g_j * P + g_k
                V[v_idx, col] += coeff
                
            # Fill W
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                # Local (r, c) in 2x2 C block
                # Global C[i, k]: i=r, k=c_c_off*2 + c
                # W index: k*n + i = g_k * N + g_i
                g_i = r
                g_k = c_c_off * 2 + c
                w_idx = g_k * N + g_i
                W[w_idx, col] += coeff
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    A = np.arange(N * M).reshape(N, M) + 1.0
    B = np.arange(M * P).reshape(M, P) + 2.0
    C = A @ B
    Cf = np.zeros((N, P))
    
    for ii in range(N):
        for kk in range(P):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(M):
                    acc += U[ii * M + jj, cc] * V[jj * P + kk, cc] * W[kk * N + ii, cc]
            Cf[ii, kk] = acc
            
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
