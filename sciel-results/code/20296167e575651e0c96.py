import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # We use a rank-28 decomposition based on 4 independent Strassen 2x2 block multiplications.
    # Block 1: A[:, 0:2] * B[0:2, :] -> C[:, 0:2]
    # Block 2: A[:, 2:4] * B[2:4, :] -> C[:, 2:4]
    # Block 3: A[:, 0:2] * B[0:2, :] -> C[:, 0:2] (Wait, B is 4x4, so rows 0:2 are top half)
    # Let's be precise.
    # A is 2x4. B is 4x4.
    # We can partition B into two 2x4 matrices: B_top (rows 0,1) and B_bottom (rows 2,3).
    # Then C = A[:, 0:2] * B_top[:, 0:2] + A[:, 2:4] * B_bottom[:, 0:2] + A[:, 0:2] * B_top[:, 2:4] + A[:, 2:4] * B_bottom[:, 2:4]
    # This decomposes into 4 independent 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7. Total rank = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen formulas for 2x2 matrix multiplication P = XY
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # P = [[p00, p01], [p10, p11]]
    #
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)(y00)
    # M3 = (x00)(y01 - y11)
    # M4 = (x11)(y10 - y00)
    # M5 = (x00 + x01)(y11)
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    #
    # p00 = M1 + M4 - M5 + M7
    # p01 = M3 + M5
    # p10 = M2 + M4
    # p11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: indices in X (00, 01, 10, 11) -> 0, 1, 2, 3
    # v_coeffs: indices in Y (00, 01, 10, 11) -> 0, 1, 2, 3
    # w_coeffs: indices in P (00, 01, 10, 11) -> 0, 1, 2, 3
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # Helper to add a term to the global matrices
    def add_term(u_map, v_map, w_map, col):
        for idx, coeff in u_map:
            U[idx, col] += coeff
        for idx, coeff in v_map:
            V[idx, col] += coeff
        for idx, coeff in w_map:
            W[idx, col] += coeff
    
    # Define blocks
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # A indices: 0,1,4,5 (rows 0,1; cols 0,1)
    # B indices: 0,1,4,5 (rows 0,1; cols 0,1)
    # C indices: W[0:4] (cols 0,1; rows 0,1) -> W[k*n+i] -> W[0*2+0], W[0*2+1], W[1*2+0], W[1*2+1]
    
    # Block 2: A[:, 0:2] * B[2:4, 0:2] -> C[:, 0:2]
    # A indices: 0,1,4,5
    # B indices: 8,9,12,13 (rows 2,3; cols 0,1)
    # C indices: W[0:4]
    
    # Block 3: A[:, 2:4] * B[0:2, 2:4] -> C[:, 2:4]
    # A indices: 2,3,6,7 (rows 0,1; cols 2,3)
    # B indices: 2,3,6,7 (rows 0,1; cols 2,3)
    # C indices: W[4:8] (cols 2,3; rows 0,1) -> W[2*2+0], W[2*2+1], W[3*2+0], W[3*2+1]
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # A indices: 2,3,6,7
    # B indices: 10,11,14,15 (rows 2,3; cols 2,3)
    # C indices: W[4:8]
    
    def map_block(a_offset_col, b_offset_row, b_offset_col, c_offset_col, start_col):
        # a_offset_col: 0 or 2 (column offset in A)
        # b_offset_row: 0 or 2 (row offset in B)
        # b_offset_col: 0 or 2 (column offset in B)
        # c_offset_col: 0 or 2 (column offset in C)
        
        for t_idx in range(7):
            u_local, v_local, w_local = strassen_terms[t_idx]
            col = start_col + t_idx
            
            # Map U
            u_global = []
            for loc_idx, coeff in u_local:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset_col + c
                g_idx = g_r * m + g_c
                u_global.append((g_idx, coeff))
            
            # Map V
            v_global = []
            for loc_idx, coeff in v_local:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_offset_row + r
                g_c = b_offset_col + c
                g_idx = g_r * p + g_c
                v_global.append((g_idx, coeff))
            
            # Map W
            # W index is k*n + i, where k is col, i is row.
            # Local P index loc_idx corresponds to row r, col c.
            # Global col is c_offset_col + c.
            # Global row is r.
            w_global = []
            for loc_idx, coeff in w_local:
                r = loc_idx // 2
                c = loc_idx % 2
                g_c = c_offset_col + c
                g_r = r
                g_idx = g_c * n + g_r
                w_global.append((g_idx, coeff))
            
            add_term(u_global, v_global, w_global, col)
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    map_block(0, 0, 0, 0, 0)
    
    # Block 2: A[:, 0:2] * B[2:4, 0:2] -> C[:, 0:2]
    map_block(0, 2, 0, 0, 7)
    
    # Block 3: A[:, 2:4] * B[0:2, 2:4] -> C[:, 2:4]
    map_block(2, 0, 2, 2, 14)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    map_block(2, 2, 2, 2, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randint(1, 10, (n, m)).astype(float)
    B = np.random.randint(1, 10, (m, p)).astype(float)
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
