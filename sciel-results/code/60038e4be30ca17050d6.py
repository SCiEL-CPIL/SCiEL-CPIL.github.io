import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Local indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Global mapping:
    # A (2x4): A[r, c] -> U[r*4+c]
    # B (4x4): B[r, c] -> V[r*4+c]
    # C (2x4): C[r, c] -> W[c*2+r] (trace convention W[k*n+i])
    
    # Define the 7 Strassen terms for P = XY
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_idx: coeff}
    # v_coeffs: dict {local_idx: coeff}
    # w_coeffs: dict {local_idx: coeff}
    
    terms = [
        # M1 = (X00+X11)(Y00+Y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (X10+X11)Y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = X00(Y01-Y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = X11(Y10-Y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (X00+X01)Y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (X10-X00)(Y00+Y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (X01-X11)(Y10+Y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # We need to compute 4 products:
    # 1. A1 * B11 -> C11 (A1: A[0:2, 0:2], B11: B[0:2, 0:2], C11: C[0:2, 0:2])
    # 2. A2 * B21 -> C21 (A2: A[0:2, 2:4], B21: B[2:4, 0:2], C21: C[0:2, 0:2])
    # 3. A1 * B12 -> C12 (A1: A[0:2, 0:2], B12: B[0:2, 2:4], C12: C[0:2, 2:4])
    # 4. A2 * B22 -> C22 (A2: A[0:2, 2:4], B22: B[2:4, 2:4], C22: C[0:2, 2:4])
    
    # Note: C1 = C11 + C21, C2 = C12 + C22
    # So we add the W contributions for C11 and C21 to the same W indices for C1.
    # And C12 and C22 to the same W indices for C2.
    
    # Helper to apply a Strassen decomposition
    # a_row_off: 0 (rows 0-1 of A)
    # a_col_off: 0 or 2 (cols 0-1 or 2-3 of A)
    # b_row_off: 0 or 2 (rows 0-1 or 2-3 of B)
    # b_col_off: 0 or 2 (cols 0-1 or 2-3 of B)
    # c_col_off: 0 or 2 (cols 0-1 or 2-3 of C)
    # start_col: starting column index in U, V, W
    
    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            
            # Map U
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_col_off + c
                g_idx = g_r * m + g_c
                U[g_idx, col] += coeff
                
            # Map V
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off + r
                g_c = b_col_off + c
                g_idx = g_r * p + g_c
                V[g_idx, col] += coeff
                
            # Map W
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off + c
                # W[k*n + i] where k is col, i is row
                g_idx = g_c * n + g_r
                W[g_idx, col] += coeff

    # 1. A1 * B11
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. A2 * B21
    apply_strassen(2, 2, 0, 0, 7)
    
    # 3. A1 * B12
    apply_strassen(0, 0, 2, 2, 14)
    
    # 4. A2 * B22
    apply_strassen(2, 2, 2, 2, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
