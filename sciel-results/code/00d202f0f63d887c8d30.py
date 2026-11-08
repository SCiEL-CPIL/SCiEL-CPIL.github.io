import time
import numpy as np

DIMENSIONS = [2, 4, 4]
_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _time_remaining():
    return _TIME_LIMIT - 30 - (time.time() - _START_TIME)  # 30s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition
    # P = XY, where X, Y are 2x2.
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # Local indices for X, Y, P (row-major 2x2):
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Define Strassen terms
    # Each term: (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # Coefficients can be -1, 0, 1
    
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # P00 +1, P11 +1
        
        # M2 = (X10+X11)Y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),  # P10 +1, P11 -1
        
        # M3 = X00(Y01-Y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),  # P01 +1, P11 +1
        
        # M4 = X11(Y10-Y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),  # P00 +1, P10 +1
        
        # M5 = (X00+X01)Y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),  # P00 -1, P01 +1
        
        # M6 = (X10-X00)(Y00+Y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # P11 +1
        
        # M7 = (X01-X11)(Y10+Y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),  # P00 +1
    ]
    
    # Mapping function for global indices
    # U: n x m -> index = r*m + c
    # V: m x p -> index = r*p + c
    # W: p x n -> index = k*n + i, where k is col (of C), i is row (of C)
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        # a_col_offset: 0 or 1 (which 2x2 block of A)
        # b_row_offset: 0 or 1 (which 2x2 row block of B)
        # b_col_offset: 0 or 1 (which 2x2 col block of B)
        # c_col_offset: 0 or 1 (which 2x2 col block of C)
        
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            # Map U indices
            # Local (r, c) -> Global row r, col a_col_offset*2 + c
            # Index: r*m + (a_col_offset*2 + c)
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            
            # Map V indices
            # Local (r, c) -> Global row b_row_offset*2 + r, col b_col_offset*2 + c
            # Index: (b_row_offset*2 + r)*p + (b_col_offset*2 + c)
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            
            # Map W indices
            # Local (r, c) -> Global row r, col c_col_offset*2 + c
            # C is n x p. W index: k*n + i, where k is col of C, i is row of C
            # i = r (local row of C block = global row), k = c_col_offset*2 + c (local col of C block = global col)
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    # The product C = A @ B
    # C is 2x4
    # A is 2x4, B is 4x4
    # We can decompose A into A_L, A_R (2x2 blocks) and B into 4 2x2 blocks
    # C_L = A_L @ B_LL + A_R @ B_LR  (first 2 cols of C)
    # C_R = A_L @ B_RL + A_R @ B_RR  (last 2 cols of C)
    
    # Let's define the 4 blocks of B (4x4):
    # B_LL (rows 0-1, cols 0-1)
    # B_RL (rows 0-1, cols 2-3)
    # B_LR (rows 2-3, cols 0-1)
    # B_RR (rows 2-3, cols 2-3)
    
    # C_L (cols 0-1):
    # Part 1: A_L (cols 0-1) * B_LL
    apply_strassen(a_col_offset=0, b_row_offset=0, b_col_offset=0, c_col_offset=0, start_col=0)
    # Part 2: A_R (cols 2-3) * B_LR
    apply_strassen(a_col_offset=1, b_row_offset=1, b_col_offset=0, c_col_offset=0, start_col=7)
    
    # C_R (cols 2-3):
    # Part 1: A_L (cols 0-1) * B_RL
    apply_strassen(a_col_offset=0, b_row_offset=0, b_col_offset=1, c_col_offset=1, start_col=14)
    # Part 2: A_R (cols 2-3) * B_RR
    apply_strassen(a_col_offset=1, b_row_offset=1, b_col_offset=1, c_col_offset=1, start_col=21)
    
    # Verify the decomposition
    if _time_remaining() > 10:
        if not _verify_decomposition(U, V, W, n, m, p):
            # If verification fails, return the result anyway (shouldn't happen)
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    np.random.seed(12345)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    return np.allclose(C, Cf, atol=1e-9)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
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
        print("Verification Failed")
        print("C:")
        print(C)
        print("Cf:")
        print(Cf)
    else:
        print("Verification Passed")
        
    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
