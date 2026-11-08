import time
import numpy as np

DIMENSIONS = [2, 4, 4]
_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _time_remaining():
    return _TIME_LIMIT - 30 - (time.time() - _START_TIME)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Each term: (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # Local indices for 2x2 blocks (row-major): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11) -> P00 +1, P11 +1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (X10+X11)Y00 -> P10 +1, P11 -1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = X00(Y01-Y11) -> P01 +1, P11 +1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = X11(Y10-Y00) -> P00 +1, P10 +1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (X00+X01)Y11 -> P00 -1, P01 +1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (X10-X00)(Y00+Y01) -> P11 +1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (X01-X11)(Y10+Y11) -> P00 +1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    # C = A @ B where A is 2x4, B is 4x4, C is 2x4
    # Split into 2x2 blocks:
    # A_L (cols 0-1), A_R (cols 2-3)
    # B_LL (rows 0-1, cols 0-1), B_RL (rows 0-1, cols 2-3)
    # B_LR (rows 2-3, cols 0-1), B_RR (rows 2-3, cols 2-3)
    
    # C_L (cols 0-1) = A_L @ B_LL + A_R @ B_LR
    apply_strassen(a_col_offset=0, b_row_offset=0, b_col_offset=0, c_col_offset=0, start_col=0)
    apply_strassen(a_col_offset=1, b_row_offset=1, b_col_offset=0, c_col_offset=0, start_col=7)
    
    # C_R (cols 2-3) = A_L @ B_RL + A_R @ B_RR
    apply_strassen(a_col_offset=0, b_row_offset=0, b_col_offset=1, c_col_offset=1, start_col=14)
    apply_strassen(a_col_offset=1, b_row_offset=1, b_col_offset=1, c_col_offset=1, start_col=21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
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
