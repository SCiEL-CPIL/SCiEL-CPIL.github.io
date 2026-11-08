import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
            
    # Define Strassen terms for a generic 2x2 product P = XY
    # Returns list of 7 tuples: (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # Indices in dicts: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    def strassen_terms():
        terms = []
        
        # M1 = (X00+X11)(Y00+Y11)
        # P00 += M1, P11 += M1
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1}
        terms.append((u, v, w))
        
        # M2 = (X10+X11)Y00
        # P10 += M2, P11 -= M2
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1}
        terms.append((u, v, w))
        
        # M3 = X00(Y01-Y11)
        # P01 += M3, P11 += M3
        u = {0: 1}
        v = {1: 1, 3: -1}
        w = {1: 1, 3: 1}
        terms.append((u, v, w))
        
        # M4 = X11(Y10-Y00)
        # P00 += M4, P10 += M4
        u = {3: 1}
        v = {2: 1, 0: -1}
        w = {0: 1, 2: 1}
        terms.append((u, v, w))
        
        # M5 = (X00+X01)Y11
        # P00 -= M5, P01 += M5
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        terms.append((u, v, w))
        
        # M6 = (X10-X00)(Y00+Y01)
        # P11 += M6
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        terms.append((u, v, w))
        
        # M7 = (X01-X11)(Y10+Y11)
        # P00 += M7
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1}
        terms.append((u, v, w))
        
        return terms

    base_terms = strassen_terms()
    
    # Function to apply a set of Strassen terms to specific blocks
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        a_col_offset: 0 for A cols 0-1, 1 for A cols 2-3
        b_row_offset: 0 for B rows 0-1, 1 for B rows 2-3
        b_col_offset: 0 for B cols 0-1, 1 for B cols 2-3
        c_col_offset: 0 for C cols 0-1, 1 for C cols 2-3
        """
        
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            
            # Map U indices
            # Local (r, c) -> Global (r, a_col_offset*2 + c)
            # Global index = r * m + (a_col_offset*2 + c)
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_col_offset * 2 + c
                g_idx = g_r * m + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            # Map V indices
            # Local (r, c) -> Global (b_row_offset*2 + r, b_col_offset*2 + c)
            # Global index = (b_row_offset*2 + r) * p + (b_col_offset*2 + c)
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * p + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            # Map W indices
            # Local (r, c) -> Global (r, c_col_offset*2 + c)
            # Output C is n x p. W index convention: W[k*n + i] where k is col, i is row.
            # So if global result is C[g_r, g_c], W index is g_c * n + g_r.
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * n + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    # We need to compute C = A @ B
    # A is 2x4, B is 4x4, C is 2x4
    # We split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # We split B into 4 blocks of 2x2:
    # B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3)
    # B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    
    # C1 (cols 0-1) = A1 @ B11 + A2 @ B21
    # C2 (cols 2-3) = A1 @ B12 + A2 @ B22
    
    # 1. C1 Part 1: A1 @ B11
    # A1: a_col_offset=0. B11: b_row_offset=0, b_col_offset=0. C1: c_col_offset=0.
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. C1 Part 2: A2 @ B21
    # A2: a_col_offset=1. B21: b_row_offset=1, b_col_offset=0. C1: c_col_offset=0.
    apply_strassen(1, 1, 0, 0, 7)
    
    # 3. C2 Part 1: A1 @ B12
    # A1: a_col_offset=0. B12: b_row_offset=0, b_col_offset=1. C2: c_col_offset=1.
    apply_strassen(0, 0, 1, 1, 14)
    
    # 4. C2 Part 2: A2 @ B22
    # A2: a_col_offset=1. B22: b_row_offset=1, b_col_offset=1. C2: c_col_offset=1.
    apply_strassen(1, 1, 1, 1, 21)
    
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
        print("FAILED")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
