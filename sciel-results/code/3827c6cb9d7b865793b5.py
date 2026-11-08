import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The known minimum rank for 2x4x4 matrix multiplication is 28.
    # This is achieved by applying the Strassen 2x2x2 decomposition (rank 7)
    # to the four 2x2 blocks of the product.
    # Total rank = 4 * 7 = 28.
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        # u_vec: list of (index, coeff)
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
            
    # Strassen 2x2 decomposition
    # P = XY, where X, Y are 2x2.
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Strassen terms defined as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_idx: coeff}
    # v_coeffs: dict {local_idx: coeff}
    # w_coeffs: dict {local_idx: coeff}
    
    def strassen_terms():
        terms = []
        
        # M1
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1}
        terms.append((u, v, w))
        
        # M2
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1}
        terms.append((u, v, w))
        
        # M3
        u = {0: 1}
        v = {1: 1, 3: -1}
        w = {1: 1, 3: 1}
        terms.append((u, v, w))
        
        # M4
        u = {3: 1}
        v = {2: 1, 0: -1}
        w = {0: 1, 2: 1}
        terms.append((u, v, w))
        
        # M5
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        terms.append((u, v, w))
        
        # M6
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        terms.append((u, v, w))
        
        # M7
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1}
        terms.append((u, v, w))
        
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Applies Strassen decomposition to a specific block of the matrix multiplication.
        
        a_offset: 0 for left half of A (cols 0,1), 1 for right half (cols 2,3)
        b_row_offset: 0 for top half of B (rows 0,1), 1 for bottom half (rows 2,3)
        b_col_offset: 0 for left half of B (cols 0,1), 1 for right half (cols 2,3)
        c_col_offset: 0 for left half of C (cols 0,1), 1 for right half (cols 2,3)
        start_col: The column index in the global U, V, W matrices to start writing
        """
        
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            
            # Map U indices
            # A is 2x4. Local 2x2 block at (0, a_offset*2)
            # Local (r, c) -> Global (r, a_offset*2 + c)
            # Global index in U: r*m + c = r*4 + c
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            # Map V indices
            # B is 4x4. Local 2x2 block at (b_row_offset*2, b_col_offset*2)
            # Local (r, c) -> Global (b_row_offset*2 + r, b_col_offset*2 + c)
            # Global index in V: r*p + c = r*4 + c
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            # Map W indices
            # C is 2x4. Local 2x2 block at (0, c_col_offset*2)
            # Local (r, c) -> Global (r, c_col_offset*2 + c)
            # W follows trace convention W[k*n + i], where k is col, i is row.
            # Global (row, col) -> index = col * n + row = col * 2 + row
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    # Apply Strassen to the 4 blocks:
    # 1. C[0:2, 0:2] = A[0:2, 0:2] @ B[0:2, 0:2]
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. C[0:2, 0:2] += A[0:2, 2:4] @ B[2:4, 0:2]
    # Note: The current program adds these to the SAME C block?
    # No, matrix multiplication is sum over j.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we split A into A_L (cols 0,1) and A_R (cols 2,3)
    # And B into B_T (rows 0,1) and B_B (rows 2,3)
    # C = A_L B_T + A_R B_B
    # So C_L (cols 0,1) = A_L B_T(:, 0:2) + A_R B_B(:, 0:2)
    # C_R (cols 2,3) = A_L B_T(:, 2:4) + A_R B_B(:, 2:4)
    
    # The current program's `apply_strassen` adds to the W matrix at specific columns.
    # Since W is sparse per column, and we are using distinct columns for each block,
    # we need to ensure the W indices don't conflict or that they are summed correctly.
    # In the current implementation, `add_term` does `W[idx, col] += coeff`.
    # If two different Strassen blocks contribute to the same global W index in the SAME column, they add up.
    # But here, each `apply_strassen` call uses a distinct range of columns (0-6, 7-13, 14-20, 21-27).
    # So the W entries for a given global index (i,k) will be spread across different columns.
    # This is correct for the decomposition sum: C = sum_c W_c M_c.
    
    # Block 2: A_R (a_offset=1) * B_B (b_row=1, b_col=0) contributes to C_L (c_col=0)
    apply_strassen(1, 1, 0, 0, 7)
    
    # Block 3: A_L (a_offset=0) * B_T (b_row=0, b_col=1) contributes to C_R (c_col=1)
    apply_strassen(0, 0, 1, 1, 14)
    
    # Block 4: A_R (a_offset=1) * B_B (b_row=1, b_col=1) contributes to C_R (c_col=1)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

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
