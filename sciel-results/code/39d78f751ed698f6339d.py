import numpy as np
import time

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 140

def _validate_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rng = np.random.RandomState(42)
    for trial in range(20):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            A_part = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    A_part[i, j] = u_r[i * m + j]
            
            B_part = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    B_part[j, k] = v_r[j * p + k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w_r[k * n + i] * (A_part * B_part).sum()
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 68 decomposition
    # A (3x3), B (3x8) -> C (3x8)
    # Split:
    # A = [A11 A12; A21 A22] (2x2, 2x1; 1x2, 1x1)
    # B = [B11 B12; B21 B22] (2x4, 2x4; 1x4, 1x4)
    # C = [C11 C12; C21 C22] (2x4, 2x4; 1x4, 1x4)
    
    # We will build lists of (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w)
    # But U, V, W are matrices. We can just append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
        """
        u_indices: list of indices in U (i*m+j)
        v_indices: list of indices in V (j*p+k)
        w_indices: list of indices in W (k*n+i)
        u_coeffs: list of coefficients
        v_coeffs: list of coefficients
        w_coeffs: list of coefficients
        """
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, coeff in zip(u_indices, u_coeffs):
            u_vec[idx] = coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            v_vec[idx] = coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            w_vec[idx] = coeff
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to get U index
    def ui(i, j): return i * m + j
    # Helper to get V index
    def vi(j, k): return j * p + k
    # Helper to get W index
    def wi(k, i): return k * n + i
    
    # Strassen 2x2 * 2x2
    # Local indices 0,1 for rows/cols
    # Global mapping for A11 (top-left 2x2 of A)
    # A[0,0]->ui(0,0), A[0,1]->ui(0,1), A[1,0]->ui(1,0), A[1,1]->ui(1,1)
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds Strassen terms for A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2] * 
        B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2] -> 
        C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        """
        # M1 = (A11 + A22) * (B11 + B22)
        add_term(
            [ui(a_row_offset, a_col_offset), ui(a_row_offset+1, a_col_offset+1)],
            [vi(b_row_offset, b_col_offset), vi(b_row_offset+1, b_col_offset+1)],
            [wi(c_col_offset, c_row_offset), wi(c_col_offset+1, c_row_offset+1)],
            [1, 1],
            [1, 1],
            [1, 1] # C11 gets +M1, C22 gets +M1
        )
        # M2 = (A21 + A22) * B11
        add_term(
            [ui(a_row_offset+1, a_col_offset), ui(a_row_offset+1, a_col_offset+1)],
            [vi(b_row_offset, b_col_offset)],
            [wi(c_col_offset, c_row_offset+1), wi(c_col_offset+1, c_row_offset+1)],
            [1, 1],
            [1],
            [1, -1] # C21 gets +M2, C22 gets -M2
        )
        # M3 = A11 * (B21 - B22)
        add_term(
            [ui(a_row_offset, a_col_offset)],
            [vi(b_row_offset+1, b_col_offset), vi(b_row_offset+1, b_col_offset+1)],
            [wi(c_col_offset+1, c_row_offset), wi(c_col_offset+1, c_row_offset+1)],
            [1],
            [1, -1],
            [1, 1] # C12 gets +M3, C22 gets +M3
        )
        # M4 = A22 * (B12 - B11)
        add_term(
            [ui(a_row_offset+1, a_col_offset+1)],
            [vi(b_row_offset, b_col_offset+1), vi(b_row_offset, b_col_offset)],
            [wi(c_col_offset, c_row_offset), wi(c_col_offset, c_row_offset+1)],
            [1],
            [1, -1],
            [1, 1] # C11 gets +M4, C21 gets +M4
        )
        # M5 = (A11 + A12) * B22
        add_term(
            [ui(a_row_offset, a_col_offset), ui(a_row_offset, a_col_offset+1)],
            [vi(b_row_offset+1, b_col_offset+1)],
            [wi(c_col_offset, c_row_offset), wi(c_col_offset+1, c_row_offset)],
            [1, 1],
            [1],
            [-1, 1] # C11 gets -M5, C12 gets +M5
        )
        # M6 = (A21 - A11) * B11
        add_term(
            [ui(a_row_offset+1, a_col_offset), ui(a_row_offset, a_col_offset)],
            [vi(b_row_offset, b_col_offset)],
            [wi(c_col_offset+1, c_row_offset+1)],
            [1, -1],
            [1],
            [1] # C22 gets +M6
        )
        # M7 = (A12 - A22) * B21
        add_term(
            [ui(a_row_offset, a_col_offset+1), ui(a_row_offset+1, a_col_offset+1)],
            [vi(b_row_offset+1, b_col_offset)],
            [wi(c_col_offset, c_row_offset), wi(c_col_offset+1, c_row_offset)],
            [1, -1],
            [1],
            [1, 1] # C11 gets +M7, C12 gets +M7 -- Wait.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # M7 is only in C11.
        )
        # Correction for M7:
        # My add_term for M7 above had w_indices [wi(c_col_offset, c_row_offset), wi(c_col_offset+1, c_row_offset)]
        # This corresponds to C11 and C12?
        # wi(k, i).
        # c_col_offset, c_row_offset -> C[c_row_offset, c_col_offset] which is C11.
        # c_col_offset+1, c_row_offset -> C[c_row_offset, c_col_offset+1] which is C12.
        # But M7 is only in C11.
        # So w_indices should be [wi(c_col_offset, c_row_offset)] and w_coeffs [1].
        
        # Let's re-verify M7 formula: C11 = ... + M7.
        # So I need to fix the M7 call.
        
    # Re-implementing add_strassen_block with correct W indices
    
    def add_strassen_block_v2(a_r, a_c, b_r, b_c, c_r, c_c):
        # M1
        add_term(
            [ui(a_r, a_c), ui(a_r+1, a_c+1)],
            [vi(b_r, b_c), vi(b_r+1, b_c+1)],
            [wi(c_c, c_r), wi(c_c+1, c_r+1)],
            [1, 1], [1, 1], [1, 1]
        )
        # M2
        add_term(
            [ui(a_r+1, a_c), ui(a_r+1, a_c+1)],
            [vi(b_r, b_c)],
            [wi(c_c, c_r+1), wi(c_c+1, c_r+1)],
            [1, 1], [1], [1, -1]
        )
        # M3
        add_term(
            [ui(a_r, a_c)],
            [vi(b_r+1, b_c), vi(b_r+1, b_c+1)],
            [wi(c_c+1, c_r), wi(c_c+1, c_r+1)],
            [1], [1, -1], [1, 1]
        )
        # M4
        add_term(
            [ui(a_r+1, a_c+1)],
            [vi(b_r, b_c+1), vi(b_r, b_c)],
            [wi(c_c, c_r), wi(c_c, c_r+1)],
            [1], [1, -1], [1, 1]
        )
        # M5
        add_term(
            [ui(a_r, a_c), ui(a_r, a_c+1)],
            [vi(b_r+1, b_c+1)],
            [wi(c_c, c_r), wi(c_c+1, c_r)],
            [1, 1], [1], [-1, 1]
        )
        # M6
        add_term(
            [ui(a_r+1, a_c), ui(a_r, a_c)],
            [vi(b_r, b_c)],
            [wi(c_c+1, c_r+1)],
            [1, -1], [1], [1]
        )
        # M7
        add_term(
            [ui(a_r, a_c+1), ui(a_r+1, a_c+1)],
            [vi(b_r+1, b_c)],
            [wi(c_c, c_r)],
            [1, -1], [1], [1]
        )
        
    # Apply Strassen to A11 * B11 (cols 0,1 of B11)
    # A11 is A[0:2, 0:2]. B11 is B[0:2, 0:4].
    # B11 is 2x4. We split it into B11_L (cols 0,1) and B11_R (cols 2,3).
    add_strassen_block_v2(0, 0, 0, 0, 0, 0) # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_block_v2(0, 0, 0, 2, 0, 2) # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    
    # Apply Strassen to A11 * B12 (cols 4,5 and 6,7)
    add_strassen_block_v2(0, 0, 0, 4, 0, 4) # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    add_strassen_block_v2(0, 0, 0, 6, 0, 6) # A[0:2,0:2] * B[0:2,6:8] -> C[0:2,6:8]
    
    # Cross terms
    # A12 * B21: A[0:2, 2] * B[2, 0:4]
    for i in range(2):
        for k in range(4):
            add_term(
                [ui(i, 2)],
                [vi(2, k)],
                [wi(k, i)],
                [1], [1], [1]
            )
            
    # A12 * B22: A[0:2, 2] * B[2, 4:8]
    for i in range(2):
        for k in range(4, 8):
            add_term(
                [ui(i, 2)],
                [vi(2, k)],
                [wi(k, i)],
                [1], [1], [1]
            )
            
    # A21 * B11: A[2, 0:2] * B[0:2, 0:4]
    for j in range(2):
        for k in range(4):
            add_term(
                [ui(2, j)],
                [vi(j, k)],
                [wi(k, 2)],
                [1], [1], [1]
            )
            
    # A21 * B12: A[2, 0:2] * B[0:2, 4:8]
    for j in range(2):
        for k in range(4, 8):
            add_term(
                [ui(2, j)],
                [vi(j, k)],
                [wi(k, 2)],
                [1], [1], [1]
            )
            
    # A22 * B21: A[2, 2] * B[2, 0:4]
    for k in range(4):
        add_term(
            [ui(2, 2)],
            [vi(2, k)],
            [wi(k, 2)],
            [1], [1], [1]
        )
        
    # A22 * B22: A[2, 2] * B[2, 4:8]
    for k in range(4, 8):
        add_term(
            [ui(2, 2)],
            [vi(2, k)],
            [wi(k, 2)],
            [1], [1], [1]
        )
        
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    # Validation
    if not _validate_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
                    if time.time() - start_time > DEADLINE_SECONDS:
                        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
