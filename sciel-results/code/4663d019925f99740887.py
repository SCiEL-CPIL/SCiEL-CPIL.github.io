import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Block decomposition:
    # i: {0,1} (2 rows) and {2} (1 row)
    # j: {0,1} (2 cols) and {2,3} (2 cols)
    # k: {0,1} (2 cols) and {2,3} (2 cols)
    #
    # 4 blocks of 2x2x2: use Strassen (rank 7 each)
    # 4 blocks of 1x2x2: use trivial (rank 4 each)
    # Total: 4*7 + 4*4 = 44
    
    total_rank = 4 * 7 + 4 * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # (12, 44)
    V = np.zeros((m * p, total_rank))  # (16, 44)
    W = np.zeros((p * n, total_rank))  # (12, 44)
    
    rank_offset = 0
    
    def add_strassen_block(i_off, j_off, k_off, r_start):
        """Add Strassen decomposition for a 2x2x2 block at offsets (i_off, j_off, k_off)."""
        # Strassen's 7 products for 2x2 matrix multiplication
        # A is 2x2: A[i,j] for i,j in {0,1}
        # B is 2x2: B[j,k] for j,k in {0,1}
        # C is 2x2: C[k,i] for k,i in {0,1}
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 - M5 - M7 + M4
        # C01 = M5 + M7
        # C10 = M1 + M2 - M3 - M6
        # C11 = M1 - M2 + M3 + M6
        
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict (i,j) -> coeff for A
        # V_coeffs: dict (j,k) -> coeff for B
        # W_coeffs: dict (k,i) -> coeff for C (output)
        
        terms = [
            # M1
            ({(0,0):1, (1,1):1}, 
             {(0,0):1, (1,1):1}, 
             {(0,0):1, (1,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, 
             {(0,0):1}, 
             {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1}, 
             {(1,0):1, (1,1):-1}, 
             {(1,0):-1, (1,1):1}),
            # M4
            ({(1,1):1}, 
             {(0,1):1, (1,0):-1}, 
             {(0,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, 
             {(1,1):1}, 
             {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, 
             {(0,0):1, (1,0):1}, 
             {(1,0):-1}),
            # M7
            ({(0,1):1, (1,1):-1}, 
             {(0,1):1, (1,1):1}, 
             {(0,0):-1, (0,1):1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            r = r_start
            for (i, j), coeff in u_coeffs.items():
                U[i_off + i, j_off + j, r] = coeff
            for (j, k), coeff in v_coeffs.items():
                V[j_off + j, k_off + k, r] = coeff
            for (k, i), coeff in w_coeffs.items():
                W[k_off + k, i_off + i, r] = coeff
            r_start += 1
        
        return r_start
    
    def add_trivial_1x2x2_block(i_off, j_off, k_off, r_start):
        """Add trivial rank-4 decomposition for a 1x2x2 block."""
        for j in range(2):
            for k in range(2):
                r = r_start
                U[i_off, j_off + j, r] = 1
                V[j_off + j, k_off + k, r] = 1
                W[k_off + k, i_off, r] = 1
                r_start += 1
        return r_start
    
    # Add 4 Strassen blocks
    rank_offset = add_strassen_block(0, 0, 0, rank_offset)
    rank_offset = add_strassen_block(0, 0, 2, rank_offset)
    rank_offset = add_strassen_block(0, 2, 0, rank_offset)
    rank_offset = add_strassen_block(0, 2, 2, rank_offset)
    
    # Add 4 trivial blocks
    rank_offset = add_trivial_1x2x2_block(2, 0, 0, rank_offset)
    rank_offset = add_trivial_1x2x2_block(2, 0, 2, rank_offset)
    rank_offset = add_trivial_1x2x2_block(2, 2, 0, rank_offset)
    rank_offset = add_trivial_1x2x2_block(2, 2, 2, rank_offset)
    
    assert rank_offset == total_rank
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u_col = U[:, r_idx].reshape(n, m)
            v_col = V[:, r_idx].reshape(m, p)
            w_col = W[:, r_idx].reshape(p, n)
            T += u_col[:, :, None] * v_col[None, :, :] * w_col.T[:, None, :]
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m + j, col] = 1
                V_s[j*p + k, col] = 1
                W_s[k*n + i, col] = 1
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
