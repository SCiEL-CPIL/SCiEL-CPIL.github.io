import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin from 150s

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Rank 42 solution using 3 blocks of rank 14 (M_{2,4,2})
    # M_{2,4,2} is computed as 2 Strassen-like blocks of 2x2?
    # Actually, M_{2,4,2} rank 14 is derived from 2 blocks of M_{2,2,2} (rank 7)?
    # No, M_{2,4,2} is 2x4 * 4x2.
    # Split A (2x4) into A1, A2 (2x2). Split B (4x2) into B1, B2 (2x2).
    # C = A1 B1 + A2 B2.
    # This is the sum of two 2x2 matrix multiplications.
    # Rank is 7 + 7 = 14.
    # So for M_{2,4,6}, we have 3 such blocks.
    # Total rank 3 * 14 = 42.

    # Strassen coefficients for 2x2 matrix multiplication
    # M = X Y
    # m1 = (X11 + X22)(Y11 + Y22)
    # m2 = (X21 + X22) Y11
    # m3 = X11 (Y12 - Y22)
    # m4 = X22 (Y21 - Y11)
    # m5 = (X11 + X12) Y22
    # m6 = (X21 - X11)(Y11 + Y12)
    # m7 = (X12 - X22)(Y21 + Y22)
    
    # We need to construct the tensor for C = A1 B1 + A2 B2
    # Let's define the 14 multiplication steps for one block.
    # Step 1-7 for A1 B1
    # Step 8-14 for A2 B2
    
    # U coefficients for A1 (indices 0,1,2,3 in 2x2)
    # A1 is top-left 2x2 of A.
    # A2 is bottom-left 2x2 of A.
    # B1 is top-right 2x2 of B (first 2 cols).
    # B2 is bottom-right 2x2 of B (first 2 cols).
    # Wait, B is 4x6.
    # Block 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-1) + A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-1)
    # This produces C1 (rows 0-1, cols 0-1).
    
    # We will build the full U, V, W matrices.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen patterns for X Y -> M
    # X indices: 0,1,2,3 (row-major)
    # Y indices: 0,1,2,3
    # M indices: 0,1,2,3
    
    # Define the 7 Strassen terms
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (idx, coeff)
    
    strassen_terms = [
        # m1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # m2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # m3
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1), (2, 1)]),
        # m4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (1, 1)]),
        # m5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # m6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, -1)]),
        # m7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    # 3 blocks for the 3 pairs of columns in C (0-1, 2-3, 4-5)
    for b_col_block in range(3):
        # Block 1: A1 * B_top + A2 * B_bottom
        # A1: rows 0-1, cols 0-1 of A
        # A2: rows 0-1, cols 2-3 of A
        # B_top: rows 0-1, cols b_col_block*2 to b_col_block*2+1 of B
        # B_bottom: rows 2-3, cols b_col_block*2 to b_col_block*2+1 of B
        
        # We need to map the local 2x2 indices to global indices.
        
        # A1 local (i,j) -> global (i, j)
        # A2 local (i,j) -> global (i, j+2)
        # B_top local (i,j) -> global (i, b_col_block*2 + j)
        # B_bottom local (i,j) -> global (i+2, b_col_block*2 + j)
        
        # We perform 7 Strassen steps for A1 * B_top
        for s in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # U: A1
            for (idx, coeff) in u_coeffs:
                i_loc, j_loc = divmod(idx, 2)
                u_row = i_loc * m + j_loc
                U[u_row, col] += coeff
                
            # V: B_top
            for (idx, coeff) in v_coeffs:
                i_loc, j_loc = divmod(idx, 2)
                v_row = i_loc * p + (b_col_block * 2 + j_loc)
                V[v_row, col] += coeff
                
            # W: C
            for (idx, coeff) in w_coeffs:
                i_loc, j_loc = divmod(idx, 2)
                # C is 2x6. Output cols are b_col_block*2 to b_col_block*2+1
                # W index is k*n + i.
                # i is row of C (0-1). k is col of C (0-5).
                w_row = (b_col_block * 2 + j_loc) * n + i_loc
                W[w_row, col] += coeff
                
            col += 1
            
        # We perform 7 Strassen steps for A2 * B_bottom
        for s in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # U: A2
            for (idx, coeff) in u_coeffs:
                i_loc, j_loc = divmod(idx, 2)
                u_row = i_loc * m + (j_loc + 2)
                U[u_row, col] += coeff
                
            # V: B_bottom
            for (idx, coeff) in v_coeffs:
                i_loc, j_loc = divmod(idx, 2)
                v_row = (i_loc + 2) * p + (b_col_block * 2 + j_loc)
                V[v_row, col] += coeff
                
            # W: C
            for (idx, coeff) in w_coeffs:
                i_loc, j_loc = divmod(idx, 2)
                w_row = (b_col_block * 2 + j_loc) * n + i_loc
                W[w_row, col] += coeff
                
            col += 1

    # Verification
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    C_decomposed = np.zeros((p, n))
    for r in range(rank):
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n)
        M = u_mat @ v_mat
        C_decomposed += M.T * w_mat

    if not np.allclose(C_decomposed, C_expected.T):
        # Fallback
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
