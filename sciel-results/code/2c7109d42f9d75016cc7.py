import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Part 1: M_{4,4,4} for first 4 columns of B
    # Part 2: M_{4,4,2} for last 2 columns of B
    
    # Generate M_{4,4,4} decomposition using recursive Strassen
    U1, V1, W1 = strassen_decomp(4, 4, 4)
    
    # Generate M_{4,4,2} decomposition using schoolbook
    U2, V2, W2 = schoolbook_decomp(4, 4, 2)
    
    # Combine
    # U1 is 16x49, U2 is 16x32
    # V1 is 16x49 (for first 4 cols of B), V2 is 8x32 (for last 2 cols of B)
    # W1 is 16x49 (for first 4 cols of C), W2 is 8x32 (for last 2 cols of C)
    
    # V needs to be 24xR (m*p = 4*6 = 24)
    # V1 corresponds to B[0:4, 0:4] -> rows 0-15 of V
    # V2 corresponds to B[0:4, 4:6] -> rows 16-23 of V
    
    # W needs to be 24xR (p*n = 6*4 = 24)
    # W1 corresponds to C[0:4, 0:4] -> rows 0-15 of W (k*n+i, k=0..3, i=0..3)
    # W2 corresponds to C[4:6, 0:4] -> rows 16-23 of W (k=4,5)
    
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Fill U
    U[:, :R1] = U1
    U[:, R1:] = U2
    
    # Fill V
    # V1 is for B[0:4, 0:4]. B is 4x6.
    # B indices: j*p+k.
    # For first 4 cols (k=0..3), j=0..3.
    # Indices: 0..15.
    V[0:16, :R1] = V1
    
    # V2 is for B[0:4, 4:6].
    # k=4,5.
    # Indices: j*6+k.
    # j=0, k=4 -> 4
    # j=0, k=5 -> 5
    # j=1, k=4 -> 10
    # ...
    # We need to map V2 rows (0..7) to V rows (4,5,10,11,16,17,22,23)
    # V2 row index corresponds to j*2 + k' where k'=0,1 (local k)
    # Local k'=0 -> global k=4
    # Local k'=1 -> global k=5
    
    for j in range(4):
        for k_local in range(2):
            v2_row = j * 2 + k_local
            v_global_row = j * 6 + (4 + k_local)
            V[v_global_row, R1:] = V2[v2_row, :]
            
    # Fill W
    # W1 is for C[0:4, 0:4].
    # C indices: k*n+i.
    # k=0..3, i=0..3.
    # Indices: 0..15.
    W[0:16, :R1] = W1
    
    # W2 is for C[4:6, 0:4].
    # k=4,5.
    # Indices: k*4+i.
    # k=4, i=0..3 -> 16..19
    # k=5, i=0..3 -> 20..23
    
    for k_local in range(2):
        for i in range(4):
            w2_row = k_local * 4 + i
            w_global_row = (4 + k_local) * 4 + i
            W[w_global_row, R1:] = W2[w2_row, :]
            
    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def strassen_decomp(n, m, p):
    # Returns U, V, W for M_{n,m,p} using recursive Strassen
    # Assumes n=m=p and even
    if n == 2:
        return strassen_base()
    
    n_half = n // 2
    m_half = m // 2
    p_half = p // 2
    
    # Recursive calls for the 7 products
    # Each product is M_{n_half, m_half, p_half}
    U_sub, V_sub, W_sub = strassen_decomp(n_half, m_half, p_half)
    r_sub = U_sub.shape[1]
    
    U = np.zeros((n*m, 7 * r_sub))
    V = np.zeros((m*p, 7 * r_sub))
    W = np.zeros((p*n, 7 * r_sub))
    
    # Define the 7 products
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # Coefficients for U (A blocks)
    # A is n x m. Blocks are n_half x m_half.
    # A11: rows 0..n_half-1, cols 0..m_half-1
    # A12: rows 0..n_half-1, cols m_half..m-1
    # A21: rows n_half..n-1, cols 0..m_half-1
    # A22: rows n_half..n-1, cols m_half..m-1
    
    # Coefficients for V (B blocks)
    # B is m x p. Blocks are m_half x p_half.
    # B11: rows 0..m_half-1, cols 0..p_half-1
    # B12: rows 0..m_half-1, cols p_half..p-1
    # B21: rows m_half..m-1, cols 0..p_half-1
    # B22: rows m_half..m-1, cols p_half..p-1
    
    # Coefficients for W (C blocks)
    # C is p x n. Blocks are p_half x n_half.
    # C11: rows 0..p_half-1, cols 0..n_half-1
    # C12: rows 0..p_half-1, cols n_half..n-1
    # C21: rows p_half..p-1, cols 0..n_half-1
    # C22: rows p_half..p-1, cols n_half..n-1
    
    # Helper to fill a block of U, V, W with a sub-decomposition
    def fill_block(U, V, W, U_sub, V_sub, W_sub, 
                   u_blocks, v_blocks, w_blocks, 
                   u_coeffs, v_coeffs, w_coeffs, 
                   offset):
        # u_blocks: list of (row_start, row_end, col_start, col_end) in U
        # u_coeffs: list of coefficients for each block
        # Similarly for V, W
        
        for i in range(r_sub):
            col = offset + i
            # U
            for idx, (rs, re, cs, ce) in enumerate(u_blocks):
                coeff = u_coeffs[idx]
                if coeff != 0:
                    U[rs:re, cs:ce, col] = coeff * U_sub[:, i]
            # V
            for idx, (rs, re, cs, ce) in enumerate(v_blocks):
                coeff = v_coeffs[idx]
                if coeff != 0:
                    V[rs:re, cs:ce, col] = coeff * V_sub[:, i]
            # W
            for idx, (rs, re, cs, ce) in enumerate(w_blocks):
                coeff = w_coeffs[idx]
                if coeff != 0:
                    W[rs:re, cs:ce, col] = coeff * W_sub[:, i]
                    
    # This helper is too complex for 3D slicing. Let's do it manually.
    
    # We will construct U, V, W by iterating over the 7 products
    for k in range(7):
        offset = k * r_sub
        # Copy sub-decomposition to U, V, W with appropriate scaling and placement
        
        # U part
        # Determine which A blocks are involved and their coefficients
        u_coeff = {}
        if k == 0: # P1
            u_coeff[(0,0)] = 1
            u_coeff[(1,1)] = 1
        elif k == 1: # P2
            u_coeff[(1,0)] = 1
            u_coeff[(1,1)] = 1
        elif k == 2: # P3
            u_coeff[(0,0)] = 1
        elif k == 3: # P4
            u_coeff[(1,1)] = 1
        elif k == 4: # P5
            u_coeff[(0,0)] = 1
            u_coeff[(0,1)] = 1
        elif k == 5: # P6
            u_coeff[(1,0)] = 1
            u_coeff[(0,0)] = -1
        elif k == 6: # P7
            u_coeff[(0,1)] = 1
            u_coeff[(1,1)] = -1
            
        # Map (row_block, col_block) to U indices
        # U is n*m x R.
        # A is n x m.
        # A[row_block, col_block] corresponds to U rows:
        # row_start = row_block * n_half
        # row_end = (row_block + 1) * n_half
        # col_start = col_block * m_half
        # col_end = (col_block + 1) * m_half
        # Wait, U index is i*m+j.
        # A[i,j] is at U[i*m+j].
        # So for a block, the indices are not contiguous in U if we consider i*m+j.
        # We need to iterate over i, j in the block.
        
        for (rb, cb), coeff in u_coeff.items():
            for i in range(rb * n_half, (rb + 1) * n_half):
                for j in range(cb * m_half, (cb + 1) * m_half):
                    u_idx = i * m + j
                    U[u_idx, offset:offset+r_sub] += coeff * U_sub[:, :]
                    
        # V part
        v_coeff = {}
        if k == 0: # P1
            v_coeff[(0,0)] = 1
            v_coeff[(1,1)] = 1
        elif k == 1: # P2
            v_coeff[(0,0)] = 1
        elif k == 2: # P3
            v_coeff[(0,1)] = 1
            v_coeff[(1,1)] = -1
        elif k == 3: # P4
            v_coeff[(1,0)] = 1
            v_coeff[(0,0)] = -1
        elif k == 4: # P5
            v_coeff[(1,1)] = 1
        elif k == 5: # P6
            v_coeff[(0,0)] = 1
            v_coeff[(0,1)] = 1
        elif k == 6: # P7
            v_coeff[(1,0)] = 1
            v_coeff[(1,1)] = 1
            
        for (rb, cb), coeff in v_coeff.items():
            for i in range(rb * m_half, (rb + 1) * m_half):
                for j in range(cb * p_half, (cb + 1) * p_half):
                    v_idx = i * p + j
                    V[v_idx, offset:offset+r_sub] += coeff * V_sub[:, :]
                    
        # W part
        # C = P1 + P4 - P5 + P7 for C11
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        w_coeff = {}
        if k == 0: # P1
            w_coeff[(0,0)] = 1
            w_coeff[(1,1)] = 1
        elif k == 1: # P2
            w_coeff[(1,0)] = 1
            w_coeff[(1,1)] = -1
        elif k == 2: # P3
            w_coeff[(0,1)] = 1
            w_coeff[(1,1)] = 1
        elif k == 3: # P4
            w_coeff[(0,0)] = 1
            w_coeff[(1,0)] = 1
        elif k == 4: # P5
            w_coeff[(0,0)] = -1
            w_coeff[(0,1)] = 1
        elif k == 5: # P6
            w_coeff[(1,1)] = 1
        elif k == 6: # P7
            w_coeff[(0,0)] = 1
            
        for (rb, cb), coeff in w_coeff.items():
            for i in range(rb * p_half, (rb + 1) * p_half):
                for j in range(cb * n_half, (cb + 1) * n_half):
                    w_idx = i * n + j
                    W[w_idx, offset:offset+r_sub] += coeff * W_sub[:, :]
                    
    return U, V, W

def strassen_base():
    # M_{2,2,2} Strassen decomposition
    # Returns U, V, W of shape 4x7, 4x7, 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # P3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W
