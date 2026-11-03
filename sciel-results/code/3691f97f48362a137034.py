import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = 4, 5, 5
    
    # --- Helper: Verify decomposition ---
    def verify(U, V, W, n, m, p):
        """
        Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
        U, V, W are numpy arrays of shape (n*m, r), (m*p, r), (p*n, r).
        """
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Helper: Schoolbook decomposition ---
    def schoolbook(n, m, p):
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

    # --- Strassen 2x2x2 decomposition (rank 7) ---
    # Returns U, V, W arrays of shape (4, 7) each
    # U[i*2+j, c] corresponds to A[i,j]
    # V[j*2+k, c] corresponds to B[j,k]
    # W[k*2+i, c] corresponds to C[k,i]
    def strassen_2x2x2():
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mi, we need to express it as a product of linear forms
        # and decompose into rank-1 terms
        
        # Let's define the 7 terms explicitly
        # Each term has U, V, W vectors of length 4 (for 2x2 matrices)
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # U1 = [1, 0, 0, 1] (A11 + A22)
        # V1 = [1, 0, 0, 1] (B11 + B22)
        # This is a 2x2 matrix product, which needs 4 rank-1 terms
        # But we can combine with the W part
        
        # Actually, let me use the standard Strassen decomposition directly
        # The 7 rank-1 terms for 2x2x2 are:
        
        U_list = []
        V_list = []
        W_list = []
        
        # Term 1: (A11+A22)(B11+B22) contributes to C11 and C22
        # We need to decompose this 2x2 product into rank-1 terms
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is 4 terms, but Strassen uses 7 total for all of C
        
        # Let me use the known 7-term Strassen decomposition
        # Reference: standard Strassen algorithm
        
        # The 7 terms are:
        # 1. U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  # M1 contributes to C11, C22
        # 2. U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]  # M2 contributes to C21
        # 3. U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,-1] # M3 contributes to C12
        # 4. U=[0,0,0,1], V=[0,-1,1,0], W=[1,0,1,0]  # M4 contributes to C11, C21
        # 5. U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]  # M5 contributes to C11, C12
        # 6. U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]  # M6 contributes to C22
        # 7. U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,1]  # M7 contributes to C11, C22
        
        # Wait, I need to be more careful. Let me derive this properly.
        
        # For 2x2 matrix multiplication:
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Strassen's 7 multiplications:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Now, each Mi is a product of two linear combinations.
        # To get a rank-1 decomposition, we need to express each Mi as
        # a single multiplication, which contributes one rank-1 term.
        
        # The U vector for Mi is the coefficients of A in the first factor
        # The V vector for Mi is the coefficients of B in the second factor
        # The W vector for Mi is the coefficients of C that Mi contributes to
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # U1 = [1, 0, 0, 1] (coefficients of A11, A12, A21, A22)
        # V1 = [1, 0, 0, 1] (coefficients of B11, B12, B21, B22)
        # W1 = [1, 0, 0, 1] (contributes to C11 and C22)
        U_list.append([1, 0, 0, 1])
        V_list.append([1, 0, 0, 1])
        W_list.append([1, 0, 0, 1])
        
        # Term 2: M2 = (A21+A22)B11
        # U2 = [0, 0, 1, 1]
        # V2 = [1, 0, 0, 0]
        # W2 = [0, 0, 1, 0] (contributes to C21)
        U_list.append([0, 0, 1, 1])
        V_list.append([1, 0, 0, 0])
        W_list.append([0, 0, 1, 0])
        
        # Term 3: M3 = A11(B12-B22)
        # U3 = [1, 0, 0, 0]
        # V3 = [0, 1, 0, -1]
        # W3 = [0, 1, 0, 0] (contributes to C12)
        U_list.append([1, 0, 0, 0])
        V_list.append([0, 1, 0, -1])
        W_list.append([0, 1, 0, 0])
        
        # Term 4: M4 = A22(B21-B11)
        # U4 = [0, 0, 0, 1]
        # V4 = [0, -1, 1, 0]
        # W4 = [1, 0, 1, 0] (contributes to C11 and C21)
        U_list.append([0, 0, 0, 1])
        V_list.append([0, -1, 1, 0])
        W_list.append([1, 0, 1, 0])
        
        # Term 5: M5 = (A11+A12)B22
        # U5 = [1, 1, 0, 0]
        # V5 = [0, 0, 0, 1]
        # W5 = [-1, 1, 0, 0] (contributes to C11 and C12)
        U_list.append([1, 1, 0, 0])
        V_list.append([0, 0, 0, 1])
        W_list.append([-1, 1, 0, 0])
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # U6 = [-1, 0, 1, 0]
        # V6 = [1, 1, 0, 0]
        # W6 = [0, 0, 0, 1] (contributes to C22)
        U_list.append([-1, 0, 1, 0])
        V_list.append([1, 1, 0, 0])
        W_list.append([0, 0, 0, 1])
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # U7 = [0, 1, 0, -1]
        # V7 = [0, 0, 1, 1]
        # W7 = [1, 0, 0, 1] (contributes to C11 and C22)
        U_list.append([0, 1, 0, -1])
        V_list.append([0, 0, 1, 1])
        W_list.append([1, 0, 0, 1])
        
        U = np.array(U_list).T  # shape (4, 7)
        V = np.array(V_list).T  # shape (4, 7)
        W = np.array(W_list).T  # shape (4, 7)
        
        return U, V, W

    # --- 4x4x4 decomposition using Strassen (rank 49) ---
    # We split 4x4 into 2x2 blocks and apply Strassen
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]] where each block is 2x2
    
    # Using Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, which has rank 7 (by Strassen)
    # So total rank = 7 * 7 = 49
    
    def get_4x4x4_strassen():
        # Get the 2x2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2x2()
        # U2, V2, W2 have shape (4, 7)
        
        # Now we need to expand this to 4x4x4
        # The 4x4 matrix A has 16 elements
        # We index them as A[i*4+j] for i,j in 0..3
        
        # Block structure:
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # Similarly for B and C.
        
        # For each of the 7 Strassen terms, we need to expand the 2x2 product
        # into 7 rank-1 terms using the 2x2x2 Strassen decomposition.
        
        # The 7 Strassen terms for 4x4 are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2 matrix product.
        # We decompose each Mi using the 2x2x2 Strassen terms.
        
        # Let's define the block indices
        # For a 4x4 matrix, block (bi, bj) where bi,bj in 0..1
        # has rows bi*2 to bi*2+1 and cols bj*2 to bj*2+1
        # The local indices within the block are 0,1,2,3 corresponding to
        # (0,0), (0,1), (1,0), (1,1)
        
        # Global index for element (i,j) in 4x4 is i*4+j
        # For block (bi,bj), local (li,lj) -> global (bi*2+li, bj*2+lj)
        # So global index = (bi*2+li)*4 + (bj*2+lj)
        
        def block_index(bi, bj, li, lj):
            """Get global index for block (bi,bj) local position (li,lj)"""
            gi = bi * 2 + li
            gj = bj * 2 + lj
            return gi * 4 + gj
        
        # Now build the 49 terms
        U4 = []
        V4 = []
        W4 = []
        
        # For each of the 7 Strassen terms M1..M7
        # And for each of the 7 2x2x2 Strassen terms
        # We generate one rank-1 term for 4x4x4
        
        # First, define the 7 Strassen terms for the block level
        # Each term specifies which blocks of A and B are combined
        # and with what coefficients
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Mi, we have:
        # - A factor: linear combination of A blocks
        # - B factor: linear combination of B blocks
        # - C contribution: which C blocks Mi contributes to
        
        # Let's define the block-level Strassen terms
        # Each term is (A_blocks, B_blocks, C_blocks)
        # where A_blocks is a dict mapping block index to coefficient
        # Similarly for B and C
        
        block_strassen = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ]
        
        # Now for each block-level term and each 2x2x2 term, generate a rank-1 term
        for a_blocks, b_blocks, c_blocks in block_strassen:
            for u2, v2, w2 in zip(U2.T, V2.T, W2.T):
                # u2, v2, w2 are vectors of length 4 (for 2x2 local indices)
                # Local index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                
                U4_term = np.zeros(16)
                V4_term = np.zeros(16)
                W4_term = np.zeros(16)
                
                # Expand A factor
                for (bi, bj), coeff_a in a_blocks.items():
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            global_idx = block_index(bi, bj, li, lj)
                            U4_term[global_idx] += coeff_a * u2[local_idx]
                
                # Expand B factor
                for (bi, bj), coeff_b in b_blocks.items():
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            global_idx = block_index(bi, bj, li, lj)
                            V4_term[global_idx] += coeff_b * v2[local_idx]
                
                # Expand C factor
                for (bi, bj), coeff_c in c_blocks.items():
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            global_idx = block_index(bi, bj, li, lj)
                            W4_term[global_idx] += coeff_c * w2[local_idx]
                
                U4.append(U4_term)
                V4.append(V4_term)
                W4.append(W4_term)
        
        U4 = np.array(U4).T  # shape (16, 49)
        V4 = np.array(V4).T  # shape (16, 49)
        W4 = np.array(W4).T  # shape (16, 49)
        
        return U4, V4, W4

    # --- Construct 4x5x5 using block decomposition ---
    # A (4x5) = [A4 | a] where A4 is 4x4, a is 4x1
    # B (5x5) = [[B4, b], [bT, beta]] where B4 is 4x4, b is 4x1, beta is scalar
    # C (4x5) = A4*B4 + a*bT + A4*b + a*beta
    
    # Term 1: A4*B4 (4x4x4) -> rank 49
    U1, V1, W1 = get_4x4x4_strassen()
    
    # Term 2: a*bT (4x1x4) -> rank 16
    # C[i,k] = a[i] * b[k] for i,k in 0..3
    # A index: i*5+4 (column 4 of A)
    # B index: 4*5+k (row 4, column k of B)
    # C index: k*4+i (row k, column i of C)
    U2 = np.zeros((20, 16))
    V2 = np.zeros((25, 16))
    W2 = np.zeros((20, 16))
    col = 0
    for i in range(4):
        for k in range(4):
            U2[i*5+4, col] = 1
            V2[4*5+k, col] = 1
            W2[k*4+i, col] = 1
            col += 1
    
    # Term 3: A4*b (4x4x1) -> rank 16
    # C[i,4] = sum_j A[i,j] * b[j] for j in 0..3
    # A index: i*5+j
    # B index: j*5+4 (row j, column 4 of B)
    # C index: 4*4+i (row 4, column i of C)
    U3 = np.zeros((20, 16))
    V3 = np.zeros((25, 16))
    W3 = np.zeros((20, 16))
    col = 0
    for i in range(4):
        for j in range(4):
            U3[i*5+j, col] = 1
            V3[j*5+4, col] = 1
            W3[4*4+i, col] = 1
            col += 1
    
    # Term 4: a*beta (4x1x1) -> rank 4
    # C[i,4] = a[i] * beta
    # A index: i*5+4
    # B index: 4*5+4 (row 4, column 4 of B)
    # C index: 4*4+i
    U4 = np.zeros((20, 4))
    V4 = np.zeros((25, 4))
    W4 = np.zeros((20, 4))
    for i in range(4):
        U4[i*5+4, i] = 1
        V4[4*5+4, i] = 1
        W4[4*4+i, i] = 1
    
    # Combine all terms
    U = np.hstack([U1, U2, U3, U4])
    V = np.hstack([V1, V2, V3, V4])
    W = np.hstack([W1, W2, W3, W4])
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
