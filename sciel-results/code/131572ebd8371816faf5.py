import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    
    # We will construct a rank 65 decomposition.
    # Part 1: Rank 49 for 4x4x4 (Strassen)
    # Part 2: Rank 16 for 4x4x1 (Schoolbook)
    
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 Strassen ---
    # We define the 7 Strassen matrices for 2x2 block multiplication.
    # Each is a 4x4 matrix (flattened) representing the linear form.
    
    # Helper to create a 4x4 matrix from 2x2 blocks of 2x2 matrices
    def make_4x4(b11, b12, b21, b22):
        # b11, b12, b21, b22 are 4-element lists (flattened 2x2)
        mat = [0.0] * 16
        # Block (0,0) is indices 0-3
        mat[0:4] = b11
        # Block (0,1) is indices 4-7
        mat[4:8] = b12
        # Block (1,0) is indices 8-11
        mat[8:12] = b21
        # Block (1,1) is indices 12-15
        mat[12:16] = b22
        return mat

    # Identity 2x2 blocks
    I = [1, 0, 0, 1]
    Z = [0, 0, 0, 0]
    
    # We need to express the Strassen terms in terms of A and B elements.
    # A is 4x4, B is 4x4 (first 4 cols).
    # A blocks: A11 (0-3), A12 (4-7), A21 (8-11), A22 (12-15)
    # B blocks: B11 (0-3), B12 (4-7), B21 (8-11), B22 (12-15)
    
    # The 7 X and Y matrices for the 2x2 block Strassen:
    # X1 = A11 + A22
    # Y1 = B11 + B22
    # X2 = A21 + A22
    # Y2 = B11
    # X3 = A11
    # Y3 = B12 - B22
    # X4 = A22
    # Y4 = B21 - B11
    # X5 = A11 + A12
    # Y5 = B22
    # X6 = A21 - A11
    # Y6 = B11 + B12
    # X7 = A12 - A22
    # Y7 = B21 + B22
    
    # We will generate the 49 terms by iterating over the 7 block-multiplications
    # and for each, applying the 2x2 Strassen decomposition.
    
    # Define the 2x2 Strassen decomposition terms (u, v, w) for a single 2x2 multiplication.
    # u, v are 4-dim vectors (coefficients for the 2x2 input matrices).
    # w is a 4-dim vector (coefficients for the 2x2 output matrix).
    
    # Strassen 2x2 terms:
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
    
    # We need to map these to the global indices.
    # For a block multiplication $S_k = X_k Y_k$, the result is a 2x2 block.
    # This block is part of the final C matrix.
    # The position of this block in C depends on k.
    # k=1: C11
    # k=2: C21
    # k=3: C12
    # k=4: C21
    # k=5: C11, C12
    # k=6: C22
    # k=7: C11, C22
    
    # Wait, the mapping of S_k to C blocks is:
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # So S1 contributes to C11 (coeff 1) and C22 (coeff 1).
    # S2 contributes to C21 (coeff 1) and C22 (coeff -1).
    # S3 contributes to C12 (coeff 1) and C22 (coeff 1).
    # S4 contributes to C11 (coeff 1) and C21 (coeff 1).
    # S5 contributes to C11 (coeff -1) and C12 (coeff 1).
    # S6 contributes to C22 (coeff 1).
    # S7 contributes to C11 (coeff 1).
    
    # For each S_k, we decompose the 2x2 product X_k Y_k using Strassen.
    # This gives 7 scalar terms.
    # Each scalar term is u_i * v_i * w_i.
    # u_i is a linear form in A, v_i in B, w_i in the 2x2 result block.
    # We need to map w_i to the global C indices.
    
    # Let's define the 7 (u, v, w) triples for 2x2 Strassen.
    # u, v are 4-dim (indices 0-3 for 2x2).
    # w is 4-dim (indices 0-3 for 2x2).
    
    def get_2x2_strassen_terms():
        terms = []
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C11=1, C12=0, C21=0, C22=1? No.
        # C11 = M1...
        # C12 = 0
        # C21 = 0
        # C22 = M1...
        # So w for M1 is [1, 0, 0, 1]
        terms.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, 0] # C21=1
        terms.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 0] # C12=1
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0] # C11=1, C21=1
        terms.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0] # C11=-1, C12=1
        terms.append((u, v, w))
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C22=1
        terms.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C11=1
        terms.append((u, v, w))
        
        return terms

    strassen_2x2 = get_2x2_strassen_terms()
    
    # Now for each of the 7 block multiplications S_k:
    # We have X_k and Y_k as 4x4 matrices (flattened).
    # We need to express the 2x2 Strassen terms of X_k Y_k in terms of global A, B, C.
    
    # The 2x2 Strassen terms assume input matrices are 2x2.
    # Here, the "elements" of X_k and Y_k are linear forms in A and B.
    # So for each term (u, v, w) in strassen_2x2:
    # u is a 4-dim vector of coefficients for the 4 elements of X_k.
    # v is a 4-dim vector of coefficients for the 4 elements of Y_k.
    # w is a 4-dim vector of coefficients for the 4 elements of the result block.
    
    # We need to expand u and v into 16-dim vectors (global A, B).
    # And w into 16-dim vectors (global C), scaled by the block position.
    
    # Let's define X_k and Y_k as 4x4 matrices (16 elements).
    # X_k is formed by combining A blocks.
    # Y_k is formed by combining B blocks.
    
    # We will create a list of 7 (X, Y) pairs.
    # X, Y are 16-element lists.
    
    # A blocks: 0-3, 4-7, 8-11, 12-15
    # B blocks: 0-3, 4-7, 8-11, 12-15
    
    # X1 = A11 + A22
    X1 = [0]*16
    X1[0:4] = [1,0,0,0]
    X1[12:16] = [1,0,0,0]
    # Wait, A11 is indices 0-3. A22 is 12-15.
    # X1[0]=1, X1[1]=0, X1[2]=0, X1[3]=0
    # X1[12]=1, X1[13]=0, X1[14]=0, X1[15]=0
    
    # This is tedious. I will use a helper to add blocks.
    def add_block(mat, block_idx, coeff):
        # block_idx 0: 0-3, 1: 4-7, 2: 8-11, 3: 12-15
        start = block_idx * 4
        for j in range(4):
            mat[start+j] += coeff * (1 if j==0 else 0) # Identity?
            # No, the blocks are just added. The elements are 1s.
            # A block is a 2x2 matrix of variables.
            # We are summing the blocks.
            # So X1 = A11 + A22 means X1 has 1s at positions corresponding to A11 and A22.
            pass
            
    # Actually, X_k is a 4x4 matrix where the (i,j) entry is a linear form.
    # For Strassen, X1 = A11 + A22.
    # This means the top-left 2x2 block of X1 is A11, and bottom-right is A22.
    # So X1 as a 16-element vector has:
    # Indices 0-3: 1,0,0,1 (Identity? No, A11 is a matrix of variables. We are just tracking coefficients.)
    # The coefficient of A_{0,0} is 1. A_{0,1} is 1? No.
    # A11 is the submatrix. We are adding the submatrices.
    # So X1 has 1s at all positions of A11 and A22.
    
    # Let's just hardcode the 49 terms. It's safer.
    # Or, I can use the known fact that the rank is 49 and just output a valid decomposition.
    # But I need to generate it.
    
    # I will use a simpler strategy:
    # Use the rank 80 schoolbook decomposition but try to find a lower rank via a known algorithm.
    # Since I can't easily generate the 49 terms, I will stick to the 65 rank decomposition.
    # I will generate the 49 terms by iterating through the 7 block multiplications and 7 scalar multiplications.
    
    # Let's define the 7 X and Y matrices (16-element vectors).
    def block_sum(b1, b2):
        return [b1[i] + b2[i] for i in range(4)]
    
    def block_diff(b1, b2):
        return [b1[i] - b2[i] for i in range(4)]
    
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3)
    # B blocks: B00, B01, B10, B11
    
    # X1 = A00 + A11
    # Y1 = B00 + B11
    # X2 = A10 + A11
    # Y2 = B00
    # X3 = A00
    # Y3 = B01 - B11
    # X4 = A11
    # Y4 = B10 - B00
    # X5 = A00 + A01
    # Y5 = B11
    # X6 = A10 - A00
    # Y6 = B00 + B01
    # X7 = A01 - A11
    # Y7 = B10 + B11
    
    # We represent each block as a 4-element vector of coefficients.
    # A00 = [1,0,0,0], A01 = [0,1,0,0], etc.
    # But we are summing blocks.
    # X1 is a 4x4 matrix.
    # Top-left 2x2 is A00. Bottom-right 2x2 is A11.
    # So X1 (16-elem) has 1s at 0-3 and 12-15.
    
    X1 = [0]*16; X1[0:4] = [1,1,1,1]; X1[12:16] = [1,1,1,1]
    Y1 = [0]*16; Y1[0:4] = [1,1,1,1]; Y1[12:16] = [1,1,1,1]
    
    X2 = [0]*16; X2[8:12] = [1,1,1,1]; X2[12:16] = [1,1,1,1]
    Y2 = [0]*16; Y2[0:4] = [1,1,1,1]
    
    X3 = [0]*16; X3[0:4] = [1,1,1,1]
    Y3 = [0]*16; Y3[4:8] = [1,1,1,1]; Y3[12:16] = [-1,-1,-1,-1]
    
    X4 = [0]*16; X4[12:16] = [1,1,1,1]
    Y4 = [0]*16; Y4[8:12] = [1,1,1,1]; Y4[0:4] = [-1,-1,-1,-1]
    
    X5 = [0]*16; X5[0:4] = [1,1,1,1]; X5[4:8] = [1,1,1,1]
    Y5 = [0]*16; Y5[12:16] = [1,1,1,1]
    
    X6 = [0]*16; X6[8:12] = [1,1,1,1]; X6[0:4] = [-1,-1,-1,-1]
    Y6 = [0]*16; Y6[0:4] = [1,1,1,1]; Y6[4:8] = [1,1,1,1]
    
    X7 = [0]*16; X7[4:8] = [1,1,1,1]; X7[12:16] = [-1,-1,-1,-1]
    Y7 = [0]*16; Y7[8:12] = [1,1,1,1]; Y7[12:16] = [1,1,1,1]
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # For each k in 0..6:
    # We have X_k and Y_k.
    # We decompose the 2x2 product of the "blocks" of X_k and Y_k.
    # Wait, X_k is a 4x4 matrix. It consists of 4 2x2 blocks.
    # Let X_k = [[Xk_00, Xk_01], [Xk_10, Xk_11]]
    # Each Xk_ij is a 2x2 matrix (4 elements).
    # The Strassen decomposition for 2x2 applies to these blocks.
    
    # So for each k, we extract the 4 blocks of X_k and Y_k.
    # Then we apply the 7 Strassen terms for 2x2.
    # Each term gives u (4-elem), v (4-elem), w (4-elem).
    # u is coefficients for the 4 blocks of X_k.
    # v is coefficients for the 4 blocks of Y_k.
    # w is coefficients for the 4 blocks of the result.
    
    # We need to expand u to 16-elem (global A).
    # u[0] * Xk_00 + u[1] * Xk_01 + u[2] * Xk_10 + u[3] * Xk_11
    # Xk_00 is a 4-elem vector (part of X_k).
    # So global_u = u[0]*Xk[0:4] + u[1]*Xk[4:8] + u[2]*Xk[8:12] + u[3]*Xk[12:16]
    
    # Similarly for v.
    
    # For w, it is a 4-elem vector for the result blocks.
    # The result block is at a specific position in C.
    # We need to map the 4 result blocks to global C indices.
    # And scale by the coefficient of S_k in the final C formula.
    
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # So for S1 (k=0):
    # Contributes to C11 (coeff 1) and C22 (coeff 1).
    # The result of S1 is a 2x2 block.
    # Let's say S1 result is R1.
    # R1 is a 2x2 matrix.
    # R1 is added to C11 (top-left 2x2 of C) and C22 (bottom-right 2x2 of C).
    
    # So for each term in the 2x2 Strassen decomposition of S1:
    # w is a 4-elem vector for R1.
    # We add w[0:4] to C11 (indices 0-3) and w[0:4] to C22 (indices 12-15).
    
    # This logic holds for all S_k.
    
    # Let's implement this.
    
    c_coeffs = {
        0: [(0, 1), (12, 1)], # S1 -> C11, C22
        1: [(8, 1), (12, -1)], # S2 -> C21, C22
        2: [(4, 1), (12, 1)], # S3 -> C12, C22
        3: [(0, 1), (8, 1)], # S4 -> C11, C21
        4: [(0, -1), (4, 1)], # S5 -> C11, C12
        5: [(12, 1)], # S6 -> C22
        6: [(0, 1)] # S7 -> C11
    }
    
    for k in range(7):
        Xk = Xs[k]
        Yk = Ys[k]
        
        # Extract blocks
        Xk_00 = Xk[0:4]
        Xk_01 = Xk[4:8]
        Xk_10 = Xk[8:12]
        Xk_11 = Xk[12:16]
        
        Yk_00 = Yk[0:4]
        Yk_01 = Yk[4:8]
        Yk_10 = Yk[8:12]
        Yk_11 = Yk[12:16]
        
        for u, v, w in strassen_2x2:
            # Expand u to 16-elem
            global_u = [0.0]*16
            for i in range(4):
                block = [Xk_00, Xk_01, Xk_10, Xk_11][i]
                for j in range(4):
                    global_u[i*4+j] += u[i] * block[j]
                    
            # Expand v to 20-elem (B has 5 cols, but we only use first 4)
            global_v = [0.0]*20
            for i in range(4):
                block = [Yk_00, Yk_01, Yk_10, Yk_11][i]
                for j in range(4):
                    global_v[i*5+j] += v[i] * block[j]
                    
            # Expand w to 20-elem (C has 5 rows, but we only use first 4)
            global_w = [0.0]*20
            for start, coeff in c_coeffs[k]:
                for j in range(4):
                    global_w[start+j] += coeff * w[j]
                    
            # Add to U, V, W
            for r in range(16):
                U[r][col] = global_u[r]
            for r in range(20):
                V[r][col] = global_v[r]
            for r in range(20):
                W[r][col] = global_w[r]
                
            col += 1
            
    # col should be 49
    
    # --- Part 2: 4x4x1 Schoolbook ---
    # C[:,4] = A @ B[:,4]
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1.0
            V[j*5+4][col] = 1.0
            W[4*4+i][col] = 1.0
            col += 1
            
    # col should be 65
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_dot_a = np.dot(U[:, r], A.flatten())
                    v_dot_b = np.dot(V[:, r], B.flatten())
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True
