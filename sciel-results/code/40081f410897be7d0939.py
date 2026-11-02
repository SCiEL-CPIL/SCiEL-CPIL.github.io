import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition using block structure.
    # Split m=4 into m1=2, m2=2.
    # A is n x m. A_top is n x m1, A_bottom is n x m2.
    # B is m x p. B_top is m1 x p, B_bottom is m2 x p.
    # AB = A_top * B_top + A_bottom * B_bottom.
    
    # Rank of A_top * B_top (3x2 * 2x7)
    # Rank of A_bottom * B_bottom (3x2 * 2x7)
    
    # We need a decomposition for 3x2 * 2x7.
    # Split n=3 into n1=1, n2=2.
    # A_top is 1x2, A_bottom is 2x2.
    # A_top * B_top is 1x2 * 2x7.
    # A_bottom * B_top is 2x2 * 2x7.
    
    # 1x2 * 2x7:
    # This is just 7 independent 1x2 * 2x1 products? No, B is 2x7.
    # Let A1 = [a, b] (1x2), B1 = [[e, f, ..., t], [g, h, ..., v]] (2x7).
    # Result C = [ae+bg, af+bh, ...]
    # Each column c_j is a*e_j + b*g_j.
    # This is a dot product.
    # We can decompose this with rank 2 * 7 = 14?
    # Actually, for each column j, we have a*e_j + b*g_j.
    # This is a sum of 2 terms.
    # So we can have 2 factors for each column?
    # Let's use the trivial decomposition for 1x2 * 2x7.
    # Rank = 1 * 2 * 7 = 14.
    
    # 2x2 * 2x7:
    # Split p=7 into p1=4, p2=3.
    # 2x2 * 2x4: Split p1=4 into 2+2.
    # 2x2 * 2x2 (Strassen, rank 7) + 2x2 * 2x2 (Strassen, rank 7) = 14.
    # 2x2 * 2x3: Split p2=3 into 1+2.
    # 2x2 * 2x1 (Trivial, rank 4) + 2x2 * 2x2 (Strassen, rank 7) = 11.
    # Total for 2x2 * 2x7 = 14 + 11 = 25.
    
    # Total for 3x2 * 2x7 = 14 + 25 = 39.
    # Total for 3x4 * 4x7 = 39 + 39 = 78.
    
    # We will construct the matrices U, V, W by concatenating the blocks.
    
    # Helper to create trivial decomposition for n x m * m x p
    def trivial_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return U, V, W, rank

    # Helper to create Strassen decomposition for 2x2 * 2x2
    def strassen_decomp():
        # Returns U, V, W for 2x2 * 2x2 with rank 7
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices in U: 00, 01, 10, 11
        # Indices in V: 00, 01, 10, 11
        # Indices in W: 00, 01, 10, 11
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # z11 += m1, z22 += m1
        
        # m2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # z21 += m2
        
        # m3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = -1 # z12 += m3, z22 -= m3
        
        # m4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = -1 # z21 += m4, z11 -= m4
        
        # m5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1 # z11 += m5, z12 += m5
        
        # m6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # z22 += m6
        
        # m7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1 # z11 += m7, z22 += m7
        
        return U, V, W, 7

    # Build 2x2 * 2x1 (trivial, rank 4)
    def decomp_2x2_2x1():
        U, V, W, r = trivial_decomp(2, 2, 1)
        return U, V, W, r

    # Build 2x2 * 2x2 (Strassen, rank 7)
    def decomp_2x2_2x2():
        U, V, W, r = strassen_decomp()
        return U, V, W, r

    # Build 2x2 * 2x3 (split 3 into 1+2)
    def decomp_2x2_2x3():
        U1, V1, W1, r1 = decomp_2x2_2x1()
        U2, V2, W2, r2 = decomp_2x2_2x2()
        
        U = np.hstack([U1, U2])
        # V needs to be mapped to 2x3 indices
        # V1 is (2*1, r1) = (2, 4). Indices: 0, 1.
        # V2 is (2*2, r2) = (4, 7). Indices: 0, 1, 2, 3.
        # We need V to be (2*3, r1+r2) = (6, 11).
        # V1 corresponds to col 0 of B.
        # V2 corresponds to cols 1, 2 of B.
        
        V_full = np.zeros((6, r1 + r2))
        # Map V1
        for j in range(2):
            V_full[j, :r1] = V1[j, :]
        # Map V2
        for j in range(2):
            for k in range(2):
                v_idx = j * 2 + k
                # In 2x3, col index is 1 + k for the second block?
                # No, B is 2x3. Rows are 0,1. Cols are 0,1,2.
                # V2 is for B[:, 1:3]. So col 0 of V2 corresponds to col 1 of B.
                # Col 1 of V2 corresponds to col 2 of B.
                V_full[v_idx, r1 + k] = V2[j * 2 + k, k] # This mapping is wrong.
        
        # Let's do it more carefully.
        # V2 has rows indexed by (j, k) where j in 0..1, k in 0..1.
        # In the 2x3 matrix, the second block is columns 1 and 2.
        # So row (j, 0) of V2 corresponds to row j, col 1 of B.
        # Row (j, 1) of V2 corresponds to row j, col 2 of B.
        
        V_full = np.zeros((6, r1 + r2))
        # Block 1: col 0
        for j in range(2):
            V_full[j, :r1] = V1[j, :]
        # Block 2: cols 1, 2
        for j in range(2):
            for k in range(2):
                v_idx = j * 3 + (1 + k) # Row index in 2x3 matrix
                v2_idx = j * 2 + k # Row index in 2x2 matrix
                V_full[v_idx, r1 + k] = V2[v2_idx, k] # This is still wrong, V2 columns are not just k.
                
        # Correct mapping for V2:
        # V2 is (4, 7). Rows are (0,0), (0,1), (1,0), (1,1).
        # We need to place these into V_full rows (0,1), (0,2), (1,1), (1,2).
        for j in range(2):
            for k in range(2):
                v2_row = j * 2 + k
                v_full_row = j * 3 + (1 + k)
                V_full[v_full_row, r1:r1+r2] = V2[v2_row, :]
                
        # W needs to be mapped to 2x3 indices
        # W1 is (1*2, r1) = (2, 4). Indices: (0,0), (1,0).
        # W2 is (2*2, r2) = (4, 7). Indices: (0,0), (0,1), (1,0), (1,1).
        # W_full is (3*2, r1+r2) = (6, 11).
        # W1 corresponds to col 0 of result.
        # W2 corresponds to cols 1, 2 of result.
        
        W_full = np.zeros((6, r1 + r2))
        for i in range(2):
            W_full[i * 2 + 0, :r1] = W1[i, :]
        for i in range(2):
            for k in range(2):
                w2_row = k * 2 + i
                w_full_row = (1 + k) * 2 + i
                W_full[w_full_row, r1:r1+r2] = W2[w2_row, :]
                
        return U, V_full, W_full, r1 + r2

    # Build 2x2 * 2x4 (split 4 into 2+2)
    def decomp_2x2_2x4():
        U1, V1, W1, r1 = decomp_2x2_2x2()
        U2, V2, W2, r2 = decomp_2x2_2x2()
        
        U = np.hstack([U1, U2])
        
        V_full = np.zeros((8, r1 + r2))
        for j in range(2):
            for k in range(2):
                v1_row = j * 2 + k
                v_full_row = j * 4 + k
                V_full[v_full_row, :r1] = V1[v1_row, :]
                
                v2_row = j * 2 + k
                v_full_row = j * 4 + (2 + k)
                V_full[v_full_row, r1:r1+r2] = V2[v2_row, :]
                
        W_full = np.zeros((8, r1 + r2))
        for i in range(2):
            for k in range(2):
                w1_row = k * 2 + i
                w_full_row = k * 2 + i
                W_full[w_full_row, :r1] = W1[w1_row, :]
                
                w2_row = k * 2 + i
                w_full_row = (2 + k) * 2 + i
                W_full[w_full_row, r1:r1+r2] = W2[w2_row, :]
                
        return U, V_full, W_full, r1 + r2

    # Build 2x2 * 2x7 (split 7 into 4+3)
    def decomp_2x2_2x7():
        U1, V1, W1, r1 = decomp_2x2_2x4()
        U2, V2, W2, r2 = decomp_2x2_2x3()
        
        U = np.hstack([U1, U2])
        
        V_full = np.zeros((14, r1 + r2))
        for j in range(2):
            for k in range(4):
                v1_row = j * 4 + k
                v_full_row = j * 7 + k
                V_full[v_full_row, :r1] = V1[v1_row, :]
                
                v2_row = j * 3 + k # Wait, V2 is for 2x3, so rows are j*3+k
                # But we only have k=0,1,2 for V2.
                # V1 covers k=0,1,2,3. V2 covers k=4,5,6.
                # So for k=0,1,2, we fill V2.
                if k < 3:
                    v2_row = j * 3 + k
                    v_full_row = j * 7 + (4 + k)
                    V_full[v_full_row, r1:r1+r2] = V2[v2_row, :]
                    
        W_full = np.zeros((14, r1 + r2))
        for i in range(2):
            for k in range(4):
                w1_row = k * 2 + i
                w_full_row = k * 2 + i
                W_full[w_full_row, :r1] = W1[w1_row, :]
                
                if k < 3:
                    w2_row = k * 2 + i
                    w_full_row = (4 + k) * 2 + i
                    W_full[w_full_row, r1:r1+r2] = W2[w2_row, :]
                    
        return U, V_full, W_full, r1 + r2

    # Build 1x2 * 2x7 (trivial, rank 14)
    def decomp_1x2_2x7():
        U, V, W, r = trivial_decomp(1, 2, 7)
        return U, V, W, r

    # Build 3x2 * 2x7 (split 3 into 1+2)
    def decomp_3x2_2x7():
        U1, V1, W1, r1 = decomp_1x2_2x7()
        U2, V2, W2, r2 = decomp_2x2_2x7()
        
        U = np.vstack([U1, U2])
        
        # V is same for both, just concatenate columns
        V_full = np.hstack([V1, V2])
        
        # W needs to be mapped
        # W1 is (7*1, r1) = (7, 14). Rows: 0..6.
        # W2 is (7*2, r2) = (14, 25). Rows: 0..13.
        # W_full is (7*3, r1+r2) = (21, 39).
        # W1 corresponds to row 0 of A.
        # W2 corresponds to rows 1, 2 of A.
        
        W_full = np.zeros((21, r1 + r2))
        for i in range(1):
            W_full[i * 7 + 0, :r1] = W1[i, :] # This is wrong, W1 rows are k*n+i. n=1. So row i.
            # Actually W1 is (p*n, r) = (7*1, r). Row index is k*1 + i.
            # So W1[k, :] corresponds to result[k, 0].
            
        for i in range(1):
            for k in range(7):
                W_full[k * 3 + 0, :r1] = W1[k, :]
                
        for i in range(2):
            for k in range(7):
                w2_row = k * 2 + i
                w_full_row = k * 3 + (1 + i)
                W_full[w_full_row, r1:r1+r2] = W2[w2_row, :]
                
        return U, V_full, W_full, r1 + r2

    # Build 3x4 * 4x7 (split 4 into 2+2)
    U1, V1, W1, r1 = decomp_3x2_2x7()
    U2, V2, W2, r2 = decomp_3x2_2x7()
    
    U_full = np.hstack([U1, U2])
    
    # V mapping
    # V1 is for B[:, 0:2]. V2 is for B[:, 2:4].
    # V1 is (14, r1). V2 is (14, r2).
    # V_full is (28, r1+r2).
    V_full = np.zeros((28, r1 + r2))
    for j in range(2):
        for k in range(7):
            v1_row = j * 7 + k
            v_full_row = j * 7 + k
            V_full[v_full_row, :r1] = V1[v1_row, :]
            
            v2_row = j * 7 + k
            v_full_row = j * 7 + (2 + k) # This is wrong. B is 4x7.
            # V1 corresponds to B[0:2, :]. Rows 0,1.
            # V2 corresponds to B[2:4, :]. Rows 2,3.
            # So V1 row (j,k) -> V_full row (j,k).
            # V2 row (j,k) -> V_full row (2+j, k).
            
            # Let's redo V mapping.
            # V1 rows: j in 0..1, k in 0..6. Index j*7+k.
            # V2 rows: j in 0..1, k in 0..6. Index j*7+k.
            # V_full rows: j in 0..3, k in 0..6. Index j*7+k.
            # V1[j*7+k] -> V_full[j*7+k]
            # V2[j*7+k] -> V_full[(2+j)*7+k]
            
    V_full = np.zeros((28, r1 + r2))
    for j in range(2):
        for k in range(7):
            v1_row = j * 7 + k
            V_full[v1_row, :r1] = V1[v1_row, :]
            
            v2_row = j * 7 + k
            v2_full_row = (2 + j) * 7 + k
            V_full[v2_full_row, r1:r1+r2] = V2[v2_row, :]
            
    # W mapping
    # W1 is (21, r1). W2 is (21, r2).
    # W_full is (21, r1+r2).
    # W1 corresponds to A[:, 0:2]. W2 corresponds to A[:, 2:4].
    # The result is the same, so we just add them.
    # W_full[:, :r1] = W1
    # W_full[:, r1:] = W2
    
    W_full = np.hstack([W1, W2])
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
