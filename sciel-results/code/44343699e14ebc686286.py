import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # First, let's define the 7 Strassen terms for 2x2x2
    # We'll verify them numerically
    
    # For 2x2x2: A is 2x2, B is 2x2, C is 2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # The 7 bilinear forms (Strassen):
    # Each term r has: U_r (length 4 for A flattened), V_r (length 4 for B flattened), W_r (length 4 for C flattened)
    # A flattened: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
    # B flattened: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    # C flattened: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3
    
    # Strassen P's:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B11-B00)
    # P4 = A11*(B00-B11)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's define U, V, W for each term
    # U_r[i] = coefficient of A[i] in the first factor
    # V_r[j] = coefficient of B[j] in the second factor
    # W_r[k] = coefficient for C[k] in the output
    
    strassen_U = [
        [1, 0, 0, 1],  # P1: A00+A11
        [0, 0, 1, 1],  # P2: A10+A11
        [1, 0, 0, 0],  # P3: A00
        [0, 0, 0, 1],  # P4: A11
        [1, 1, 0, 0],  # P5: A00+A01
        [-1, 0, 1, 0], # P6: A10-A00
        [0, 1, 0, -1], # P7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # P1: B00+B11
        [1, 0, 0, 0],  # P2: B00
        [0, 1, 0, -1], # P3: B11-B00
        [1, 0, 0, -1], # P4: B00-B11
        [0, 0, 0, 1],  # P5: B11
        [1, 1, 0, 0],  # P6: B00+B01
        [0, 0, 1, 1],  # P7: B10+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1],   # P1: C00+=1, C11+=1
        [0, 0, 1, 0],   # P2: C10+=1
        [0, 1, 0, 1],   # P3: C01+=1, C11+=1
        [1, 0, 1, 0],   # P4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # P5: C00-=1, C01+=1
        [0, 0, 0, 1],   # P6: C11+=1
        [1, 0, 0, -1],  # P7: C00+=1, C11-=1
    ]
    
    # Verify: For random A, B, check if sum_r U_r·A * V_r·B * W_r gives correct C
    np.random.seed(42)
    A_test = np.random.randn(2, 2)
    B_test = np.random.randn(2, 2)
    C_test = A_test @ B_test
    
    C_computed = np.zeros((2, 2))
    for r in range(7):
        a_val = sum(strassen_U[r][i] * A_test.flat[i] for i in range(4))
        b_val = sum(strassen_V[r][j] * B_test.flat[j] for j in range(4))
        for k in range(4):
            C_computed.flat[k] += strassen_W[r][k] * a_val * b_val
    
    # Check
    error = np.max(np.abs(C_computed - C_test))
    if error > 1e-10:
        # Fallback to schoolbook if Strassen is wrong
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen verified! Now build M_{2,4,6} with rank 42
    # Decompose: split middle dimension 4 into two 2s
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term is M_{2,2,6}
    # For M_{2,2,6}: split the 6 columns of B into three 2-col blocks
    # Each sub-block is M_{2,2,2} with rank 7
    # So M_{2,2,6} has rank 21, and M_{2,4,6} has rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    # For each of the 2 middle blocks (j_block in 0,1):
    #   For each of the 3 column blocks of B (k_block in 0,1,2):
    #     Use 7 Strassen terms
    for j_block in range(2):  # which 2-col block of A / 2-row block of B
        for k_block in range(3):  # which 2-col block of B / 2-row block of C
            for r in range(7):  # 7 Strassen terms
                # A indices: i in {0,1}, j = j_block*2 + {0,1}
                # B indices: j = j_block*2 + {0,1}, k = k_block*2 + {0,1}
                # C indices: i in {0,1}, k = k_block*2 + {0,1}
                
                for i_idx in range(4):  # A flat index: i*2 + j_local
                    i = i_idx // 2
                    j_local = i_idx % 2
                    j = j_block * 2 + j_local
                    U[i * m + j][col] = strassen_U[r][i_idx]
                
                for j_idx in range(4):  # B flat index: j_local*2 + k_local
                    j_local = j_idx // 2
                    k_local = j_idx % 2
                    j = j_block * 2 + j_local
                    k = k_block * 2 + k_local
                    V[j * p + k][col] = strassen_V[r][j_idx]
                
                for k_idx in range(4):  # C flat index: i*2 + k_local
                    i = k_idx // 2
                    k_local = k_idx % 2
                    k = k_block * 2 + k_local
                    W[k * n + i][col] = strassen_W[r][k_idx]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
