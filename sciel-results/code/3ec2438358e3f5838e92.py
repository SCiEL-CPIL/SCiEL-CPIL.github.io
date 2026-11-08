import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    # Generate rank 49 decomposition for 4x4x4
    U_44, V_44, W_44 = _strassen_4x4()
    
    # Duplicate for 4x4x8
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First 49 terms: use first 4 columns of B and first 4 rows of C
    for k in range(49):
        U[:, k] = U_44[:, k]
        V[:4*4, k] = V_44[:, k] # V_44 is 16x49, so V_44[:, k] is 16x1
        V[4*4:, k] = 0
        W[:4*4, k] = W_44[:, k] # W_44 is 16x49
        W[4*4:, k] = 0
        
    # Next 49 terms: use last 4 columns of B and last 4 rows of C
    for k in range(49):
        U[:, 49 + k] = U_44[:, k]
        V[:4*4, 49 + k] = 0
        V[4*4:, 49 + k] = V_44[:, k]
        W[:4*4, 49 + k] = 0
        W[4*4:, 49 + k] = W_44[:, k]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2_triples():
    # Returns 7 triples of (U, V, W) 2x2 matrices
    # T = sum U_k (x) V_k (x) W_k
    # U, V, W are 2x2
    triples = []
    
    # 1. M1 = (A11+A22)(B11+B22)
    U1 = np.array([[1, 0], [0, 1]])
    V1 = np.array([[1, 1], [0, 0]])
    W1 = np.array([[1, 0], [0, 1]])
    triples.append((U1, V1, W1))
    
    # 2. M2 = (A21+A22)B11
    U2 = np.array([[0, 0], [1, 1]])
    V2 = np.array([[1, 0], [0, 0]])
    W2 = np.array([[0, 0], [1, 0]])
    triples.append((U2, V2, W2))
    
    # 3. M3 = A11(B12-B22)
    U3 = np.array([[1, 0], [0, 0]])
    V3 = np.array([[0, 1], [0, -1]])
    W3 = np.array([[0, 1], [0, 0]])
    triples.append((U3, V3, W3))
    
    # 4. M4 = A22(B21-B11)
    U4 = np.array([[0, 0], [0, 1]])
    V4 = np.array([[-1, 0], [1, 0]])
    W4 = np.array([[0, 0], [1, 0]])
    triples.append((U4, V4, W4))
    
    # 5. M5 = (A11+A12)B22
    U5 = np.array([[1, 1], [0, 0]])
    V5 = np.array([[0, 0], [0, 1]])
    W5 = np.array([[0, 0], [0, 1]])
    triples.append((U5, V5, W5))
    
    # 6. M6 = (A21-A11)(B11+B12)
    U6 = np.array([[-1, 0], [1, 0]])
    V6 = np.array([[1, 1], [0, 0]])
    W6 = np.array([[0, 0], [0, 1]])
    triples.append((U6, V6, W6))
    
    # 7. M7 = (A12-A22)(B21+B22)
    U7 = np.array([[0, 1], [0, -1]])
    V7 = np.array([[0, 0], [1, 1]])
    W7 = np.array([[0, 1], [0, 0]])
    triples.append((U7, V7, W7))
    
    return triples

def _strassen_4x4():
    base_triples = _strassen_2x2_triples()
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Outer Strassen steps for 4x4 (blocks of 2x2)
    # X_k and Y_k are 2x2 block matrices (4x4 total)
    # We define the block coefficients for X and Y
    
    outer_steps = [
        # (X_blocks, Y_blocks, W_block_coeff)
        # X_blocks is 2x2 matrix of coefficients for A blocks
        # Y_blocks is 2x2 matrix of coefficients for B blocks
        # W_block_coeff is 2x2 matrix of coefficients for C blocks
        (np.array([[1, 0], [0, 1]]), np.array([[1, 1], [0, 0]]), np.array([[1, 0], [0, 1]])), # M1 -> C11, C22
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, 0]])), # M2 -> C21
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]]), np.array([[0, 1], [0, 0]])), # M3 -> C12
        (np.array([[0, 0], [0, 1]]), np.array([[-1, 0], [1, 0]]), np.array([[0, 0], [1, 0]])), # M4 -> C11, C21
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 1]])), # M5 -> C11, C12
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])), # M6 -> C22
        (np.array([[0, 1], [0, -1]]), np.array([[0, 0], [1, 1]]), np.array([[0, 1], [0, 0]])), # M7 -> C11, C12
    ]
    
    for k in range(7):
        X_blocks, Y_blocks, W_block_coeff = outer_steps[k]
        
        # For each of the 7 base Strassen triples
        for l in range(7):
            U_base, V_base, W_base = base_triples[l]
            
            # Construct U_44_col
            # U_44 is 4x4. It is formed by U_base acting on X_blocks?
            # No, U_base is 2x2 scalar. X_blocks is 2x2 scalar.
            # The block matrix X is formed by X_blocks * A_blocks.
            # The rank-1 term for the block multiplication is:
            # (U_base * X_blocks) (as 2x2 block matrix)
            # This means each block (i,j) of U_44 is U_base[i,j] * X_blocks[i,j] * A_block?
            # No, U_base is the coefficient for the rank-1 term of the 2x2 multiplication.
            # The 2x2 multiplication is X * Y.
            # X is a 2x2 block matrix.
            # The rank-1 decomposition of X*Y is sum (U_base * X) (x) (Y * V_base^T) (x) W_base?
            # Here U_base, V_base, W_base are 2x2 scalar matrices.
            # U_base * X means multiplying the 2x2 scalar matrix U_base with the 2x2 block matrix X.
            # Result is a 2x2 block matrix.
            
            U_44_block = U_base @ X_blocks # 2x2 matrix of coefficients
            V_44_block = Y_blocks @ V_base.T # 2x2 matrix of coefficients
            W_44_block = W_base @ W_block_coeff # 2x2 matrix of coefficients
            
            # Now expand these 2x2 block matrices into 4x4 matrices
            # U_44_col is 4x4
            U_col = np.zeros((4, 4))
            V_col = np.zeros((4, 4))
            W_col = np.zeros((4, 4))
            
            for i in range(2):
                for j in range(2):
                    # Block (i,j) of U_44_col
                    U_col[i*2:(i+1)*2, j*2:(j+1)*2] = U_44_block[i, j] * np.eye(2)
                    V_col[i*2:(i+1)*2, j*2:(j+1)*2] = V_44_block[i, j] * np.eye(2)
                    W_col[i*2:(i+1)*2, j*2:(j+1)*2] = W_44_block[i, j] * np.eye(2)
                    
            idx = k * 7 + l
            U_44[:, idx] = U_col.flatten()
            V_44[:, idx] = V_col.flatten()
            W_44[:, idx] = W_col.flatten()
            
    return U_44, V_44, W_44

