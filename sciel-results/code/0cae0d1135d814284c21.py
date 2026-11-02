import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def build_2x3x2_rank11():
        """Build a rank-11 decomposition for 2x3x2 matrix multiplication.
        
        U: (6, 11), V: (6, 11), W: (4, 11)
        Constraint: for all i in {0,1}, j in {0,1,2}, k in {0,1}:
            sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        """
        # 2x3x2: A is 2x3, B is 3x2, C is 2x2
        # Split A into A1 (2x1, col 0) and A2 (2x2, cols 1,2)
        # A1*B: 2x1x2, rank 4
        # A2*B: 2x2x2, rank 7 (Strassen)
        # Total: rank 11
        
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Block 1: A1*B where A1 is 2x1 (col 0), B is 3x2
        # This is 2x1x2: T[i,0,k] = 1
        # U indices: i*3+0 for i in {0,1}
        # V indices: 0*2+k for k in {0,1} (row 0 of B)
        # W indices: k*2+i for k in {0,1}, i in {0,1}
        col = 0
        for i in range(2):
            for k in range(2):
                U[i*3+0, col] = 1
                V[0*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
        # col = 4
        
        # Block 2: A2*B where A2 is 2x2 (cols 1,2), B is 3x2
        # This is 2x2x2: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1}
        # In original indexing:
        # A2 corresponds to A[:, 1:3], so j_global = j_local + 1
        # B is 3x2, all rows used
        # For the sub-problem:
        # U indices: i*3 + (j_local+1) for i in {0,1}, j_local in {0,1}
        # V indices: (j_local+1)*2 + k for j_local in {0,1}, k in {0,1}
        # W indices: k*2 + i for k in {0,1}, i in {0,1}
        
        # Strassen decomposition for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        strassen_U = [
            [1, 1],  # M1: A00, A11
            [0, 1],  # M2: A10, A11
            [1, 0],  # M3: A00
            [1, 1],  # M4: A11
            [1, 1],  # M5: A00, A01
            [1, -1], # M6: A10, A00
            [1, -1], # M7: A01, A11
        ]
        strassen_V = [
            [1, 1],  # M1: B00, B11
            [1, 0],  # M2: B00
            [1, -1], # M3: B10, B11
            [1, -1], # M4: B01, B00
            [0, 1],  # M5: B11
            [1, 1],  # M6: B00, B10
            [1, 1],  # M7: B01, B11
        ]
        strassen_W_correct = [
            [1, 0, 0, -1],  # M1
            [0, -1, 0, -1], # M2
            [0, 0, 1, 1],   # M3
            [1, 1, 0, 0],   # M4
            [-1, 0, 1, 0],  # M5
            [0, 1, 0, 0],   # M6
            [1, 0, 0, 1],   # M7
        ]
        
        for r in range(7):
            # U: i in {0,1}, j_local in {0,1}
            # j_global = j_local + 1
            U[0*3 + 1, col+r] = strassen_U[r][0]  # i=0, j_local=0
            U[1*3 + 1, col+r] = strassen_U[r][1]  # i=1, j_local=0
            U[0*3 + 2, col+r] = strassen_U[r][0]  # i=0, j_local=1
            U[1*3 + 2, col+r] = strassen_U[r][1]  # i=1, j_local=1
            
            # V: j_local in {0,1}, k in {0,1}
            # j_global = j_local + 1
            V[1*2 + 0, col+r] = strassen_V[r][0]  # j_local=0, k=0
            V[1*2 + 1, col+r] = strassen_V[r][1]  # j_local=0, k=1
            V[2*2 + 0, col+r] = strassen_V[r][0]  # j_local=1, k=0
            V[2*2 + 1, col+r] = strassen_V[r][1]  # j_local=1, k=1
            
            # W: k in {0,1}, i in {0,1}
            # W index = k*2 + i
            W[0*2 + 0, col+r] = strassen_W_correct[r][0]  # k=0, i=0 (C00)
            W[0*2 + 1, col+r] = strassen_W_correct[r][1]  # k=0, i=1 (C10)
            W[1*2 + 0, col+r] = strassen_W_correct[r][2]  # k=1, i=0 (C01)
            W[1*2 + 1, col+r] = strassen_W_correct[r][3]  # k=1, i=1 (C11)
        
        col += 7
        # col = 11
        assert col == 11, f"Expected col=11, got {col}"
        
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the 2x3x6 multiplication tensor."""
        R = U.shape[1]
        if V.shape[1] != R or W.shape[1] != R:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build the 2x3x6 decomposition using three 2x3x2 blocks
    U1, V1, W1 = build_2x3x2_rank11()
    U2, V2, W2 = build_2x3x2_rank11()
    U3, V3, W3 = build_2x3x2_rank11()
    
    # For the first block (k=0,1):
    # U: (6, 11)
    # V: first 6 rows of V1, last 12 rows are 0
    # W: first 4 rows of W1, last 8 rows are 0
    
    # For the second block (k=2,3):
    # U: (6, 11)
    # V: first 6 rows are 0, middle 6 rows are V2, last 6 rows are 0
    # W: first 4 rows are 0, middle 4 rows are W2, last 4 rows are 0
    
    # For the third block (k=4,5):
    # U: (6, 11)
    # V: first 12 rows are 0, last 6 rows are V3
    # W: first 8 rows are 0, last 4 rows are W3
    
    R = 11
    U = np.zeros((6, 3*R))
    V = np.zeros((18, 3*R))
    W = np.zeros((12, 3*R))
    
    # First block (k=0,1)
    U[:, 0:R] = U1
    V[0:6, 0:R] = V1
    W[0:4, 0:R] = W1
    
    # Second block (k=2,3)
    U[:, R:2*R] = U2
    V[6:12, R:2*R] = V2
    W[4:8, R:2*R] = W2
    
    # Third block (k=4,5)
    U[:, 2*R:3*R] = U3
    V[12:18, 2*R:3*R] = V3
    W[8:12, 2*R:3*R] = W3
    
    if verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
