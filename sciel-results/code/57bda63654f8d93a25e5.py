import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank from the schoolbook 60
    to 56. The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 1. Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # 2. Schoolbook 2x2x1 decomposition (rank 4)
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    # 3. Schoolbook 1x2x2 decomposition (rank 4)
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    # 4. Schoolbook 1x2x1 decomposition (rank 2)
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Helper to add a block
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u_vec = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u_vec[idx] = U_blk[i, r]
            U_parts.append(u_vec)
            
            v_vec = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v_vec[idx] = V_blk[i, r]
            V_parts.append(v_vec)
            
            w_vec = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w_vec[idx] = W_blk[i, r]
            W_parts.append(w_vec)

    # U indices: A[i,j] -> i*m + j
    # Row 0: 0,1,2,3; Row 1: 4,5,6,7; Row 2: 8,9,10,11
    
    # V indices: B[i,j] -> i*p + j
    # Row 0: 0,1,2,3,4; Row 1: 5,6,7,8,9; Row 2: 10,11,12,13,14; Row 3: 15,16,17,18,19
    
    # W indices: C[i,k] -> k*n + i
    # C[0,0]=0, C[0,1]=3, C[0,2]=6, C[0,3]=9, C[0,4]=12
    # C[1,0]=1, C[1,1]=4, C[1,2]=7, C[1,3]=10, C[1,4]=13
    # C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11, C[2,4]=14
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2)
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (2x2x2)
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    
    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x2x1)
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    
    # Block 4: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2)
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    
    # Block 5: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4] (2x2x2)
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    
    # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x2x1)
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    
    # Block 7: A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2] (1x2x2)
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    
    # Block 8: A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4] (1x2x2)
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    
    # Block 9: A[2:3,0:2] * B[0:2,4:5] -> C[2:3,4:5] (1x2x1)
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    
    # Block 10: A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2] (1x2x2)
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    
    # Block 11: A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4] (1x2x2)
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    
    # Block 12: A[2:3,2:4] * B[2:4,4:5] -> C[2:3,4:5] (1x2x1)
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
