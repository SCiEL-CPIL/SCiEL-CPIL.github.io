import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition strategy achieving rank 56.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i, idx in enumerate(u_idx):
                u_col[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                v_col[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                w_col[idx] = W_blk[i, r]
                
            U_parts.append(u_col)
            V_parts.append(v_col)
            W_parts.append(w_col)

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        W[1, 0] = -1; W[2, 0] = -1
        
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

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Indices for U (3x4 flattened):
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    
    # Indices for V (4x5 flattened):
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Row 3: 15,16,17,18,19
    
    # Indices for W (5x3 flattened, k*n+i):
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    # k=3: 9,10,11
    # k=4: 12,13,14
    
    # Part 1: 2x4x5 (Rows 0,1 of A)
    # Split m=4 into 2+2.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], *strassen_2x2x2())
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] (2x2x2)
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], *strassen_2x2x2())
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1)
    add_block([0, 1, 4, 5], [4, 9], [12, 13], *schoolbook_2x2x1())
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] (2x2x2)
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], *strassen_2x2x2())
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] (2x2x2)
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], *strassen_2x2x2())
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] (2x2x1)
    add_block([2, 3, 6, 7], [14, 19], [12, 13], *schoolbook_2x2x1())
    
    # Part 2: 1x4x5 (Row 2 of A)
    # Split m=4 into 2+2.
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] (1x2x2)
    add_block([8, 9], [0, 1, 5, 6], [2, 5], *schoolbook_1x2x2())
    
    # Block 8: A[2:3, 0:2] * B[0:2, 2:4] (1x2x2)
    add_block([8, 9], [2, 3, 7, 8], [8, 11], *schoolbook_1x2x2())
    
    # Block 9: A[2:3, 0:2] * B[0:2, 4:5] (1x2x1)
    add_block([8, 9], [4, 9], [14], *schoolbook_1x2x1())
    
    # Block 10: A[2:3, 2:4] * B[2:4, 0:2] (1x2x2)
    add_block([10, 11], [10, 11, 15, 16], [2, 5], *schoolbook_1x2x2())
    
    # Block 11: A[2:3, 2:4] * B[2:4, 2:4] (1x2x2)
    add_block([10, 11], [12, 13, 17, 18], [8, 11], *schoolbook_1x2x2())
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] (1x2x1)
    add_block([10, 11], [14, 19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
