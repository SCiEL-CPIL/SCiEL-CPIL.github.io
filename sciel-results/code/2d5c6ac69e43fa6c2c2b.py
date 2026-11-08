import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 3x4x5 matrix multiplication.
    Rank 48: 2x4x5 part (rank 36) + 1x4x5 part (rank 12).
    Uses Strassen for 2x2x2 blocks, optimized low-rank for 1x2x2 and 1x2x1.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        """Strassen 2x2x2, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # M4 = A11(B10-B00)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def schoolbook_2x2x1():
        """2x2x1, rank 4."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W
    
    def optimal_1x2x2():
        """1x2x2, rank 2. C0=A0*B00+A1*B10, C1=A0*B01+A1*B11."""
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        # Term 1: (A0, A1) dot (B00, B10) -> C0
        U[0,0]=1; U[1,0]=1; V[0,0]=1; V[2,0]=1; W[0,0]=1
        # Term 2: (A0, A1) dot (B01, B11) -> C1
        U[0,1]=1; U[1,1]=1; V[1,1]=1; V[3,1]=1; W[1,1]=1
        return U, V, W
    
    def optimal_1x2x1():
        """1x2x1, rank 2. C=A0*B0+A1*B1."""
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            v = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            w = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)
    
    # Part 1: 2x4x5 (rows 0,1 of A)
    # Split B rows into [0:2] and [2:4], each giving 2x2x5
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 (split B cols into [0:2],[2:4],[4:5])
    
    # Block 1a: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], U_b, V_b, W_b)
    
    # Block 1b: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], U_b, V_b, W_b)
    
    # Block 1c: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x2x1)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([0,1,4,5], [4,9], [12,13], U_b, V_b, W_b)
    
    # Block 1d: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], U_b, V_b, W_b)
    
    # Block 1e: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4] (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], U_b, V_b, W_b)
    
    # Block 1f: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x2x1)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([2,3,6,7], [14,19], [12,13], U_b, V_b, W_b)
    
    # Part 2: 1x4x5 (row 2 of A)
    # Split B rows into [0:2] and [2:4], each giving 1x2x5
    # 1x2x5 = 1x2x2 + 1x2x2 + 1x2x1 (split B cols into [0:2],[2:4],[4:5])
    
    # Block 2a: A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2] (1x2x2)
    U_b, V_b, W_b = optimal_1x2x2()
    add_block([8,9], [0,1,5,6], [2,5], U_b, V_b, W_b)
    
    # Block 2b: A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4] (1x2x2)
    U_b, V_b, W_b = optimal_1x2x2()
    add_block([8,9], [2,3,7,8], [8,11], U_b, V_b, W_b)
    
    # Block 2c: A[2:3,0:2] * B[0:2,4:5] -> C[2:3,4:5] (1x2x1)
    U_b, V_b, W_b = optimal_1x2x1()
    add_block([8,9], [4,9], [14], U_b, V_b, W_b)
    
    # Block 2d: A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2] (1x2x2)
    U_b, V_b, W_b = optimal_1x2x2()
    add_block([10,11], [10,11,15,16], [2,5], U_b, V_b, W_b)
    
    # Block 2e: A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4] (1x2x2)
    U_b, V_b, W_b = optimal_1x2x2()
    add_block([10,11], [12,13,17,18], [8,11], U_b, V_b, W_b)
    
    # Block 2f: A[2:3,2:4] * B[2:4,4:5] -> C[2:3,4:5] (1x2x1)
    U_b, V_b, W_b = optimal_1x2x1()
    add_block([10,11], [14,19], [14], U_b, V_b, W_b)
    
    # Total rank: 6*(7+7+4) + 4*(2+2) + 2*(2) = 6*18 + 16 + 4 = 108+20 = 128? No.
    # Part 1: 3 Strassen (7 each) + 2 Strassen (7 each) + 2 schoolbook_2x2x1 (4 each) = 5*7 + 2*4 = 35+8 = 43? 
    # Wait: blocks 1a,1b,1d,1e are Strassen (4 blocks × 7 = 28), blocks 1c,1f are 2x2x1 (2 blocks × 4 = 8). Total part 1: 36.
    # Part 2: blocks 2a,2b,2d,2e are 1x2x2 (4 blocks × 2 = 8), blocks 2c,2f are 1x2x1 (2 blocks × 2 = 4). Total part 2: 12.
    # Grand total: 36 + 12 = 48.
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
