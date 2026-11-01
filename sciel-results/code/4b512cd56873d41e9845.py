"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    rank = 7
    # U shape (n*m, rank) -> (4, 7)
    # V shape (m*p, rank) -> (4, 7)
    # W shape (p*n, rank) -> (4, 7)
    
    # Indices for U, V: i*m+j, j*p+k
    # Indices for W: k*n+i
    
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]
    
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]
    
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0] (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] (k=0, i=1) -- Wait, W index is k*n+i.
        # W[0] is k=0, i=0 -> C[0,0]
        # W[1] is k=0, i=1 -> C[1,0]? No.
        # W index = k*n + i.
        # n=2.
        # k=0, i=0 -> 0. C[0,0]
        # k=0, i=1 -> 1. C[1,0]
        # k=1, i=0 -> 2. C[0,1]
        # k=1, i=1 -> 3. C[1,1]
        
        # Let's re-map W_data based on k*n+i.
        # Row 0: k=0, i=0 (C[0,0])
        # Row 1: k=0, i=1 (C[1,0])
        # Row 2: k=1, i=0 (C[0,1])
        # Row 3: k=1, i=1 (C[1,1])
        
        # My previous w vectors were:
        # w1: C00=1, C11=1 -> [1, 0, 0, 1]
        # w2: C10=1, C11=-1 -> [0, 1, 0, -1]
        # w3: C01=1, C11=1 -> [0, 0, 1, 1]
        # w4: C00=1, C10=1 -> [1, 1, 0, 0]
        # w5: C00=-1, C01=1 -> [-1, 0, 1, 0]
        # w6: C11=1 -> [0, 0, 0, 1]
        # w7: C00=1 -> [1, 0, 0, 0]
        
        # Let's check the mapping again.
        # W[k*n+i, c]
        # c=0: w1 = [1, 0, 0, 1]
        # c=1: w2 = [0, 1, 0, -1]
        # c=2: w3 = [0, 0, 1, 1]
        # c=3: w4 = [1, 1, 0, 0]
        # c=4: w5 = [-1, 0, 1, 0]
        # c=5: w6 = [0, 0, 0, 1]
        # c=6: w7 = [1, 0, 0, 0]
    ]
    
    # Correct W_data rows:
    # Row 0 (C00): w1[0]=1, w2[0]=0, w3[0]=0, w4[0]=1, w5[0]=-1, w6[0]=0, w7[0]=1
    # Row 1 (C10): w1[1]=0, w2[1]=1, w3[1]=0, w4[1]=1, w5[1]=0, w6[1]=0, w7[1]=0
    # Row 2 (C01): w1[2]=0, w2[2]=0, w3[2]=1, w4[2]=0, w5[2]=1, w6[2]=0, w7[2]=0
    # Row 3 (C11): w1[3]=1, w2[3]=-1, w3[3]=1, w4[3]=0, w5[3]=0, w6[3]=1, w7[3]=0
    
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)
    
    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 8
        rank = 8
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
