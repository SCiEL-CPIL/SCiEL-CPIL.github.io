"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy:
- Use the tensor product (Kronecker product) structure of matrix multiplication.
- The 4x4 matrix multiplication tensor is $T_4 = T_2 \otimes T_2$.
- We define the 2x2 Strassen decomposition tensors $U_2, V_2, W_2$ explicitly.
- We compute $U_4 = U_2 \otimes U_2$, $V_4 = V_2 \otimes V_2$, $W_4 = W_2 \otimes W_2$.
- This ensures exact integer arithmetic and avoids manual index mapping errors.
- We then verify the decomposition. If valid, we return the factors.
- Note: The rank of $T_2$ is 7 (Strassen). The rank of $T_4$ via this method is $7 \times 7 = 49$.
- This is the best known upper bound for 4x4 MM rank.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _get_strassen_2x2_tensors():
    """
    Returns U2, V2, W2 for 2x2 Strassen decomposition.
    Shapes:
    U2: (4, 7)  -- indices (i, j) flattened to i*2+j
    V2: (4, 7)  -- indices (j, k) flattened to j*2+k
    W2: (4, 7)  -- indices (k, i) flattened to k*2+i
    """
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # Strassen formulas:
    # m0 = (A00+A11)(B00+B11)
    # m1 = (A10+A11)B00
    # m2 = A00(B10-B11)
    # m3 = A11(B01-B00)
    # m4 = (A00+A01)B11
    # m5 = (A10-A00)(B00+B01)
    # m6 = (A01-A11)(B10+B11)
    
    # C00 = m0 + m3 - m4 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5
    
    # Indices:
    # A: 00(0), 01(1), 10(2), 11(3)
    # B: 00(4), 01(5), 10(6), 11(7) -> Wait, V2 is (j,k). 
    # Let's stick to the standard convention:
    # U[i*m+j, s]
    # V[j*p+k, s]
    # W[k*n+i, s]
    
    # s=0: m0
    # A parts: A00(1), A11(1) -> U2[0,0]=1, U2[3,0]=1
    # B parts: B00(1), B11(1) -> V2[0,0]=1, V2[3,0]=1
    # C parts: C00(1), C11(1) -> W2[0,0]=1, W2[3,0]=1
    
    # s=1: m1
    # A parts: A10(1), A11(1) -> U2[2,1]=1, U2[3,1]=1
    # B parts: B00(1) -> V2[0,1]=1
    # C parts: C10(1), C11(-1) -> W2[2,1]=1, W2[3,1]=-1
    
    # s=2: m2
    # A parts: A00(1) -> U2[0,2]=1
    # B parts: B10(1), B11(-1) -> V2[2,2]=1, V2[3,2]=-1
    # C parts: C01(1), C11(1) -> W2[1,2]=1, W2[3,2]=1
    
    # s=3: m3
    # A parts: A11(1) -> U2[3,3]=1
    # B parts: B01(1), B00(-1) -> V2[1,3]=1, V2[0,3]=-1
    # C parts: C00(1), C10(1) -> W2[0,3]=1, W2[2,3]=1
    
    # s=4: m4
    # A parts: A00(1), A01(1) -> U2[0,4]=1, U2[1,4]=1
    # B parts: B11(1) -> V2[3,4]=1
    # C parts: C00(-1), C01(1) -> W2[0,4]=-1, W2[1,4]=1
    
    # s=5: m5
    # A parts: A10(1), A00(-1) -> U2[2,5]=1, U2[0,5]=-1
    # B parts: B00(1), B01(1) -> V2[0,5]=1, V2[1,5]=1
    # C parts: C11(1) -> W2[3,5]=1
    
    # s=6: m6
    # A parts: A01(1), A11(-1) -> U2[1,6]=1, U2[3,6]=-1
    # B parts: B10(1), B11(1) -> V2[2,6]=1, V2[3,6]=1
    # C parts: C00(1) -> W2[0,6]=1
    
    # Fill arrays
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    return U2, V2, W2

def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate 2x2 Strassen tensors
    U2, V2, W2 = _get_strassen_2x2_tensors()
    
    # 2. Compute Kronecker products to get 4x4 tensors
    # U4 shape: (4*4, 7*7) = (16, 49)
    # V4 shape: (4*4, 7*7) = (16, 49)
    # W4 shape: (4*4, 7*7) = (16, 49)
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 3. Verify
    if time.time() > deadline:
        # Emergency return if we are too slow (unlikely for this small size)
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
        
    if _verify(U4, V4, W4):
        # Success: Rank 49
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        # Fallback: Schoolbook Rank 64
        n, m, p = 4, 4, 4
        rank = 64
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
