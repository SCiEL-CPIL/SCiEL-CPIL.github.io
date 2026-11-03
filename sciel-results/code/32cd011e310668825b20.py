import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: u = i*2+j, v = j*2+k, w = k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each product, set U, V, W entries
    # Product 1: (a11+a22)(b11+b22)
    # U: a11 at (0,0), a22 at (1,1) -> U[0,0]=1, U[3,0]=1
    # V: b11 at (0,0), b22 at (1,1) -> V[0,0]=1, V[3,0]=1
    # W: contributes to C11 (+1), C22 (+1) -> W[0,0]=1, W[3,0]=1
    
    # Product 2: (a21+a22)b11
    # U: a21 at (1,0), a22 at (1,1) -> U[2,1]=1, U[3,1]=1
    # V: b11 at (0,0) -> V[0,1]=1
    # W: C21 (+1), C22 (-1) -> W[1,1]=1, W[3,1]=-1
    
    # Product 3: a11(b21-b22)
    # U: a11 at (0,0) -> U[0,2]=1
    # V: b21 at (1,0), b22 at (1,1) -> V[2,2]=1, V[3,2]=-1
    # W: C12 (+1), C22 (+1) -> W[2,2]=1, W[3,2]=1
    
    # Product 4: a22(b12-b11)
    # U: a22 at (1,1) -> U[3,3]=1
    # V: b12 at (0,1), b11 at (0,0) -> V[1,3]=1, V[0,3]=-1
    # W: C11 (+1), C21 (+1) -> W[0,3]=1, W[1,3]=1
    
    # Product 5: (a11+a12)b22
    # U: a11 at (0,0), a12 at (0,1) -> U[0,4]=1, U[1,4]=1
    # V: b22 at (1,1) -> V[3,4]=1
    # W: C11 (-1), C12 (+1) -> W[0,4]=-1, W[2,4]=1
    
    # Product 6: (a21-a11)(b11+b12)
    # U: a21 at (1,0), a11 at (0,0) -> U[2,5]=1, U[0,5]=-1
    # V: b11 at (0,0), b12 at (0,1) -> V[0,5]=1, V[1,5]=1
    # W: C22 (+1) -> W[3,5]=1
    
    # Product 7: (a12-a22)(b21+b22)
    # U: a12 at (0,1), a22 at (1,1) -> U[1,6]=1, U[3,6]=-1
    # V: b21 at (1,0), b22 at (1,1) -> V[2,6]=1, V[3,6]=1
    # W: C11 (+1) -> W[0,6]=1
    
    # Fill in
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
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
    return U, V, W

def _decompose_3x3x3():
    """Decompose 3x3x3 using Strassen for 2x2 blocks.
    Split A into 2x2, 2x1, 1x2, 1x1 blocks.
    Use Strassen for 2x2x2, schoolbook for others.
    Returns (U, V, W) with shapes (9, r), (9, r), (9, r)
    """
    # A is 3x3, B is 3x3, C is 3x3
    # Split indices: first 2, last 1
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (2x1x2, schoolbook rank 4)
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 2] (2x1x2, schoolbook rank 4)
    # A[2, 2] * B[2, 2] -> C[2, 2] (1x1x1, schoolbook rank 1)
    # A[0:2, 2] * B[0:2, 2] -> C[0:2, 2] (2x2x2, Strassen rank 7)
    # A[2, 0:2] * B[0:2, 2] -> C[2, 2] (2x2x2, Strassen rank 7)
    # A[0:2, 0:2] * B[2, 0:2] -> C[0:2, 2] (2x1x2, schoolbook rank 4)
    # A[2, 2] * B[0:2, 2] -> C[2, 2] (1x2x2, schoolbook rank 4)
    
    # Wait, let me be more careful.
    # C = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's split i, j, k into {0,1} and {2}
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2]*B[2, 0:2]
    # C[0:2, 2] = A[0:2, 0:2]*B[0:2, 2] + A[0:2, 2]*B[2, 2]
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2]*B[2, 0:2]
    # C[2, 2] = A[2, 0:2]*B[0:2, 2] + A[2, 2]*B[2, 2]
    
    # Terms:
    # 1. A[0:2,0:2]*B[0:2,0:2] -> 2x2x2, Strassen rank 7
    # 2. A[0:2,2]*B[2,0:2] -> 2x1x2, schoolbook rank 4
    # 3. A[0:2,0:2]*B[0:2,2] -> 2x2x2, Strassen rank 7
    # 4. A[0:2,2]*B[2,2] -> 2x1x1, schoolbook rank 2
    # 5. A[2,0:2]*B[0:2,0:2] -> 2x2x2, Strassen rank 7
    # 6. A[2,2]*B[2,0:2] -> 1x1x2, schoolbook rank 2
    # 7. A[2,0:2]*B[0:2,2] -> 2x2x2, Strassen rank 7
    # 8. A[2,2]*B[2,2] -> 1x1x1, schoolbook rank 1
    # Total: 7+4+7+2+7+2+7+1 = 37
    
    # That's worse than schoolbook (27). Let me reconsider.
    # The issue is that Strassen saves 1 product for 2x2x2 (7 vs 8), but we have many blocks.
    
    # Alternative: just use schoolbook for 3x3x3, rank 27.
    # Or try a smarter decomposition.
    
    # Let me just use schoolbook for now and focus on the 3x3x7 decomposition.
    U, V, W = _schoolbook_block(3, 3, 3)
    return U, V, W

def _decompose_3x3x7():
    """Decompose 3x3x7 by splitting into 3x3x3 + 3x3x4.
    Use Strassen-based decomposition for each part.
    """
    # 3x3x3: use schoolbook for now, rank 27
    U1, V1, W1 = _decompose_3x3x3()
    
    # 3x3x4: split into 3x3x2 + 3x3x2
    # 3x3x2: schoolbook rank 18, or try Strassen-based
    # Let's use schoolbook for 3x3x4: rank 36
    U2, V2, W2 = _schoolbook_block(3, 3, 4)
    
    # Concatenate
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try the hybrid decomposition
    U, V, W = _decompose_3x3x7()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _schoolbook_block(3, 3, 7)
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

def _verify(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(reconstructed, T, atol=1e-9)

def _schoolbook(n, m, p):
    U, V, W = _schoolbook_block(n, m, p)
    return U.tolist(), V.tolist(), W.tolist()
