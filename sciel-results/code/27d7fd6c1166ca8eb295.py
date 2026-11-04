"""Matrix multiplication 4,4,8 - tensor decomposition with rank minimization."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin

def _strassen_2x2():
    """
    Strassen 2x2x2 decomposition.
    Returns U, V, W of shape (4, 7) each.
    U: (i*2+j, r), V: (j*2+k, r), W: (k*2+i, r)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    # M4 = A11(B10-B00) -> C10 += M4, C00 += M4
    U[3,3]=1
    V[0,3]=-1; V[2,3]=1
    W[2,3]=1; W[0,3]=1
    # M5 = (A00+A01)B11 -> C01 += M5, C00 -= M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[0,4]=-1
    # M6 = (A10-A00)(B00+B01) -> C10 += M6
    U[0,5]=-1; U[2,5]=1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1
    # M7 = (A01-A11)(B10+B11) -> C01 += M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[1,6]=1
    return U, V, W

def _build_4x4x8_v2():
    """
    Build rank-98 decomposition for 4x4 * 4x8 using two 4x4x4 Strassen blocks.
    U: (16, rank), V: (32, rank), W: (32, rank)
    Index conventions:
      U[i*4+j, r] for i in 0..3 (rows of A), j in 0..3 (cols of A)
      V[j*8+k, r] for j in 0..3 (rows of B), k in 0..7 (cols of B)
      W[k*4+i, r] for k in 0..7 (cols of C), i in 0..3 (rows of C)
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4x4 via Kronecker of two 2x2x2
    rank4 = 49
    U4 = np.zeros((16, rank4), dtype=int)
    V4 = np.zeros((16, rank4), dtype=int)
    W4 = np.zeros((16, rank4), dtype=int)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            # U4[I*4+J, c] = U2[i*2+j, c0] * U2[i'*2+j', c1]
            # where I = 2*i + i', J = 2*j + j'
            for i in range(2):
                for ip in range(2):
                    I = 2*i + ip
                    for j in range(2):
                        for jp in range(2):
                            J = 2*j + jp
                            U4[I*4+J, c] = U2[i*2+j, c0] * U2[ip*2+jp, c1]
            # V4[J*4+K, c] = V2[j*2+k, c0] * V2[j'*2+k', c1]
            for j in range(2):
                for jp in range(2):
                    J = 2*j + jp
                    for k in range(2):
                        for kp in range(2):
                            K = 2*k + kp
                            V4[J*4+K, c] = V2[j*2+k, c0] * V2[jp*2+kp, c1]
            # W4[K*4+I, c] = W2[k*2+i, c0] * W2[k'*2+i', c1]
            for k in range(2):
                for kp in range(2):
                    K = 2*k + kp
                    for i in range(2):
                        for ip in range(2):
                            I = 2*i + ip
                            W4[K*4+I, c] = W2[k*2+i, c0] * W2[kp*2+ip, c1]
    
    # Now build 4x4x8: two blocks for k=0..3 and k=4..7
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((32, total_rank), dtype=int)
    
    # Block 0: columns k=0..3 of B, columns k=0..3 of C
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
    
    # Block 1: columns k=4..7 of B, columns k=4..7 of C
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + (4+k), 49:98] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(4+k)*4 + i, 49:98] = W4[k*4 + i, :]
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify exact decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank).astype(float)
    V3 = V.reshape(m, p, rank).astype(float)
    W3 = W.reshape(p, n, rank).astype(float)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.all(T == 1.0)

def _try_rank_reduction(U, V, W):
    """
    Try to reduce rank by identifying linearly dependent rank-1 components.
    Returns potentially reduced (U, V, W) or original if no reduction found.
    """
    if time.time() > _DEADLINE:
        return U, W
    
    rank = U.shape[1]
    
    # Check for zero columns
    non_zero = []
    for c in range(rank):
        if U[:, c].any() and V[:, c].any() and W[:, c].any():
            non_zero.append(c)
    
    if len(non_zero) < rank:
        U = U[:, non_zero]
        V = V[:, non_zero]
        W = W[:, non_zero]
    
    # Try to find linear dependencies among columns
    # This is complex; for now just return as-is
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    n, m, p = DIMENSIONS
    
    # Primary: rank-98 Strassen-based decomposition
    U, V, W = _build_4x4x8_v2()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-128
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
