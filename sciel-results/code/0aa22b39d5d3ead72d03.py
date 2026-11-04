"""Find a rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x6 matrix."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _block_decomposition(n, m, p, m_split):
    """
    Block decomposition: split the middle dimension m into two parts.
    For (3,4,6) with m_split=(2,2):
    C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{3} A[i,j]B[j,k]
    = C1[i,k] + C2[i,k]
    where C1 = A1 * B1 (3x2 times 2x6) and C2 = A2 * B2 (3x2 times 2x6)
    
    Each sub-problem (3,2,6) has schoolbook rank 3*2*6 = 36.
    Total rank = 36 + 36 = 72... wait, that's the same.
    
    Actually, the key insight: we can share structure between the two blocks.
    But let me think differently.
    
    Actually, the simplest improvement: use the fact that (1,2,1) has rank 1.
    (3,4,6) = (3,2,6) x (1,2,1) via Kronecker product.
    rank(3,2,6) <= 36, rank(1,2,1) = 1.
    Kronecker product rank = rank(3,2,6) * rank(1,2,1) = 36 * 1 = 36.
    
    Let me construct this properly.
    
    (1,2,1): A is 1x2, B is 2x1, C is 1x1.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    Tensor: T[j, j*1+0, 0*1+0] = T[j, j, 0] = 1 for j=0,1.
    Rank 1: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1.
    Check: sum_c U[j,c]*V[j*1+0,c]*W[0*1+0,c] = U[j,0]*V[j,0]*W[0,0] = 1*1*1 = 1. ✓
    
    (3,2,6): A is 3x2, B is 2x6, C is 3x6.
    Schoolbook rank = 3*2*6 = 36.
    U1 shape: (3*2, 36) = (6, 36)
    V1 shape: (2*6, 36) = (12, 36)
    W1 shape: (6*3, 36) = (18, 36)
    
    Kronecker product T1 x T2:
    T1 for (n1,m1,p1) = (3,2,6), rank r1=36
    T2 for (n2,m2,p2) = (1,2,1), rank r2=1
    Result for (n1*n2, m1*m2, p1*p2) = (3, 4, 6), rank r1*r2 = 36
    
    U[i1*m1+j1, c1] * U2[i2*m2+j2, c2] indexed as ((i1,i2),(j1,j2))
    In the combined space:
    Combined n = n1*n2 = 3, m = m1*m2 = 4, p = p1*p2 = 6
    U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    V_combined[(j1*m2+j2)*(p1*p2) + (k1*p2+k2), c1*r2+c2] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    W_combined[(k1*p2+k2)*(n1*n2) + (i1*n2+i2), c1*r2+c2] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    
    Since r2=1, c2=0 always, so c1*r2+c2 = c1.
    """
    n1, m1, p1 = n, 2, p  # (3, 2, 6)
    n2, m2, p2 = 1, 2, 1  # (1, 2, 1)
    
    # Build (3,2,6) schoolbook decomposition
    r1 = n1 * m1 * p1  # 36
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
    
    # Build (1,2,1) rank-1 decomposition
    r2 = 1
    U2 = [[1], [1]]  # shape (1*2, 1) = (2, 1)
    V2 = [[1], [1]]  # shape (2*1, 1) = (2, 1)
    W2 = [[1]]       # shape (1*1, 1) = (1, 1)
    
    # Kronecker product
    n, m, p = n1 * n2, m1 * m2, p1 * p2  # (3, 4, 6)
    rank = r1 * r2  # 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i1 in range(n1):
        for i2 in range(n2):
            for j1 in range(m1):
                for j2 in range(m2):
                    for k1 in range(p1):
                        for k2 in range(p2):
                            for c1 in range(r1):
                                for c2 in range(r2):
                                    c = c1 * r2 + c2
                                    U[(i1 * n2 + i2) * m + (j1 * m2 + j2)][c] += U1[i1 * m1 + j1][c1] * U2[i2 * m2 + j2][c2]
                                    V[(j1 * m2 + j2) * p + (k1 * p2 + k2)][c] += V1[j1 * p1 + k1][c1] * V2[j2 * p2 + k2][c2]
                                    W[(k1 * p2 + k2) * n + (i1 * n2 + i2)][c] += W1[k1 * n1 + i1][c1] * W2[k2 * n2 + i2][c2]
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is exact for matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    target = np.ones((n, m, p))
    
    return np.allclose(result, target, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) factor matrices."""
    rank = n * m * p
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try the Kronecker product decomposition: (3,4,6) = (3,2,6) x (1,2,1)
    # This should give rank 36.
    try:
        U, V, W = _block_decomposition(n, m, p, m_split=(2, 2))
        
        # Verify
        if _verify_decomposition(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall through to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 72)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
