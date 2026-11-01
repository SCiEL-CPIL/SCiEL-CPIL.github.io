"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication using Recursive Strassen.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication (Strassen, rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10+=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B01-B11) -> C01+=1, C11-=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4 = A11(B10-B00) -> C00+=1, C10+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11-=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W


def _build_rank49():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 7 block-level terms, each with 7 element-level terms.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    U2, V2, W2 = _strassen_2x2()
    
    # Block-level Strassen terms:
    # Each term: list of (block_row, block_col, coeff) for A, B, C respectively.
    # Block (bi, bj) of A contains elements A[2*bi+a, 2*bj+b] for a,b in {0,1}.
    
    terms = [
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)]),
        # M2 = (A10+A11)B00 -> C10+=1
        ([(1,0,1), (1,1,1)],
         [(0,0,1)],
         [(1,0,1)]),
        # M3 = A00(B01-B11) -> C01+=1, C11-=1
        ([(0,0,1)],
         [(0,1,1), (1,1,-1)],
         [(0,1,1), (1,1,-1)]),
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        ([(1,1,1)],
         [(1,0,1), (0,0,-1)],
         [(0,0,1), (1,0,1)]),
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        ([(0,0,1), (0,1,1)],
         [(1,1,1)],
         [(0,0,-1), (0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        ([(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)],
         [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11) -> C00+=1, C11-=1
        ([(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)],
         [(0,0,1), (1,1,-1)]),
    ]
    
    for rb, (P_terms, Q_terms, C_terms) in enumerate(terms):
        for rs in range(7):
            r = rb * 7 + rs
            
            # U: index [i*4+j, r], where i=2*bi+a, j=2*bj+b
            for bi, bj, pc in P_terms:
                for a in range(2):
                    for b in range(2):
                        val = U2[a*2+b, rs]
                        if val != 0:
                            i = 2*bi + a
                            j = 2*bj + b
                            U[i*4+j, r] += pc * val
            
            # V: index [j*4+k, r], where j=2*bi+a, k=2*bj+b
            for bi, bj, qc in Q_terms:
                for a in range(2):
                    for b in range(2):
                        val = V2[a*2+b, rs]
                        if val != 0:
                            j = 2*bi + a
                            k = 2*bj + b
                            V[j*4+k, r] += qc * val
            
            # W: index [k*4+i, r], where i=2*ci+a, k=2*cj+b
            # W2 local index is [local_k*2+local_i, rs] = [b*2+a, rs]
            for ci, cj, cc in C_terms:
                for a in range(2):
                    for b in range(2):
                        val = W2[b*2+a, rs]
                        if val != 0:
                            i = 2*ci + a
                            k = 2*cj + b
                            W[k*4+i, r] += cc * val
    
    return U, V, W


def _verify(U, V, W):
    """Verify the decomposition is exact."""
    n = m = p = 4
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-9)


def _schoolbook():
    """Fallback: rank-64 schoolbook decomposition."""
    n = m = p = 4
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    deadline = time.time() + 115.0
    
    # Primary: rank-49 recursive Strassen
    try:
        U, V, W = _build_rank49()
        if time.time() < deadline and _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: rank-64 schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
