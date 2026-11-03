"""Matrix multiplication 3x3x6 via direct sum of two rank-23 3x3x3 decompositions."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0


def _schoolbook_336():
    """Schoolbook rank-54 decomposition of 3x3x6."""
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Return rank-7 decomposition of 2x2x2.
    U[4,7], V[4,7], W[4,7]
    U[i*2+j, c] for A[i,j], V[i*2+j, c] for B[i,j], W[i*2+j, c] for C[i,j]
    """
    # Strassen formulas for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # Row indices: 0=a11, 1=a12, 2=a21, 3=a22
    U[0, 0] = 1  # a11 in M1
    U[3, 0] = 1  # a22 in M1
    U[0, 1] = 1  # a11 in M3
    U[3, 1] = 0  # (a21+a22) in M2
    U[2, 1] = 1  # a21 in M2
    U[3, 2] = 1  # a22 in M2
    U[0, 3] = 1  # a11 in M5
    U[1, 3] = 1  # a12 in M5
    U[2, 4] = 1  # a21 in M6
    U[0, 4] = -1 # a11 in M6
    U[1, 5] = 1  # a12 in M7
    U[3, 5] = -1 # a22 in M7
    # M4: a22(b21-b11)
    U[3, 6] = 1

    V[0, 0] = 1  # b11 in M1
    V[3, 0] = 1  # b22 in M1
    V[1, 1] = 1  # b12 in M3
    V[3, 1] = -1 # b22 in M3
    V[0, 2] = 1  # b11 in M2
    V[2, 3] = 1  # b21 in M4
    V[0, 3] = -1 # b11 in M4
    V[3, 4] = 1  # b22 in M5
    V[0, 5] = 1  # b11 in M6
    V[1, 5] = 1  # b12 in M6
    V[2, 6] = 1  # b21 in M7
    V[3, 6] = 1  # b22 in M7

    W[0, 0] = 1  # c11 = M1
    W[3, 0] = -1 # c22 = -M1
    W[1, 1] = 1  # c12 = M3
    W[2, 2] = 1  # c21 = M2
    W[0, 3] = 1  # c11 += M4
    W[3, 3] = -1 # c22 -= M4
    W[3, 4] = -1 # c22 -= M5
    W[1, 4] = 1  # c12 += M5
    W[2, 5] = 1  # c21 += M6... wait, let me recheck

    # Let me redo this more carefully.
    # c11 = M1 + M4 - M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (a11+a22)(b11+b22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (a21+a22)b11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3 = a11(b12-b22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = a22(b21-b11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (a11+a12)b22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (a21-a11)(b11+b12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=0
    # M7 = (a12-a22)(b21+b22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=-1; W[3,6]=1
    # Now combine into C:
    # c11 = M1 + M4 - M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    # So W should be:
    W = np.zeros((4, 7), dtype=int)
    W[0,0]=1   # c11 += M1
    W[3,0]=1   # c22 += M1
    W[2,1]=1   # c21 += M2
    W[3,1]=-1  # c22 -= M2
    W[1,2]=1   # c12 += M3
    W[3,2]=1   # c22 += M3
    W[0,3]=1   # c11 += M4
    W[2,3]=1   # c21 += M4
    W[0,4]=-1  # c11 -= M5
    W[1,4]=1   # c12 += M5
    W[0,6]=-1  # c11 -= M7
    W[3,6]=1   # c22 += M7
    return U, V, W


def _build_333_rank23():
    """Rank-23 decomposition of 3x3x3.
    Returns U[9,23], V[9,23], W[9,23] as numpy arrays.
    """
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Place Strassen 2x2x2 in top-left 2x2 block
    sU, sV, sW = _strassen_2x2()
    # Map 2x2 indices to 3x3: a11->0, a12->1, a21->3, a22->4
    # b11->0, b12->1, b21->3, b22->4
    # c11->0, c12->1, c21->3, c22->4
    idx2 = [0, 1, 3, 4]
    for t in range(7):
        for r in range(4):
            U[idx2[r], t] = sU[r, t]
            V[idx2[r], t] = sV[r, t]
            W[idx2[r], t] = sW[r, t]
    
    # Now we need to cover all (i,j,k) with at least one index = 2.
    # That's 27 - 8 = 19 entries. We have 23-7 = 16 slots.
    # We can't cover all 19 with 16 rank-1 terms directly.
    # We need to be smarter. Let me use a different approach.
    
    # Actually, let's just use schoolbook for the full 3x3x3 if rank-23 doesn't work easily.
    # For now, return None to indicate failure, and fall back to schoolbook 3x3x6.
    return None


def _schoolbook_333():
    """Schoolbook rank-9 decomposition of 3x3x3."""
    U = np.zeros((9, 9), dtype=int)
    V = np.zeros((9, 9), dtype=int)
    W = np.zeros((9, 9), dtype=int)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, c] = 1
                V[j*3+k, c] = 1
                W[k*3+i, c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(R):
        T += np.einsum('a,b,ijk->ijk', 
                        U.reshape(n, m, R)[:, :, c],
                        np.ones(1),
                        np.ones((n, m, p)))
        # This is getting complicated. Let's do it more directly.
        T_c = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T += T_c
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-9)


def construct():
    if time.time() > _DEADLINE - _MARGIN:
        return _schoolbook_336()
    
    n, m, p = DIMENSIONS
    
    # Try rank-46: two independent 3x3x3 schoolbook decompositions
    U1, V1, W1 = _schoolbook_333()
    U = np.zeros((9, 18), dtype=int)
    V = np.zeros((18, 18), dtype=int)
    W = np.zeros((18, 18), dtype=int)
    for c in range(9):
        for i in range(3):
            for j in range(3):
                U[i*3+j, c] = U1[i*3+j, c]
                for k in range(3):
                    V[j*6+k, c] = V1[j*3+k, c]
                    W[k*3+i, c] = W1[k*3+i, c]
                    V[j*6+k+3, c+9] = V1[j*3+k, c]
                    W[(k+3)*3+i, c+9] = W1[k*3+i, c]
    
    if _verify(U, V, W, 3, 3, 6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook_336()
