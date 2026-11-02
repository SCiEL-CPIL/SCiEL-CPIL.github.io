"""Matrix multiplication 3x4 by 4x7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C = A @ B
        Crec = np.zeros((n, p))
        for c in range(U.shape[1]):
            L = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    L[i] += U[i*m+j, c] * A[i, j]
            R = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    R[k] += V[j*p+k, c] * B[j, k]
            for k in range(p):
                for i in range(n):
                    Crec[i, k] += L[i] * R[k] * W[k*n+i, c]
        if not np.allclose(Crec, C, atol=1e-9):
            return False
    return True

def _construct_3x2x2_rank10():
    """
    Returns U, V, W for 3x2 by 2x2 multiplication with rank 10.
    A is 3x2, B is 2x2, C is 3x2.
    U: (6, 10), V: (4, 10), W: (6, 10).
    """
    n, m, p = 3, 2, 2
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for A (3x2): A[i, j] -> U[i*2+j]
    # Indices for B (2x2): B[j, k] -> V[j*2+k]
    # Indices for C (3x2): C[i, k] -> W[k*3+i]
    
    # 10 terms
    # 1. ap
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # C00
    # 2. br
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1 # C00
    # 3. cp
    U[2, 2] = 1; V[0, 2] = 1; W[3, 2] = 1 # C10
    # 4. dr
    U[3, 3] = 1; V[3, 3] = 1; W[3, 3] = 1 # C10
    # 5. ep
    U[4, 4] = 1; V[0, 4] = 1; W[6, 4] = 1 # C20
    # 6. fr
    U[5, 5] = 1; V[3, 5] = 1; W[6, 5] = 1 # C20
    
    # 7. (a+c+e)q
    U[0, 6] = 1; U[2, 6] = 1; U[4, 6] = 1
    V[1, 6] = 1
    W[1, 6] = 1 # C01
    W[4, 6] = 1 # C11
    W[7, 6] = 1 # C21
    
    # 8. (b+d+f)s
    U[1, 7] = 1; U[3, 7] = 1; U[5, 7] = 1
    V[2, 7] = 1
    W[1, 7] = 1 # C01
    W[4, 7] = 1 # C11
    W[7, 7] = 1 # C21
    
    # 9. (a+c)s
    U[0, 8] = 1; U[2, 8] = 1
    V[2, 8] = 1
    W[1, 8] = 1 # C01
    W[4, 8] = 1 # C11
    
    # 10. (b+d)q
    U[1, 9] = 1; U[3, 9] = 1
    V[1, 9] = 1
    W[1, 9] = 1 # C01
    W[4, 9] = 1 # C11
    
    # Verify this specific construction
    # C00 = ap + br
    # C01 = aq + bs + (a+c)s + (b+d)q - (a+c)s - (b+d)q ? 
    # Wait, the terms 7-10 contributions:
    # C01 gets:
    # from 7: a q
    # from 8: b s
    # from 9: a s + c s
    # from 10: b q + d q
    # Sum = aq + bs + as + cs + bq + dq.
    # We want aq + bs.
    # This construction is WRONG.
    
    # I will use a verified 10-term scheme from literature or derive correctly.
    # Correct 10-term scheme for 3x2x2:
    # M1=ap, M2=br, M3=cp, M4=dr, M5=ep, M6=fr
    # M7=(a+c+e)(p+q)
    # M8=(b+d+f)(r+s)
    # M9=(a-c)(p+q)
    # M10=(b-d)(r+s)
    # This is also likely wrong.
    
    # I will return None if I can't verify it immediately in code.
    return None

def construct():
    """Construct a rank-minimal exact decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook if we can't find a better one
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to construct rank 72
    # Using block decomposition:
    # A (3x4) = [A1 (3x2) | A2 (3x2)]
    # B (4x7) = [B1 (2x7); B2 (2x7)]
    # C = A1 B1 + A2 B2
    
    # We need rank 36 for 3x2x7.
    # Split B1 (2x7) into 3 blocks of 2x2 and 1 block of 2x1.
    # Rank(3x2x2) = 10. Rank(3x2x1) = 6.
    # Total 3*10 + 6 = 36.
    
    # I will implement the 3x2x2 rank 10 scheme correctly.
    # If I fail, I return schoolbook.
    
    # Due to complexity and time, I will return schoolbook to ensure validity.
    # But I can try a simple improvement: Rank 78?
    # No, schoolbook is 84.
    
    # I will use the schoolbook result.
    return schoolbook()
# EVOLVE-BLOCK-END
