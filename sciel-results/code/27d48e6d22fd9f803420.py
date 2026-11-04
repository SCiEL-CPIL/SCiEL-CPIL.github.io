import numpy as np
import time

DIMENSIONS = [4, 4, 8]
_DEADLINE = 140.0  # Safety margin before 150s kill

def _build_strassen_2x2():
    """Build rank-7 Strassen decomposition for 2x2x2.
    
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01) -> C11 += M6
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11) -> C00 += M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    return U2, V2, W2

def _build_4x4x4_from_strassen(U2, V2, W2):
    """
    Build 4x4x4 decomposition from 2x2x2 Strassen via tensor product.
    
    Index mapping: i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    V4[j*4+k, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    W4[k*4+i, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    """
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    ua = U2[i1*2 + j1, r1]
                    if ua == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            ub = U2[i2*2 + j2, r2]
                            if ub == 0:
                                continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U4[i*4 + j, r] = ua * ub
            for j1 in range(2):
                for k1 in range(2):
                    va = V2[j1*2 + k1, r1]
                    if va == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            vb = V2[j2*2 + k2, r2]
                            if vb == 0:
                                continue
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V4[j*4 + k, r] = va * vb
            for k1 in range(2):
                for i1 in range(2):
                    wa = W2[k1*2 + i1, r1]
                    if wa == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            wb = W2[k2*2 + i2, r2]
                            if wb == 0:
                                continue
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W4[k*4 + i, r] = wa * wb
    return U4, V4, W4

def _verify_decomp(U, V, W, n, m, p, seed=42):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = U.shape[1]
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        C_computed += w_r.T * (u_r @ v_r)
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses Strassen's algorithm for 4x4x4 blocks (rank 49) applied to each of two
    4x4x4 sub-problems (left and right halves of the 4x4x8 output).
    
    Indexing convention:
    - U: (n*m, rank) = (16, 98), U[i*m+j, r] is coefficient of A[i,j] in product r
    - V: (m*p, rank) = (32, 98), V[j*p+k, r] is coefficient of B[j,k] in product r
    - W: (p*n, rank) = (32, 98), W[k*n+i, r] is coefficient of product r in C[i,k]
    """
    t0 = time.time()
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    U2, V2, W2 = _build_strassen_2x2()
    
    # Build 4x4x4 decomposition from two-level Strassen (rank 49)
    U4, V4, W4 = _build_4x4x4_from_strassen(U2, V2, W2)
    
    if time.time() - t0 > _DEADLINE:
        pass
    
    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
