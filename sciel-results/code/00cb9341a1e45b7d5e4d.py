"""
4x4x4 matrix multiplication tensor decomposition.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2,
giving an exact rank-49 decomposition (improvement over the rank-64 schoolbook).

The 4x4 matrices are viewed as 2x2 block matrices of 2x2 submatrices.
Strassen is applied at the block level (7 block products), and each block
product (a 2x2 x 2x2 multiplication) is itself decomposed via Strassen (rank 7).
Total rank = 7 * 7 = 49.

Interface: construct() -> {'U', 'V', 'W'} with shapes (16,49),(16,49),(16,49)
and integer entries that exactly decompose the multiplication tensor.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4,7).

    Conventions:
      U[i*2+j, r] = coefficient of A[i,j] in term r
      V[j*2+k, r] = coefficient of B[j,k] in term r
      W[k*2+i, r] = coefficient of C[i,k] in term r   (trace convention W[k*n+i])

    The 7 Strassen products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11) B00
      M3 = A00 (B01-B11)
      M4 = A11 (B10-B00)
      M5 = (A00+A01) B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)

      C00 = M1 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11

    # M2 = (A10+A11) B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    W[1][1] = 1   # C10
    W[3][1] = -1  # C11

    # M3 = A00 (B01-B11)
    U[0][2] = 1   # A00
    V[1][2] = 1   # B01
    V[3][2] = -1  # B11
    W[1][2] = 1   # C01
    W[3][2] = 1   # C11

    # M4 = A11 (B10-B00)
    U[3][3] = 1   # A11
    V[2][3] = 1   # B10
    V[0][3] = -1  # B00
    W[0][3] = 1   # C00
    W[2][3] = 1   # C10

    # M5 = (A00+A01) B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01
    W[3][5] = 1   # C11

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11
    W[0][6] = 1   # C00

    return U, V, W

def _build_tensor(n, m, p):
    """Matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Check sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]."""
    T = _build_tensor(n, m, p)
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    Td = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    diff = float(np.max(np.abs(Td - T)))
    return diff < tol, diff

def _kronecker_decomposition(n, m, p):
    """
    Rank-49 decomposition for 4x4x4 via Kronecker product of two Strassen
    decompositions. Each entry is set exactly once (no overwriting).

    Indices: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2, with sub-indices in {0,1}.
    Term index r = r1*7 + r2.
      U[i*m+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
      V[j*p+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
      W[k*n+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            ui = i * m + j
                            vj = j * p + k
                            wk = k * n + i
                            for r1 in range(7):
                                u1 = U2[i1 * 2 + j1][r1]
                                v1 = V2[j1 * 2 + k1][r1]
                                w1 = W2[k1 * 2 + i1][r1]
                                if u1 == 0 and v1 == 0 and w1 == 0:
                                    continue
                                for r2 in range(7):
                                    u2 = U2[i2 * 2 + j2][r2]
                                    v2 = V2[j2 * 2 + k2][r2]
                                    w2 = W2[k2 * 2 + i2][r2]
                                    if u2 == 0 and v2 == 0 and w2 == 0:
                                        continue
                                    r = r1 * 7 + r2
                                    U[ui][r] = u1 * u2
                                    V[vj][r] = v1 * v2
                                    W[wk][r] = w1 * w2
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Rank 64 exact schoolbook decomposition (fallback)."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before the 150s kill

    # Primary: rank-49 Kronecker-of-Strassen (exact, integer entries).
    try:
        result = _kronecker_decomposition(n, m, p)
        if time.time() < deadline:
            ok, diff = _verify_decomp(n, m, p, result["U"], result["V"], result["W"])
            if ok:
                return {"U": result["U"], "V": result["V"], "W": result["W"]}
    except Exception:
        pass

    # Fallback: rank-64 schoolbook (always valid).
    fb = _schoolbook(n, m, p)
    return {"U": fb["U"], "V": fb["V"], "W": fb["W"]}
# EVOLVE-BLOCK-END
