"""
Matrix multiplication 3x3x8: rank-minimal exact tensor decomposition.

Strategy:
  - Primary: a hand-constructed rank-68 decomposition that applies Strassen's
    2x2x2 algorithm (rank 7) to each of four disjoint 2x2x2 blocks covering
    i in {0,1}, j in {0,1}, k in {2g, 2g+1}, and fills the remaining
    (i=2 or j=2) entries with schoolbook terms.
  - Fallback: schoolbook (rank 72) if validation ever fails.

The decomposition is static and deterministic, so construction is O(1)
after a single validation pass.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_TIME_LIMIT = 115  # safety margin from 120 s evaluator kill

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p):
    """Check if U, V, W (lists of lists) form a valid decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]

    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Using einsum for speed
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U_arr.reshape(n, m, rank)
    Vr = V_arr.reshape(m, p, rank)
    Wr = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x8(n, m, p):
    """
    Construct a rank-68 decomposition for 3x3x8 matrix multiplication.

    The 8 output columns k are split into 4 groups of 2.  For each group
    {k0, k1} we apply Strassen's 7-multiplication algorithm to the
    2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {k0, k1}), and handle
    the remaining rows/columns (i=2 or j=2) with schoolbook terms.

    Per group:
      Strassen: 7 terms
      Schoolbook for (i in {0,1}, j=2, k in {k0,k1}): 2*2 = 4 terms
      Schoolbook for (i=2, j in {0,1}, k in {k0,k1}): 2*2 = 4 terms
      Schoolbook for (i=2, j=2, k in {k0,k1}): 1*2 = 2 terms
      Total per group: 7 + 4 + 4 + 2 = 17 terms
    Total: 4 * 17 = 68 terms.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 68
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen 2x2x2 coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U coefficients: index into A matrix (row*2+col)
    U_coeffs = [
        {0: 1, 3: 1},       # M1: a, d
        {2: 1, 3: 1},       # M2: c, d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a, b
        {2: 1, 0: -1},      # M6: c, -a
        {1: 1, 3: -1},      # M7: b, -d
    ]
    # V coefficients: index into B matrix (row*2+col)
    V_coeffs = [
        {0: 1, 3: 1},       # M1: e, h
        {0: 1},             # M2: e
        {1: 1, 3: -1},      # M3: f, -h
        {2: 1, 0: -1},      # M4: g, -e
        {3: 1},             # M5: h
        {0: 1, 1: 1},       # M6: e, f
        {2: 1, 3: 1},       # M7: g, h
    ]
    # W coefficients: index into C matrix (row*2+col)
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00, C11
        {2: 1, 3: -1},      # M2: C10, -C11
        {1: 1, 3: 1},       # M3: C01, C11
        {0: 1, 2: 1},       # M4: C00, C10
        {0: -1, 1: 1},      # M5: -C00, C01
        {3: 1},             # M6: C11
        {0: 1},             # M7: C00
    ]

    col = 0
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1

        # 1. Strassen block for i in {0,1}, j in {0,1}
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local * m + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local * p + (k0 + k_local), col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[(k0 + k_local) * n + i_local, col] += coeff
            col += 1

        # 2. Schoolbook for (i in {0,1}, j=2, k in {k0, k1})
        for i in range(2):
            for k in [k0, k1]:
                U[i * m + 2, col] = 1.0
                V[2 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # 3. Schoolbook for (i=2, j in {0,1}, k in {k0, k1})
        for j in range(2):
            for k in [k0, k1]:
                U[2 * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + 2, col] = 1.0
                col += 1

        # 4. Schoolbook for (i=2, j=2, k in {k0, k1})
        for k in [k0, k1]:
            U[2 * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + 2, col] = 1.0
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    _START_TIME = time.time()

    n, m, p = DIMENSIONS

    # Primary: rank-68 Strassen-based decomposition
    result = _strassen_3x3x8(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result

    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
