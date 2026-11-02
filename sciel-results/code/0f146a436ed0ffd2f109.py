"""Block-decomposition program for 4x4 @ 4x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 rank-1 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is (U dict, V dict, W dict) with integer entries."""
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # M2 = (A21+A22)B11
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    # M3 = A11(B21-B22)
    terms.append(({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    # M4 = A22(B12-B11)
    terms.append(({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    # M5 = (A11+A12)B22
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    # M6 = (A21-A11)(B11+B12)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # M7 = (A12-A22)(B21+B22)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _strassen_naive_2x2x3():
    """Return 9 rank-1 terms for 2x2 A, 2x3 B -> 2x3 C.
    Strassen on the first 2 columns of B (7 terms) + naive on the 3rd column (2 terms).
    All entries are integers."""
    terms = []
    for (u, v, w) in _strassen_2x2():
        # Strassen term on B columns 0,1 -> C columns 0,1
        terms.append((u, v, w))
    # Naive term for C[:,2]: A[i,:] * B[:,2]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    terms.append(({(0,0): 1, (0,1): 1}, {(0,2): 1, (1,2): 1}, {(0,2): 1}))
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    terms.append(({(1,0): 1, (1,1): 1}, {(0,2): 1, (1,2): 1}, {(1,2): 1}))
    return terms

def _build_block_decomp():
    """Build a verified 4x4 @ 4x5 decomposition using block decomposition.
    Split into 4 blocks of (2x2 A, 2x2 or 2x3 B). Rank = 7+9+7+9 = 32.
    Returns (U, V, W) as lists of lists of ints, or None if verification fails."""
    n, m, p = DIMENSIONS
    S22 = _strassen_2x2()
    S223 = _strassen_naive_2x2x3()
    
    blocks = [
        (0, 0, 0, 2, 0, 2, S22),   # i0-1, j0-1, k0-1
        (0, 2, 2, 4, 0, 3, S223),  # i0-1, j2-3, k0-2
        (2, 0, 0, 2, 2, 4, S22),   # i2-3, j0-1, k2-3
        (2, 2, 2, 4, 2, 5, S223),  # i2-3, j2-3, k2-4
    ]
    
    rank = sum(len(s) for (_, _, _, _, _, _, s) in blocks)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, j0, k0, i1, j1, k1, terms) in blocks:
        for (u, v, w) in terms:
            for (bi, bj), cu in u.items():
                i = i0 + bi
                U[i * m + (j0 + bj)][col] += cu
            for (bj, bk), cv in v.items():
                j = j0 + bj
                V[j * p + (k0 + bk)][col] += cv
            for (bk, bi), cw in w.items():
                k = k0 + bk
                W[k * n + (i0 + bi)][col] += cw
            col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j] * Va[j*p+k], Wa[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_halfint(U, V, W, tol=1e-6):
    for M in (U, V, W):
        for row in M:
            for x in row:
                if abs(2.0 * x - round(2.0 * x)) > tol:
                    return False
    return True

def _schoolbook():
    """Fallback schoolbook decomposition rank n*m*p = 80."""
    n, m, p = DIMENSIONS
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
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start = time.time()
    n, m, p = DIMENSIONS

    # Primary: verified block decomposition (rank 32)
    try:
        U, V, W = _build_block_decomp()
        if _verify(U, V, W, n, m, p) and _verify_halfint(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: schoolbook (rank 80)
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
