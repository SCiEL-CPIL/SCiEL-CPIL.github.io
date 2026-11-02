"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication.
    
    For C = A*B where A, B, C are 2x2:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    terms = []
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}})
    terms.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}})
    terms.append({'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}})
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}})
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}})
    return terms

def _strassen_2x2x3():
    """Return a rank-9 decomposition for 2x2 * 2x3 -> 2x3.
    
    C (2x3) = A (2x2) * B (2x3)
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We decompose B into two 2x2 matrices: B_left (cols 0-1) and B_right (col 2).
    C_left = A * B_left (2x2x2, rank 7 via Strassen)
    C_right = A * B_right (2x2x1, rank 2 schoolbook)
    Total: 7 + 2 = 9 terms.
    
    U indexed by (i,j) for A: i,j in {0,1}
    V indexed by (j,k) for B: j in {0,1}, k in {0,1,2}
    W indexed by (k,i) for C: k in {0,1,2}, i in {0,1}
    """
    terms = []
    # 7 Strassen terms for the 2x2x2 part (columns 0 and 1)
    s_terms = _strassen_2x2()
    for term in s_terms:
        u = dict(term['U'])
        v = {}
        for (j, k), coeff in term['V'].items():
            v[(j, k)] = coeff  # k is 0 or 1, same as original
        w = {}
        for (k, i), coeff in term['W'].items():
            w[(k, i)] = coeff  # k is 0 or 1, same as original
        terms.append({'U': u, 'V': v, 'W': w})
    
    # 2 schoolbook terms for the 2x2x1 part (column 2)
    for j in range(2):
        u = {(0, j): 1, (1, j): 1}
        v = {(j, 2): 1}
        w = {(0, 0): 1, (1, 0): 1}
        # Wait, W should be indexed by (k, i) where k=2 for this term
        # C[0,2] += A[0,j]*B[j,2] and C[1,2] += A[1,j]*B[j,2]
        # So W[(2,0)] = 1 and W[(2,1)] = 1
        w = {(2, 0): 1, (2, 1): 1}
        terms.append({'U': u, 'V': v, 'W': w})
    
    return terms

def _build_4x4x5_block():
    """Build 4x4 * 4x5 -> 4x5 using block Strassen decomposition.
    
    A (4x4) split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    B (4x5) split into 2x2 blocks: B = [[B00(2x2), B01(2x3)], [B10(2x2), B11(2x3)]]
    C (4x5) split into 2x2 blocks: C = [[C00(2x2), C01(2x3)], [C10(2x2), C11(2x3)]]
    
    C00 = A00*B00 + A01*B10  (sum of two 2x2x2)
    C01 = A00*B01 + A01*B11  (sum of two 2x2x3)
    C10 = A10*B00 + A11*B10  (sum of two 2x2x2)
    C11 = A10*B01 + A11*B11  (sum of two 2x2x3)
    
    For each 2x2x2: rank 7 (Strassen)
    For each 2x2x3: rank 9
    
    Total: 4*7 + 4*9 = 28 + 36 = 64
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    s2_terms = _strassen_2x2()
    s23_terms = _strassen_2x2x3()
    
    rank = 4 * 7 + 4 * 9  # 64
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 20 x 64
    W = [[0] * rank for _ in range(p * n)]  # 20 x 64
    
    col = 0
    
    # C00 = A00*B00 + A01*B10
    # A00*B00: A block (0,0), B block (0,0), C block (0,0)
    for term in s2_terms:
        for (li, lj), cu in term['U'].items():
            i = li  # 0 or 1
            j = lj  # 0 or 1
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = lj  # 0 or 1
            k = lk  # 0 or 1
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = lk  # 0 or 1
            i = li  # 0 or 1
            W[k*n + i][col] += cw
        col += 1
    
    # A01*B10: A block (0,1), B block (1,0), C block (0,0)
    for term in s2_terms:
        for (li, lj), cu in term['U'].items():
            i = li  # 0 or 1
            j = 2 + lj  # 2 or 3
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = 2 + lj  # 2 or 3
            k = lk  # 0 or 1
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = lk  # 0 or 1
            i = li  # 0 or 1
            W[k*n + i][col] += cw
        col += 1
    
    # C10 = A10*B00 + A11*B10
    # A10*B00: A block (1,0), B block (0,0), C block (1,0)
    for term in s2_terms:
        for (li, lj), cu in term['U'].items():
            i = 2 + li  # 2 or 3
            j = lj  # 0 or 1
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = lj  # 0 or 1
            k = lk  # 0 or 1
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = lk  # 0 or 1
            i = 2 + li  # 2 or 3
            W[k*n + i][col] += cw
        col += 1
    
    # A11*B10: A block (1,1), B block (1,0), C block (1,0)
    for term in s2_terms:
        for (li, lj), cu in term['U'].items():
            i = 2 + li  # 2 or 3
            j = 2 + lj  # 2 or 3
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = 2 + lj  # 2 or 3
            k = lk  # 0 or 1
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = lk  # 0 or 1
            i = 2 + li  # 2 or 3
            W[k*n + i][col] += cw
        col += 1
    
    # C01 = A00*B01 + A01*B11
    # A00*B01: A block (0,0), B block (0,1) which is 2x3, C block (0,1) which is 2x3
    for term in s23_terms:
        for (li, lj), cu in term['U'].items():
            i = li  # 0 or 1
            j = lj  # 0 or 1
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = lj  # 0 or 1
            k = 2 + lk  # 2, 3, or 4
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = 2 + lk  # 2, 3, or 4
            i = li  # 0 or 1
            W[k*n + i][col] += cw
        col += 1
    
    # A01*B11: A block (0,1), B block (1,1) which is 2x3, C block (0,1) which is 2x3
    for term in s23_terms:
        for (li, lj), cu in term['U'].items():
            i = li  # 0 or 1
            j = 2 + lj  # 2 or 3
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = 2 + lj  # 2 or 3
            k = 2 + lk  # 2, 3, or 4
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = 2 + lk  # 2, 3, or 4
            i = li  # 0 or 1
            W[k*n + i][col] += cw
        col += 1
    
    # C11 = A10*B01 + A11*B11
    # A10*B01: A block (1,0), B block (0,1) which is 2x3, C block (1,1) which is 2x3
    for term in s23_terms:
        for (li, lj), cu in term['U'].items():
            i = 2 + li  # 2 or 3
            j = lj  # 0 or 1
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = lj  # 0 or 1
            k = 2 + lk  # 2, 3, or 4
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = 2 + lk  # 2, 3, or 4
            i = 2 + li  # 2 or 3
            W[k*n + i][col] += cw
        col += 1
    
    # A11*B11: A block (1,1), B block (1,1) which is 2x3, C block (1,1) which is 2x3
    for term in s23_terms:
        for (li, lj), cu in term['U'].items():
            i = 2 + li  # 2 or 3
            j = 2 + lj  # 2 or 3
            U[i*m + j][col] += cu
        for (lj, lk), cv in term['V'].items():
            j = 2 + lj  # 2 or 3
            k = 2 + lk  # 2, 3, or 4
            V[j*p + k][col] += cv
        for (lk, li), cw in term['W'].items():
            k = 2 + lk  # 2, 3, or 4
            i = 2 + li  # 2 or 3
            W[k*n + i][col] += cw
        col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary approach: block Strassen decomposition (rank 64)
    try:
        result = _build_4x4x5_block()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 80)
    return _build_schoolbook()
# EVOLVE-BLOCK-END
