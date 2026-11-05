import time
import numpy as np
from fractions import Fraction
import sys

DIMS = (2, 5, 5)
TIME_LIMIT = 135.0

def time_left():
    return TIME_LIMIT - time.time()

def get_schoolbook_decomp(n, m, p):
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}

def check_decomp(n, m, p, U, V, W):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for c in range(r):
                    acc += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if acc != 1:
                    return False
    return True

def construct_strassen_222(U, V, W, i0, j0, k0, rank_offset):
    """
    Add a 2x2x2 Strassen decomposition for the block A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    contributing to C[i0:i0+2, k0:k0+2].
    
    C11 = M1 + M4 - M5 + M6
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M7
    """
    r = rank_offset
    
    # M1 = (A11 + A22)(B11 + B22)
    # Contributes: C11(+1), C22(+1)
    U[i0*m + j0][r] = 1
    U[(i0+1)*m + (j0+1)][r] = 1
    V[j0*p + k0][r] = 1
    V[(j0+1)*p + (k0+1)][r] = 1
    W[k0*n + i0][r] = 1
    W[(k0+1)*n + (i0+1)][r] = 1
    r += 1
    
    # M2 = (A21 + A22)B11
    # Contributes: C21(+1), C22(-1)
    U[i0*m + j0][r] = 0
    U[i0*m + (j0+1)][r] = 0
    U[(i0+1)*m + j0][r] = 1
    U[(i0+1)*m + (j0+1)][r] = 1
    V[j0*p + k0][r] = 1
    V[j0*p + (k0+1)][r] = 0
    V[(j0+1)*p + k0][r] = 0
    V[(j0+1)*p + (k0+1)][r] = 0
    W[k0*n + i0][r] = 0
    W[(k0+1)*n + i0][r] = 0
    W[k0*n + (i0+1)][r] = 1
    W[(k0+1)*n + (i0+1)][r] = -1
    r += 1
    
    # M3 = A11(B12 - B22)
    # Contributes: C12(+1), C22(+1)
    U[i0*m + j0][r] = 1
    U[i0*m + (j0+1)][r] = 0
    U[(i0+1)*m + j0][r] = 0
    U[(i0+1)*m + (j0+1)][r] = 0
    V[j0*p + k0][r] = 0
    V[j0*p + (k0+1)][r] = 1
    V[(j0+1)*p + k0][r] = 0
    V[(j0+1)*p + (k0+1)][r] = -1
    W[k0*n + i0][r] = 0
    W[(k0+1)*n + i0][r] = 1
    W[k0*n + (i0+1)][r] = 0
    W[(k0+1)*n + (i0+1)][r] = 1
    r += 1
    
    # M4 = A22(B21 - B11)
    # Contributes: C11(+1), C21(+1)
    U[i0*m + j0][r] = 0
    U[i0*m + (j0+1)][r] = 0
    U[(i0+1)*m + j0][r] = 0
    U[(i0+1)*m + (j0+1)][r] = 1
    V[j0*p + k0][r] = -1
    V[j0*p + (k0+1)][r] = 0
    V[(j0+1)*p + k0][r] = 1
    V[(j0+1)*p + (k0+1)][r] = 0
    W[k0*n + i0][r] = 1
    W[(k0+1)*n + i0][r] = 0
    W[k0*n + (i0+1)][r] = 1
    W[(k0+1)*n + (i0+1)][r] = 0
    r += 1
    
    # M5 = (A11 + A12)B22
    # Contributes: C11(-1), C12(+1)
    U[i0*m + j0][r] = 1
    U[i0*m + (j0+1)][r] = 1
    U[(i0+1)*m + j0][r] = 0
    U[(i0+1)*m + (j0+1)][r] = 0
    V[j0*p + k0][r] = 0
    V[j0*p + (k0+1)][r] = 0
    V[(j0+1)*p + k0][r] = 0
    V[(j0+1)*p + (k0+1)][r] = 1
    W[k0*n + i0][r] = -1
    W[(k0+1)*n + i0][r] = 1
    W[k0*n + (i0+1)][r] = 0
    W[(k0+1)*n + (i0+1)][r] = 0
    r += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    # Contributes: C11(+1)
    U[i0*m + j0][r] = -1
    U[i0*m + (j0+1)][r] = 0
    U[(i0+1)*m + j0][r] = 1
    U[(i0+1)*m + (j0+1)][r] = 0
    V[j0*p + k0][r] = 1
    V[j0*p + (k0+1)][r] = 1
    V[(j0+1)*p + k0][r] = 0
    V[(j0+1)*p + (k0+1)][r] = 0
    W[k0*n + i0][r] = 1
    W[(k0+1)*n + i0][r] = 0
    W[k0*n + (i0+1)][r] = 0
    W[(k0+1)*n + (i0+1)][r] = 0
    r += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    # Contributes: C22(+1)
    U[i0*m + j0][r] = 0
    U[i0*m + (j0+1)][r] = 1
    U[(i0+1)*m + j0][r] = 0
    U[(i0+1)*m + (j0+1)][r] = -1
    V[j0*p + k0][r] = 0
    V[j0*p + (k0+1)][r] = 0
    V[(j0+1)*p + k0][r] = 1
    V[(j0+1)*p + (k0+1)][r] = 1
    W[k0*n + i0][r] = 0
    W[(k0+1)*n + i0][r] = 0
    W[k0*n + (i0+1)][r] = 0
    W[(k0+1)*n + (i0+1)][r] = 1
    r += 1
    
    return r

def add_2x1x2(U, V, W, i0, j0, k0, rank_offset):
    """
    Add a 2x1x2 decomposition for A[i0:i0+2, j0] * B[j0, k0:k0+2]
    contributing to C[i0:i0+2, k0:k0+2].
    """
    r = rank_offset
    # C1 = A1 B1, C2 = A2 B2
    # Term 1: A1 B1
    U[i0*m + j0][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    r += 1
    # Term 2: A2 B2
    U[(i0+1)*m + j0][r] = 1
    V[j0*p + (k0+1)][r] = 1
    W[(k0+1)*n + (i0+1)][r] = 1
    r += 1
    return r

def add_2x2x1(U, V, W, i0, j0, k0, rank_offset):
    """
    Add a 2x2x1 decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0]
    contributing to C[i0:i0+2, k0].
    """
    r = rank_offset
    # C1 = A1 B1 + A2 B2
    # Term 1: A1 B1
    U[i0*m + j0][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    r += 1
    # Term 2: A2 B2
    U[i0*m + (j0+1)][r] = 1
    V[(j0+1)*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    r += 1
    # Term 3: A1 B2
    U[i0*m + (j0+1)][r] = 1
    V[(j0+1)*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    # Wait, this is not right.
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    # We need 4 terms? Or can we do it in 2?
    # No, 2x2x1 rank is 2.
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    # This is not separable into 2 rank-1 terms easily?
    # Actually, rank of 2x2x1 is 2.
    # Let's use 4 terms for simplicity? No, we want low rank.
    # But 2x2x1 is just two dot products.
    # C1 = sum A1j Bj1
    # C2 = sum A2j Bj1
    # We can use 2 terms if we combine?
    # No, A and B are different.
    # Let's just use 4 terms: A11B11, A12B21, A21B11, A22B21.
    # This adds 4 to the rank.
    # My previous estimate was 2. That was wrong.
    # 2x2x1 rank is 2?
    # T(i,j,k) = 1 if i=1,j=1,k=1 or i=1,j=2,k=1 or i=2,j=1,k=1 or i=2,j=2,k=1.
    # This tensor is the sum of 4 rank-1 terms.
    # Can it be done in 2?
    # No, the slice for k=1 is a 2x2 matrix of all 1s, which has rank 2.
    # So the tensor rank is at least 2.
    # But the support is 4 points.
    # Actually, the rank of the 2x2x1 tensor with all 1s is 2.
    # Because it is the outer product of [1,1] and [[1,1],[1,1]]?
    # No.
    # Let's just use 4 terms to be safe.
    
    # Reset r to rank_offset
    r = rank_offset
    # Term 1: A11 B11
    U[i0*m + j0][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    r += 1
    # Term 2: A12 B21
    U[i0*m + (j0+1)][r] = 1
    V[(j0+1)*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    r += 1
    # Term 3: A21 B11
    U[(i0+1)*m + j0][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + (i0+1)][r] = 1
    r += 1
    # Term 4: A22 B21
    U[(i0+1)*m + (j0+1)][r] = 1
    V[(j0+1)*p + k0][r] = 1
    W[k0*n + (i0+1)][r] = 1
    r += 1
    return r

def add_2x1x1(U, V, W, i0, j0, k0, rank_offset):
    """
    Add a 2x1x1 decomposition for A[i0:i0+2, j0] * B[j0, k0]
    contributing to C[i0:i0+2, k0].
    """
    r = rank_offset
    # Term 1: A1 B
    U[i0*m + j0][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + i0][r] = 1
    r += 1
    # Term 2: A2 B
    U[(i0+1)*m + j0][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + (i0+1)][r] = 1
    r += 1
    return r

def build():
    n, m, p = DIMS
    # Split 5 into 2+2+1
    # A: 2x5 -> [2x2, 2x2, 2x1]
    # B: 5x5 -> 3x3 blocks of 2x2, 2x1, 1x2, 1x1
    # C: 2x5 -> [2x2, 2x2, 2x1]
    
    # Indices for blocks
    # j: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    # k: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    
    # C1 (cols 0,1):
    # A1 B11 (2x2x2)
    # A2 B21 (2x2x2)
    # A3 B31 (2x1x2)
    
    # C2 (cols 2,3):
    # A1 B12 (2x2x2)
    # A2 B22 (2x2x2)
    # A3 B32 (2x1x2)
    
    # C3 (col 4):
    # A1 B13 (2x2x1)
    # A2 B23 (2x2x1)
    # A3 B33 (2x1x1)
    
    # Total rank:
    # C1: 7 + 7 + 2 = 16
    # C2: 7 + 7 + 2 = 16
    # C3: 4 + 4 + 2 = 10
    # Total: 42
    
    # Initialize U, V, W with size 42
    r = 42
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    offset = 0
    
    # C1
    offset = construct_strassen_222(U, V, W, 0, 0, 0, offset) # A1 B11
    offset = construct_strassen_222(U, V, W, 0, 2, 0, offset) # A2 B21
    offset = add_2x1x2(U, V, W, 0, 4, 0, offset)             # A3 B31
    
    # C2
    offset = construct_strassen_222(U, V, W, 0, 0, 2, offset) # A1 B12
    offset = construct_strassen_222(U, V, W, 0, 2, 2, offset) # A2 B22
    offset = add_2x1x2(U, V, W, 0, 4, 2, offset)             # A3 B32
    
    # C3
    offset = add_2x2x1(U, V, W, 0, 0, 4, offset)             # A1 B13
    offset = add_2x2x1(U, V, W, 0, 2, 4, offset)             # A2 B23
    offset = add_2x1x1(U, V, W, 0, 4, 4, offset)             # A3 B33
    
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        res = build()
        n, m, p = DIMS
        if check_decomp(n, m, p, res['U'], res['V'], res['W']):
            return res
        else:
            return get_schoolbook_decomp(n, m, p)
    except Exception:
        n, m, p = DIMS
        return get_schoolbook_decomp(n, m, p)
