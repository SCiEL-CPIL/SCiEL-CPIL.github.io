"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W give a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                if is_valid:
                    if abs(s - 1.0) > tol:
                        return False
                else:
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _strassen_2x2(A_indices, B_indices, C_indices, U, V, W, col):
    """
    Add Strassen terms for 2x2 * 2x2 multiplication.
    A_indices: list of (i,j) for A[0,0], A[0,1], A[1,0], A[1,1]
    B_indices: list of (j,k) for B[0,0], B[0,1], B[1,0], B[1,1]
    C_indices: list of (i,k) for C[0,0], C[0,1], C[1,0], C[1,1]
    Returns new col.
    """
    n, m, p = 3, 4, 4
    # A_indices[0] = (i0,j0), A_indices[1] = (i0,j1), A_indices[2] = (i1,j0), A_indices[3] = (i1,j1)
    # B_indices[0] = (j0,k0), B_indices[1] = (j0,k1), B_indices[2] = (j1,k0), B_indices[3] = (j1,k1)
    # C_indices[0] = (i0,k0), C_indices[1] = (i0,k1), C_indices[2] = (i1,k0), C_indices[3] = (i1,k1)
    
    a00, a01, a10, a11 = A_indices
    b00, b01, b10, b11 = B_indices
    c00, c01, c10, c11 = C_indices
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(i, k):
        return k * n + i
    
    # M1 = (A00+A11)*(B00+B11)
    U[a_idx(*a00)][col] += 1
    U[a_idx(*a11)][col] += 1
    V[b_idx(*b00)][col] += 1
    V[b_idx(*b11)][col] += 1
    W[c_idx(*c00)][col] += 1
    W[c_idx(*c11)][col] += 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[a_idx(*a10)][col] += 1
    U[a_idx(*a11)][col] += 1
    V[b_idx(*b00)][col] += 1
    W[c_idx(*c10)][col] += 1
    W[c_idx(*c11)][col] += -1
    col += 1
    
    # M3 = A00*(B10-B11)
    U[a_idx(*a00)][col] += 1
    V[b_idx(*b10)][col] += 1
    V[b_idx(*b11)][col] += -1
    W[c_idx(*c01)][col] += 1
    W[c_idx(*c11)][col] += 1
    col += 1
    
    # M4 = A11*(B01-B00)
    U[a_idx(*a11)][col] += 1
    V[b_idx(*b01)][col] += 1
    V[b_idx(*b00)][col] += -1
    W[c_idx(*c00)][col] += 1
    W[c_idx(*c10)][col] += 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[a_idx(*a00)][col] += 1
    U[a_idx(*a01)][col] += 1
    V[b_idx(*b11)][col] += 1
    W[c_idx(*c00)][col] += -1
    W[c_idx(*c01)][col] += 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[a_idx(*a10)][col] += 1
    U[a_idx(*a00)][col] += -1
    V[b_idx(*b00)][col] += 1
    V[b_idx(*b01)][col] += 1
    W[c_idx(*c11)][col] += 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[a_idx(*a01)][col] += 1
    U[a_idx(*a11)][col] += -1
    V[b_idx(*b10)][col] += 1
    V[b_idx(*b11)][col] += 1
    W[c_idx(*c00)][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try the Strassen-based decomposition with rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Schoolbook for i=2 (bottom row): 16 terms
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Strassen for A_top_left * B11
    # A_top_left: A[0,0], A[0,1], A[1,0], A[1,1]
    # B11: B[0,0], B[0,1], B[1,0], B[1,1]
    # C top-left: C[0,0], C[0,1], C[1,0], C[1,1]
    col = _strassen_2x2(
        [(0,0), (0,1), (1,0), (1,1)],  # A_top_left
        [(0,0), (0,1), (1,0), (1,1)],  # B11
        [(0,0), (0,1), (1,0), (1,1)],  # C top-left
        U, V, W, col
    )
    
    # Strassen for A_top_right * B21
    # A_top_right: A[0,2], A[0,3], A[1,2], A[1,3]
    # B21: B[2,0], B[2,1], B[3,0], B[3,1]
    # C top-left: C[0,0], C[0,1], C[1,0], C[1,1]
    col = _strassen_2x2(
        [(0,2), (0,3), (1,2), (1,3)],  # A_top_right
        [(2,0), (2,1), (3,0), (3,1)],  # B21
        [(0,0), (0,1), (1,0), (1,1)],  # C top-left
        U, V, W, col
    )
    
    # Strassen for A_top_left * B12
    # A_top_left: A[0,0], A[0,1], A[1,0], A[1,1]
    # B12: B[0,2], B[0,3], B[1,2], B[1,3]
    # C top-right: C[0,2], C[0,3], C[1,2], C[1,3]
    col = _strassen_2x2(
        [(0,0), (0,1), (1,0), (1,1)],  # A_top_left
        [(0,2), (0,3), (1,2), (1,3)],  # B12
        [(0,2), (0,3), (1,2), (1,3)],  # C top-right
        U, V, W, col
    )
    
    # Strassen for A_top_right * B22
    # A_top_right: A[0,2], A[0,3], A[1,2], A[1,3]
    # B22: B[2,2], B[2,3], B[3,2], B[3,3]
    # C top-right: C[0,2], C[0,3], C[1,2], C[1,3]
    col = _strassen_2x2(
        [(0,2), (0,3), (1,2), (1,3)],  # A_top_right
        [(2,2), (2,3), (3,2), (3,3)],  # B22
        [(0,2), (0,3), (1,2), (1,3)],  # C top-right
        U, V, W, col
    )
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
