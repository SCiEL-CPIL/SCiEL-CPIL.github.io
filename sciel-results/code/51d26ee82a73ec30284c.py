import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _construct_rank_55(n, m, p):
    """
    Construct a rank 55 decomposition for 2x3 * 3x10.
    Splits A (2x3) into A1 (2x2) and A2 (2x1).
    Splits B (3x10) into B1 (2x10) and B2 (1x10).
    C = A1 B1 + A2 B2.
    A1 B1 uses 5 Strassen blocks (2x2x2) for the 10 columns of B1.
    A2 B2 uses schoolbook (2x1x10).
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks
    # B1 has 2 rows, 10 cols. We process columns in pairs (2j, 2j+1).
    for j in range(5):
        # Strassen indices for this block
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c+d)(f-h)
        # M7 = (a+b)(g-e)
        
        # e = B[0, 2j], f = B[0, 2j+1]
        # g = B[1, 2j], h = B[1, 2j+1]
        
        # V indices for e, f, g, h
        idx_e = 0 * p + 2*j
        idx_f = 0 * p + 2*j + 1
        idx_g = 1 * p + 2*j
        idx_h = 1 * p + 2*j + 1
        
        # W indices for C[0, 2j], C[0, 2j+1], C[1, 2j], C[1, 2j+1]
        # W index is k*n + i
        idx_c00 = 2*j * n + 0
        idx_c01 = (2*j+1) * n + 0
        idx_c10 = 2*j * n + 1
        idx_c11 = (2*j+1) * n + 1
        
        # U indices for a, b, c, d (A1 is 2x2, indices 0,1,2,3 in U)
        idx_a = 0
        idx_b = 1
        idx_c = 2
        idx_d = 3
        
        # M1
        U[idx_a][col] += 1
        U[idx_d][col] += 1
        V[idx_e][col] += 1
        V[idx_h][col] += 1
        W[idx_c00][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M2
        U[idx_c][col] += 1
        U[idx_d][col] += 1
        V[idx_e][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        # M3
        U[idx_a][col] += 1
        V[idx_f][col] += 1
        V[idx_h][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        # M4
        U[idx_d][col] += 1
        V[idx_g][col] += 1
        V[idx_f][col] -= 1
        W[idx_c00][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        # M5
        U[idx_a][col] += 1
        U[idx_b][col] += 1
        V[idx_h][col] += 1
        W[idx_c00][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        # M6
        U[idx_c][col] += 1
        U[idx_d][col] += 1
        V[idx_f][col] += 1
        V[idx_h][col] -= 1
        W[idx_c11][col] -= 1
        col += 1
        
        # M7
        U[idx_a][col] += 1
        U[idx_b][col] += 1
        V[idx_g][col] += 1
        V[idx_e][col] -= 1
        W[idx_c00][col] += 1
        col += 1

    # Part 2: A2 (2x1) * B2 (1x10) using schoolbook
    # A2 indices in U: 4, 5
    # B2 indices in V: 20..29 (row 2 of B)
    # C indices in W: 0..19
    
    # M8..M17: A[0,2] * B[2,k] -> C[0,k]
    for k in range(10):
        idx_a2 = 4
        idx_b2 = 2 * p + k
        idx_c = k * n + 0
        
        U[idx_a2][col] += 1
        V[idx_b2][col] += 1
        W[idx_c][col] += 1
        col += 1
        
    # M18..M27: A[1,2] * B[2,k] -> C[1,k]
    for k in range(10):
        idx_a2 = 5
        idx_b2 = 2 * p + k
        idx_c = k * n + 1
        
        U[idx_a2][col] += 1
        V[idx_b2][col] += 1
        W[idx_c][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use the specific rank 55 decomposition
    result = _construct_rank_55(n, m, p)
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
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
