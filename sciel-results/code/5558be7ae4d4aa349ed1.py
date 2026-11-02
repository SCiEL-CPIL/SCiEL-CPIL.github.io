import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p
    # We will use Strassen for two 2x2x2 blocks and schoolbook for the rest.
    # Block 1: j in {0, 1}, k in {0, 1}
    # Block 2: j in {2, 3}, k in {2, 3}
    
    # Total rank
    # 2 blocks * 7 (Strassen) + remaining (50 - 8 - 8) * 1 (Schoolbook)
    # = 14 + 34 = 48
    rank = 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Helper to add a Strassen block
    # j0, k0 are the starting indices for the 2x2 block
    def add_strassen_block(j0, k0):
        nonlocal col
        # Strassen terms for 2x2x2
        # A indices: i in {0,1}, j in {j0, j0+1}
        # B indices: j in {j0, j0+1}, k in {k0, k0+1}
        # C indices: i in {0,1}, k in {k0, k0+1}
        
        # M1 = (A0j0 + A1j0) * (Bj0k0 + Bj0+1k0+1)
        # C00 += M1, C11 += M1
        U[0*m + j0][col] = 1
        U[1*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + 0][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # M2 = (A1j0 + A1j0+1) * Bj0k0  -> (c+d)*e
        # C10 += M2
        U[1*m + j0][col] = 1
        U[1*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M3 = A0j0 * (Bj0k0+1 - Bj0k0) -> a*(f-e)
        # C01 += M3
        U[0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[j0*p + k0][col] = -1
        W[(k0+1)*n + 0][col] = 1
        col += 1
        
        # M4 = A1j0+1 * (Bj0+1k0 - Bj0k0) -> d*(g-e)
        # C00 += M4, C10 += M4
        U[1*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M5 = (A0j0 + A0j0+1) * Bj0+1k0+1 -> (a+b)*h
        # C00 -= M5, C01 += M5
        U[0*m + j0][col] = 1
        U[0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + 0][col] = -1
        W[(k0+1)*n + 0][col] = 1
        col += 1
        
        # M6 = (A1j0 - A0j0) * (Bj0k0 + Bj0k0+1) -> (c-a)*(e+f)
        # C11 += M6
        U[1*m + j0][col] = 1
        U[0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # M7 = (A0j0+1 - A1j0+1) * (Bj0+1k0 + Bj0+1k0+1) -> (b-d)*(g+h)
        # C00 += M7, C11 += M7
        U[0*m + (j0+1)][col] = 1
        U[1*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + 0][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1

    # Add Strassen blocks
    add_strassen_block(0, 0)
    add_strassen_block(2, 2)
    
    # Add schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (j,k) pair is in a Strassen block
                in_block1 = (j in [0, 1]) and (k in [0, 1])
                in_block2 = (j in [2, 3]) and (k in [2, 3])
                if not (in_block1 or in_block2):
                    add_schoolbook(i, j, k)
                    
    return {"U": U, "V": V, "W": W}
