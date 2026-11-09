import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U shape (n*m, rank) = (6, rank)
    # V shape (m*p, rank) = (24, rank)
    # W shape (p*n, rank) = (16, rank)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # A indices: 0,1,3,4
    # B indices: 0..7, 8..15
    # C indices: 0..15
    
    for j in range(4):
        # Block j corresponds to B columns 2j, 2j+1
        # B[0, 2j] is V[2j]
        # B[0, 2j+1] is V[2j+1]
        # B[1, 2j] is V[8+2j]
        # B[1, 2j+1] is V[8+2j+1]
        
        b00_idx = 2 * j
        b01_idx = 2 * j + 1
        b10_idx = 8 + 2 * j
        b11_idx = 8 + 2 * j + 1
        
        # C indices
        c00_idx = 4 * j      # C[0, 2j]
        c01_idx = 4 * j + 2  # C[0, 2j+1]
        c10_idx = 4 * j + 1  # C[1, 2j]
        c11_idx = 4 * j + 3  # C[1, 2j+1]
        
        # Strassen products
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][r] += 1
        U[4][r] += 1
        V[b00_idx][r] += 1
        V[b11_idx][r] += 1
        W[c00_idx][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        # M2 = (A10 + A11) * B00
        U[3][r] += 1
        U[4][r] += 1
        V[b00_idx][r] += 1
        W[c10_idx][r] += 1
        r += 1
        
        # M3 = A00 * (B01 - B11)
        U[0][r] += 1
        V[b01_idx][r] += 1
        V[b11_idx][r] -= 1
        W[c01_idx][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        # M4 = A11 * (B10 - B00)
        U[4][r] += 1
        V[b10_idx][r] += 1
        V[b00_idx][r] -= 1
        W[c00_idx][r] += 1
        W[c10_idx][r] += 1
        r += 1
        
        # M5 = (A00 + A01) * B11
        U[0][r] += 1
        U[1][r] += 1
        V[b11_idx][r] += 1
        W[c00_idx][r] -= 1
        W[c01_idx][r] += 1
        r += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[3][r] += 1
        U[0][r] -= 1
        V[b00_idx][r] += 1
        V[b01_idx][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][r] += 1
        U[4][r] -= 1
        V[b10_idx][r] += 1
        V[b11_idx][r] += 1
        W[c00_idx][r] += 1
        W[c11_idx][r] -= 1
        r += 1

    # Part 2: A[0:2, 2] * B[2, 0:8]
    # A indices: 2, 5
    # B indices: 16..23
    # C indices: 0..15
    
    for k in range(8):
        # C[0, k] += A[0, 2] * B[2, k]
        # C[1, k] += A[1, 2] * B[2, k]
        
        b_idx = 16 + k
        c0_idx = 2 * k
        c1_idx = 2 * k + 1
        
        U[2][r] += 1
        V[b_idx][r] += 1
        W[c0_idx][r] += 1
        r += 1
        
        U[5][r] += 1
        V[b_idx][r] += 1
        W[c1_idx][r] += 1
        r += 1

    return {"U": U, "V": V, "W": W}
