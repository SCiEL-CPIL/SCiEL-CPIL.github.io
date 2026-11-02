import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 26 + 3 * 3 * 4  # 26 + 36 = 62
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 3x3x3 (k=0,1,2) using Strassen-based rank-26 decomposition
    # A indices: A[i,j] -> U[i*m+j]
    # B indices: B[j,k] -> V[j*p+k]
    # C indices: C[k,i] -> W[k*n+i]  (trace convention: W[k*n+i])
    
    # Strassen for A[0:2,0:2] * B[0:2,0:2] producing C[0:2,0:2]
    # P1 = (A00+A11)*(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # P2 = (A10+A11)*B00
    U[3][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[6][col] = 1
    col += 1
    
    # P3 = A00*(B10-B11)
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # P4 = A11*(B01-B00)
    U[3][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # P5 = (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # P6 = (A10-A00)*(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[4][col] = 1
    W[3][col] = 1; W[6][col] = 1
    col += 1
    
    # P7 = (A01-A11)*(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # Now add cross terms:
    # C[0,0] += A[0,2]*B[2,0]
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # C[0,1] += A[0,2]*B[2,1]
    U[2][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    
    # C[1,0] += A[2,2]*B[2,0]
    U[8][col] = 1
    V[6][col] = 1
    W[6][col] = 1
    col += 1
    
    # C[1,1] += A[2,2]*B[2,1]
    U[8][col] = 1
    V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # C[0,2] += A[0,2]*B[2,2]
    U[2][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # C[1,2] += A[1,2]*B[2,2]
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    # C[2,0] += A[2,0]*B[0,0]
    U[6][col] = 1
    V[0][col] = 1
    W[6][col] = 1
    col += 1
    
    # C[2,0] += A[2,1]*B[1,0]
    U[7][col] = 1
    V[3][col] = 1
    W[6][col] = 1
    col += 1
    
    # C[2,1] += A[2,0]*B[0,1]
    U[6][col] = 1
    V[4][col] = 1
    W[7][col] = 1
    col += 1
    
    # C[2,1] += A[2,1]*B[1,1]
    U[7][col] = 1
    V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # C[2,2] += A[2,0]*B[0,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # C[2,2] += A[2,1]*B[1,2]
    U[7][col] = 1
    V[5][col] = 1
    W[8][col] = 1
    col += 1
    
    # C[2,2] += A[2,2]*B[2,2]
    U[8][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    assert col == 26, f"Part 1: expected 26, got {col}"
    
    # Part 2: Schoolbook for k=3,4,5,6
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"Total: expected {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
