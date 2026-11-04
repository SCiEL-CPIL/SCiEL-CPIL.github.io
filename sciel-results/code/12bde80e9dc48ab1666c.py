import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: j=2 (rank 20)
    # Part 2: j=0,1 (rank 35)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=2
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j=0,1 using Strassen on 5 blocks of k
    # Strassen coefficients for 2x2x2
    # M1 = A11(B11+B22)
    # M2 = (A11+A21)B22
    # M3 = A22(B11-B21)
    # M4 = A11(B12-B22)
    # M5 = (A11+A12)B21
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A: i in 0..1, j in 0..1
    # Indices for B: j in 0..1, k in 0..1 (local)
    # Indices for C: i in 0..1, k in 0..1 (local)
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A: (i, j, coeff), B: (j, k, coeff), C: (i, k, coeff)
        # M1
        ([((0,0,1), (1,0,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,0,1))]), # C11+=M1, C21+=M1? No, C22+=M1.
        # Wait, Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 multipliers and their contributions
    ]
    
    # It is easier to hardcode the 7 terms for each block
    # Term 1: M1 = A11 * (B11 + B22)
    # A: (0,0, 1)
    # B: (0,0, 1), (1,1, 1)
    # C: (0,0, 1) [for C11], (1,1, 1) [for C22]
    
    # Term 2: M2 = (A11 + A21) * B22
    # A: (0,0, 1), (1,0, 1)
    # B: (1,1, 1)
    # C: (1,0, 1) [for C21], (1,1, -1) [for C22]
    
    # Term 3: M3 = A22 * (B11 - B21)
    # A: (1,1, 1)
    # B: (0,0, 1), (1,0, -1)
    # C: (0,1, 1) [for C12], (1,1, 1) [for C22]
    
    # Term 4: M4 = A11 * (B12 - B22)
    # A: (0,0, 1)
    # B: (0,1, 1), (1,1, -1)
    # C: (0,0, 1) [for C11], (1,0, 1) [for C21]
    
    # Term 5: M5 = (A11 + A12) * B21
    # A: (0,0, 1), (0,1, 1)
    # B: (1,0, 1)
    # C: (0,0, -1) [for C11], (0,1, 1) [for C12]
    
    # Term 6: M6 = (A21 - A11) * (B11 + B12)
    # A: (1,0, 1), (0,0, -1)
    # B: (0,0, 1), (0,1, 1)
    # C: (1,1, 1) [for C22]
    
    # Term 7: M7 = (A12 - A22) * (B21 + B22)
    # A: (0,1, 1), (1,1, -1)
    # B: (1,0, 1), (1,1, 1)
    # C: (0,0, 1) [for C11]
    
    terms = [
        # (list of (i,j,coeff) for A, list of (j,k,coeff) for B, list of (i,k,coeff) for C)
        ([((0,0,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]), # M1
        ([((0,0,1), (1,0,1))], [((1,1,1))], [((1,0,1), (1,1,-1))]), # M2
        ([((1,1,1))], [((0,0,1), (1,0,-1))], [((0,1,1), (1,1,1))]), # M3
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,0,1), (1,0,1))]), # M4
        ([((0,0,1), (0,1,1))], [((1,0,1))], [((0,0,-1), (0,1,1))]), # M5
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]), # M6
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))]), # M7
    ]
    
    # Map local indices to global
    # Local A: i in 0..1, j in 0..1. Global A: i, j.
    # Local B: j in 0..1, k in 0..1. Global B: j, 2b+k.
    # Local C: i in 0..1, k in 0..1. Global C: i, 2b+k.
    
    for b in range(5):
        for term in terms:
            A_coeffs, B_coeffs, C_coeffs = term
            for (li, lj, ca) in A_coeffs:
                U[li * m + lj][col] = ca
            for (lj, lk, cb) in B_coeffs:
                V[lj * p + (2*b + lk)][col] = cb
            for (li, lk, cc) in C_coeffs:
                W[(2*b + lk) * n + li][col] = cc
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
