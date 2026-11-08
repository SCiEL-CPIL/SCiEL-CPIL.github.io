import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _get_laderman_3x3():
    # Returns U, V, W for 3x3x3 multiplication using Laderman's algorithm (rank 23)
    # U: 9x23, V: 9x23, W: 9x23
    # Indices: A is 3x3 (0..8), B is 3x3 (0..8), C is 3x3 (0..8)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Helper to set a term
    # term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict {(i,j): val}
    # b_coeffs: dict {(j,k): val}
    # c_coeffs: dict {(k,i): val}
    
    terms = []
    
    # M1
    terms.append(( {(0,0): 1}, {(0,0): 1}, {(0,0): 1} ))
    # M2
    terms.append(( {(0,0): 1}, {(1,1): 1}, {(1,0): 1} ))
    # M3
    terms.append(( {(0,1): 1}, {(1,0): 1}, {(0,1): 1} ))
    # M4
    terms.append(( {(0,2): 1}, {(2,0): 1}, {(2,0): 1} ))
    # M5
    terms.append(( {(1,0): 1}, {(0,1): 1}, {(0,2): 1} ))
    # M6
    terms.append(( {(1,1): 1}, {(1,1): 1}, {(1,1): 1} ))
    # M7
    terms.append(( {(1,2): 1}, {(2,1): 1}, {(1,2): 1} ))
    # M8
    terms.append(( {(2,0): 1}, {(0,2): 1}, {(2,2): 1} ))
    # M9
    terms.append(( {(2,1): 1}, {(1,2): 1}, {(2,1): 1} ))
    # M10
    terms.append(( {(2,2): 1}, {(2,2): 1}, {(2,2): 1} ))
    
    # M11: (A00+A11)(B00+B11)
    terms.append(( {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): -1, (1,1): -1, (0,1): 1, (1,0): 1} ))
    # M12: (A01+A12)(B10+B20)
    terms.append(( {(0,1): 1, (1,2): 1}, {(1,0): 1, (2,0): 1}, {(0,2): -1, (1,0): -1, (0,0): 1, (1,2): 1} ))
    # M13: (A10+A22)(B01+B21)
    terms.append(( {(1,0): 1, (2,2): 1}, {(0,1): 1, (2,1): 1}, {(1,1): -1, (2,2): -1, (1,0): 1, (2,1): 1} ))
    # M14: (A20+A02)(B02+B12)
    terms.append(( {(2,0): 1, (0,2): 1}, {(0,2): 1, (1,2): 1}, {(2,0): -1, (1,2): -1, (2,2): 1, (0,0): 1} ))
    # M15: (A00+A22)(B00+B22)
    terms.append(( {(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}, {(0,0): -1, (2,2): -1, (0,2): 1, (2,0): 1} ))
    # M16: (A01+A22)(B10+B02)
    terms.append(( {(0,1): 1, (2,2): 1}, {(1,0): 1, (0,2): 1}, {(0,1): -1, (2,2): -1, (0,0): 1, (2,1): 1} ))
    # M17: (A10+A02)(B11+B20)
    terms.append(( {(1,0): 1, (0,2): 1}, {(1,1): 1, (2,0): 1}, {(1,0): -1, (0,2): -1, (1,2): 1, (0,1): 1} ))
    # M18: (A11+A20)(B12+B01)
    terms.append(( {(1,1): 1, (2,0): 1}, {(1,2): 1, (0,1): 1}, {(1,1): -1, (2,0): -1, (1,0): 1, (2,2): 1} ))
    # M19: (A12+A21)(B21+B10)
    terms.append(( {(1,2): 1, (2,1): 1}, {(2,1): 1, (1,0): 1}, {(1,2): -1, (2,1): -1, (1,1): 1, (2,0): 1} ))
    # M20: (A20+A01)(B02+B11)
    terms.append(( {(2,0): 1, (0,1): 1}, {(0,2): 1, (1,1): 1}, {(2,0): -1, (0,1): -1, (2,2): 1, (0,0): 1} ))
    # M21: (A21+A00)(B12+B20)
    terms.append(( {(2,1): 1, (0,0): 1}, {(1,2): 1, (2,0): 1}, {(2,1): -1, (0,0): -1, (2,0): 1, (0,2): 1} ))
    # M22: (A22+A10)(B21+B01)
    terms.append(( {(2,2): 1, (1,0): 1}, {(2,1): 1, (0,1): 1}, {(2,2): -1, (1,0): -1, (2,1): 1, (1,2): 1} ))
    # M23: (A00+A11+A22)(B00+B11+B22)
    terms.append(( {(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): -1, (1,1): -1, (2,2): -1, (0,1): 1, (1,0): 1, (0,2): 1, (2,0): 1, (1,2): 1, (2,1): 1} ))
    
    for r, (a, b, c) in enumerate(terms):
        for (i, j), val in a.items():
            U[i*m+j][r] = val
        for (j, k), val in b.items():
            V[j*p+k][r] = val
        for (k, i), val in c.items():
            W[k*n+i][r] = val
            
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55: 2 * 23 (Laderman for 3x3x3) + 9 (Schoolbook for 3x3x2)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U3, V3, W3 = _get_laderman_3x3()
    
    # Block 1: Columns 0, 1, 2 of B
    # U is same
    # V maps 3x3 to 3x8 (cols 0-2)
    # W maps 3x3 to 3x8 (cols 0-2)
    for r in range(23):
        for i in range(n):
            for j in range(m):
                U[i*m+j][r] = U3[i*m+j][r]
        for j in range(m):
            for k in range(3):
                V[j*p+k][r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(n):
                W[k*n+i][r] = W3[k*3+i][r]
                
    # Block 2: Columns 3, 4, 5 of B
    for r in range(23):
        for i in range(n):
            for j in range(m):
                U[i*m+j][r+23] = U3[i*m+j][r]
        for j in range(m):
            for k in range(3):
                V[j*p+(3+k)][r+23] = V3[j*3+k][r]
        for k in range(3):
            for i in range(n):
                W[(3+k)*n+i][r+23] = W3[k*3+i][r]
                
    # Block 3: Columns 6, 7 of B (Schoolbook 3x3x2)
    col = 46
    for i in range(n):
        for j in range(m):
            for k in range(6, 8):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
