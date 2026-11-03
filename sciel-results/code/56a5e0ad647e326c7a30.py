import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 28 + 28 + 4 = 60
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for 2x2 multiplication
    # A is 2x2 at rows i0, i1 and cols j0, j1
    # B is 2x2 at rows j0, j1 and cols k0, k1
    # C is 2x2 at rows i0, i1 and cols k0, k1
    # Strassen coefficients:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # Map local indices (0,1) to global
        # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C indices: (i0, k0), (i0, k1), (i1, k0), (i1, k1)
        
        # P1
        # U: A11, A22 -> 1
        U[i0*m + j0][col] = 1
        U[i1*m + j1][col] = 1
        # V: B11, B22 -> 1
        V[j0*p + k0][col] = 1
        V[j1*p + k1][col] = 1
        # W: C11, C22 -> 1; C12, C21 -> 0
        W[k0*n + i0][col] = 1
        W[k1*n + i1][col] = 1
        col += 1
        
        # P2
        # U: A21, A22 -> 1
        U[i1*m + j0][col] = 1
        U[i1*m + j1][col] = 1
        # V: B11 -> 1
        V[j0*p + k0][col] = 1
        # W: C21 -> 1
        W[k0*n + i1][col] = 1
        col += 1
        
        # P3
        # U: A11 -> 1
        U[i0*m + j0][col] = 1
        # V: B12 -> 1, B22 -> -1
        V[j0*p + k1][col] = 1
        V[j1*p + k1][col] = -1
        # W: C12 -> 1
        W[k1*n + i0][col] = 1
        col += 1
        
        # P4
        # U: A22 -> 1
        U[i1*m + j1][col] = 1
        # V: B21 -> 1, B11 -> -1
        V[j1*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        # W: C11 -> 1
        W[k0*n + i0][col] = 1
        col += 1
        
        # P5
        # U: A11, A12 -> 1
        U[i0*m + j0][col] = 1
        U[i0*m + j1][col] = 1
        # V: B22 -> 1
        V[j1*p + k1][col] = 1
        # W: C11 -> -1 (from C11 = P1+P4-P5+P7)
        W[k0*n + i0][col] = -1
        col += 1
        
        # P6
        # U: A21, A11 -> 1, -1
        U[i1*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        # V: B11, B12 -> 1
        V[j0*p + k0][col] = 1
        V[j0*p + k1][col] = 1
        # W: C22 -> 1
        W[k1*n + i1][col] = 1
        col += 1
        
        # P7
        # U: A12, A22 -> 1, -1
        U[i0*m + j1][col] = 1
        U[i1*m + j1][col] = -1
        # V: B21, B22 -> 1
        V[j1*p + k0][col] = 1
        V[j1*p + k1][col] = 1
        # W: C11 -> 1
        W[k0*n + i0][col] = 1
        col += 1

    # Sub-problem 1.1: i in {0,1}, j in {0..3}, k in {0..3}
    # C_left = A11 B11 + A12 B21
    # A11: i 0,1; j 0,1
    # B11: j 0,1; k 0,1
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # A12: i 0,1; j 2,3
    # B21: j 2,3; k 0,1
    add_strassen(0, 1, 2, 3, 0, 1)
    
    # C_right = A11 B12 + A12 B22
    # A11: i 0,1; j 0,1
    # B12: j 0,1; k 2,3
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # A12: i 0,1; j 2,3
    # B22: j 2,3; k 2,3
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Sub-problem 1.2: i in {0,1}, j in {0..3}, k in {4..7}
    # Same structure, k offset by 4
    add_strassen(0, 1, 0, 1, 4, 5)
    add_strassen(0, 1, 2, 3, 4, 5)
    add_strassen(0, 1, 0, 1, 6, 7)
    add_strassen(0, 1, 2, 3, 6, 7)
    
    # Sub-problem 2: i=2, j in {0..3}, k in {0..7}
    # Rank 4
    # U[2*m+j, col] = 1
    # V[j*p+k, col] = delta(j, r)
    # W[k*n+2, col] = 1
    for r in range(4):
        # U
        U[2*m + r][col] = 1
        # V
        for j in range(4):
            if j == r:
                for k in range(8):
                    V[j*p + k][col] = 1
        # W
        for k in range(8):
            W[k*n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
