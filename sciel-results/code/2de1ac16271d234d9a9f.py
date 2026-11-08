import numpy as np

def construct():
    n, m, p = 3, 3, 3
    rank = 26
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0
        col += 1

    # Indices mapping:
    # U[i*m + k] for A[i,k]
    # V[k*p + j] for B[k,j]
    # W[j*n + i] for C[i,j] (Note: Task says W[k*n+i, r] where k is col, i is row? 
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,j] is index j*n + i in column-major? Or i*p + j?
    # Let's check the current program's comment:
    # "W index for C_{ik} is k*3 + i."
    # So W index = col * n + row.
    
    # 1. Strassen for top-left 2x2 (A[0:2, 0:2] * B[0:2, 0:2])
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices (col*3+row):
    # C00 (0,0) -> 0
    # C01 (1,0) -> 3
    # C10 (0,1) -> 1
    # C11 (1,1) -> 4
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00 -> C10, C11(-)
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11) -> C01, C11
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00) -> C00, C10
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11 -> C00(-), C01
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11) -> C00
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Border terms
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A02 (idx 2), A12 (idx 5)
    # B20 (idx 6), B21 (idx 7)
    # C00 (0), C01 (3), C10 (1), C11 (4)
    
    add_term(2, 6, 0, 1) # A02 B20 -> C00
    add_term(2, 7, 3, 1) # A02 B21 -> C01
    add_term(5, 6, 1, 1) # A12 B20 -> C10
    add_term(5, 7, 4, 1) # A12 B21 -> C11
    
    # A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A00(0), A01(1), A10(3), A11(4)
    # B02(2), B12(5)
    # C02 (2*3+0=6), C12 (2*3+1=7)
    
    add_term(0, 2, 6, 1) # A00 B02 -> C02
    add_term(1, 5, 6, 1) # A01 B12 -> C02
    add_term(3, 2, 7, 1) # A10 B02 -> C12
    add_term(4, 5, 7, 1) # A11 B12 -> C12
    
    # A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    # A02(2), A12(5)
    # B22(8)
    # C02(6), C12(7)
    
    add_term(2, 8, 6, 1) # A02 B22 -> C02
    add_term(5, 8, 7, 1) # A12 B22 -> C12
    
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # A20(6), A21(7)
    # B00(0), B01(1), B10(3), B11(4)
    # C20 (0*3+2=2), C21 (1*3+2=5)
    
    add_term(6, 0, 2, 1) # A20 B00 -> C20
    add_term(7, 3, 2, 1) # A21 B10 -> C20
    add_term(6, 1, 5, 1) # A20 B01 -> C21
    add_term(7, 4, 5, 1) # A21 B11 -> C21
    
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    # A22(8)
    # B20(6), B21(7)
    # C20(2), C21(5)
    
    add_term(8, 6, 2, 1) # A22 B20 -> C20
    add_term(8, 7, 5, 1) # A22 B21 -> C21
    
    # A[2, 0:2] * B[0:2, 2] -> C[2, 2]
    # A20(6), A21(7)
    # B02(2), B12(5)
    # C22 (2*3+2=8)
    
    add_term(6, 2, 8, 1) # A20 B02 -> C22
    add_term(7, 5, 8, 1) # A21 B12 -> C22
    
    # A[2, 2] * B[2, 2] -> C[2, 2]
    # A22(8), B22(8)
    # C22(8)
    
    add_term(8, 8, 8, 1) # A22 B22 -> C22
    
    return {"U": U, "V": V, "W": W}
