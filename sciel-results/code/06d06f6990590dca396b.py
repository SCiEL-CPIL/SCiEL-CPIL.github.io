import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Part 1: Strassen for C[0:2, 0:2]
    # A indices: 00, 01, 10, 11 -> flat: 0, 1, 3, 4
    # B indices: 00, 01, 10, 11 -> flat: 0, 1, 5, 6
    # C indices: 00, 01, 10, 11 -> flat: 0, 1, 3, 4 (k*n+i)
    # k=0, i=0 -> 0
    # k=1, i=0 -> 1
    # k=0, i=1 -> 3
    # k=1, i=1 -> 4
    
    # Strassen Multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][idx] = 1; U[4][idx] = 1
    V[0][idx] = 1; V[6][idx] = 1
    W[0][idx] = 1; W[1][idx] = -1; W[3][idx] = 1; W[4][idx] = -1
    idx += 1
    
    # M2 = (A10 + A11) * B00
    U[3][idx] = 1; U[4][idx] = 1
    V[0][idx] = 1
    W[0][idx] = -1; W[3][idx] = 1
    idx += 1
    
    # M3 = A00 * (B10 - B00)
    U[0][idx] = 1
    V[5][idx] = 1; V[0][idx] = -1
    W[1][idx] = 1; W[3][idx] = 1
    idx += 1
    
    # M4 = A11 * (B11 - B01)
    U[4][idx] = 1
    V[6][idx] = 1; V[1][idx] = -1
    W[0][idx] = -1; W[1][idx] = 1
    idx += 1
    
    # M5 = (A00 + A01) * B11
    U[0][idx] = 1; U[1][idx] = 1
    V[6][idx] = 1
    W[0][idx] = 1; W[4][idx] = 1
    idx += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3][idx] = 1; U[0][idx] = -1
    V[0][idx] = 1; V[1][idx] = 1
    W[4][idx] = 1
    idx += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][idx] = 1; U[4][idx] = -1
    V[5][idx] = 1; V[6][idx] = 1
    W[0][idx] = -1; W[3][idx] = 1
    idx += 1
    
    # Part 2: Schoolbook for remaining 3x3x3
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # Helper to add schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal idx
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
        idx += 1
        
    # C02 (i=0, k=2)
    add_schoolbook(0, 0, 2)
    add_schoolbook(0, 1, 2)
    add_schoolbook(0, 2, 2)
    
    # C12 (i=1, k=2)
    add_schoolbook(1, 0, 2)
    add_schoolbook(1, 1, 2)
    add_schoolbook(1, 2, 2)
    
    # C20 (i=2, k=0)
    add_schoolbook(2, 0, 0)
    add_schoolbook(2, 1, 0)
    add_schoolbook(2, 2, 0)
    
    # C21 (i=2, k=1)
    add_schoolbook(2, 0, 1)
    add_schoolbook(2, 1, 1)
    add_schoolbook(2, 2, 1)
    
    # C22 (i=2, k=2)
    add_schoolbook(2, 0, 2)
    add_schoolbook(2, 1, 2)
    add_schoolbook(2, 2, 2)
    
    # Part 3: Schoolbook for 3x3x2 (k=3,4)
    # C03, C04, C13, C14, C23, C24
    for i in range(3):
        for k in [3, 4]:
            for j in range(3):
                add_schoolbook(i, j, k)
                
    return {"U": U, "V": V, "W": W}
