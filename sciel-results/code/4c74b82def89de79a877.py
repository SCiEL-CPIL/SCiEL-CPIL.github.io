"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]
n, m, p = DIMENSIONS

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2, rank 7."""
    # Strassen's 7 multiplications
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j,k in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)*(B11+B22) -> contributes to C11, C22
    # A11: i=0,j=0; A22: i=1,j=1
    # B11: j=0,k=0; B22: j=1,k=1
    # C11: k=0,i=0; C22: k=1,i=1
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    W[0][0] = 1  # C11 (k=0,i=0)
    W[3][0] = 1  # C22 (k=1,i=1)
    
    # M2: (A21+A22)*B11 -> contributes to C21
    # A21: i=1,j=0; A22: i=1,j=1
    # B11: j=0,k=0
    # C21: k=0,i=1
    U[2][1] = 1  # A21 (i=1,j=0)
    U[3][1] = 1  # A22 (i=1,j=1)
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21 (k=0,i=1)
    
    # M3: A11*(B12-B22) -> contributes to C12, C22
    # A11: i=0,j=0
    # B12: j=0,k=1; B22: j=1,k=1
    # C12: k=1,i=0; C22: k=1,i=1
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12 (j=0,k=1)
    V[3][2] = -1  # B22 (j=1,k=1)
    W[1][2] = 1  # C12 (k=1,i=0)
    W[3][2] = 1  # C22 (k=1,i=1)
    
    # M4: A22*(B21-B11) -> contributes to C11, C21
    # A22: i=1,j=1
    # B21: j=1,k=0; B11: j=0,k=0
    # C11: k=0,i=0; C21: k=0,i=1
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21 (j=1,k=0)
    V[0][3] = -1  # B11 (j=0,k=0)
    W[0][3] = 1  # C11 (k=0,i=0)
    W[2][3] = 1  # C21 (k=0,i=1)
    
    # M5: (A11+A12)*B22 -> contributes to C11, C12
    # A11: i=0,j=0; A12: i=0,j=1
    # B22: j=1,k=1
    # C11: k=0,i=0; C12: k=1,i=0
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12 (i=0,j=1)
    V[3][4] = 1  # B22
    W[0][4] = -1  # C11 (negative)
    W[1][4] = 1  # C12
    
    # M6: (A21-A11)*(B11+B12) -> contributes to C22
    # A21: i=1,j=0; A11: i=0,j=0
    # B11: j=0,k=0; B12: j=0,k=1
    # C22: k=1,i=1
    U[2][5] = 1  # A21
    U[0][5] = -1  # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[3][5] = 1  # C22
    
    # M7: (A12-A22)*(B21+B22) -> contributes to C11, C22
    # A12: i=0,j=1; A22: i=1,j=1
    # B21: j=1,k=0; B22: j=1,k=1
    # C11: k=0,i=0; C22: k=1,i=1
    U[1][6] = 1  # A12
    U[3][6] = -1  # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    W[0][6] = 1  # C11
    W[3][6] = -1  # C22 (negative)
    
    return U, V, W

def _matmul_2x2x3():
    """Decomposition for 2x2 * 2x3, rank 12 (schoolbook) or try to find rank 11."""
    n1, m1, p1 = 2, 2, 3
    rank = n1 * m1 * p1
    U = [[0]*rank for _ in range(n1*m1)]
    V = [[0]*rank for _ in range(m1*p1)]
    W = [[0]*rank for _ in range(p1*n1)]
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[i*m1+j][col] = 1
                V[j*p1+k][col] = 1
                W[k*n1+i][col] = 1
                col += 1
    return U, V, W

def _kron_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two matmul decompositions."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            for a1 in range(n1*m1):
                for a2 in range(n2*m2):
                    a = a1 * (n2*m2) + a2
                    U[a][c] = U1[a1][c1] * U2[a2][c2]
            for b1 in range(m1*p1):
                for b2 in range(m2*p2):
                    b = b1 * (m2*p2) + b2
                    V[b][c] = V1[b1][c1] * V2[b2][c2]
            for w1 in range(p1*n1):
                for w2 in range(p2*n2):
                    w = w1 * (p2*n2) + w2
                    W[w][c] = W1[w1][c1] * W2[w2][c2]
    
    return U, V, W

def _to_fraction_list(matrix):
    rows = []
    for row in matrix:
        frow = []
        for v in row:
            if isinstance(v, Fraction):
                frow.append(v)
            else:
                frow.append(Fraction(v).limit_denominator(10**6))
        rows.append(frow)
    return rows

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Strassen 2x2*2x2
    U1, V1, W1 = _strassen_2x2()
    # 2x2*2x3 schoolbook
    U2, V2, W2 = _matmul_2x2x3()
    
    # Kronecker product
    U, V, W = _kron_product(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 3)
    
    rank = len(U[0])
    
    # Convert to fractions
    U = _to_fraction_list(U)
    V = _to_fraction_list(V)
    W = _to_fraction_list(W)
    
    # Try to eliminate zero columns
    new_U = []
    new_V = []
    new_W = []
    for c in range(rank):
        if any(U[a][c] != 0 for a in range(len(U))) or \
           any(V[b][c] != 0 for b in range(len(V))) or \
           any(W[w][c] != 0 for w in range(len(W))):
            new_U.append([row[c] for row in U])
            new_V.append([row[c] for row in V])
            new_W.append([row[c] for row in W])
    
    if len(new_U) < rank:
        U = [col for col in zip(*new_U)]
        U = [list(row) for row in U]
        V = [col for col in zip(*new_V)]
        V = [list(row) for row in V]
        W = [col for col in zip(*new_W)]
        W = [list(row) for row in W]
        rank = len(U[0])
    
    return {"U": U, "V": V, "W": W}
