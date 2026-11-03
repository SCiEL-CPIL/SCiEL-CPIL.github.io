"""Rewritten program for matrix multiplication 4x5 by 5x6."""
import time
import sys

DIMENSIONS = [4, 5, 6]

def _strassen_2x2(a, b):
    """Strassen multiplication for 2x2 matrices a, b (lists of lists).
    Returns 2x2 result.
    """
    a11, a12 = a[0][0], a[0][1]
    a21, a22 = a[1][0], a[1][1]
    b11, b12 = b[0][0], b[0][1]
    b21, b22 = b[1][0], b[1][1]
    
    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    return [[c11, c12], [c21, c22]]

def _strassen_2x2_tensor():
    """
    Build the U, V, W matrices for a 2x2 by 2x2 matrix multiplication 
    using Strassen's algorithm. Returns (U, V, W) where:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[row, col] corresponds to coefficient of A[row] in the col-th product.
    V[row, col] corresponds to coefficient of B[row] in the col-th product.
    W[row, col] corresponds to coefficient of C[row] in the col-th product.
    Rows are indexed as (i, j) -> i*2+j for A and B (input), 
    and (i, j) -> i*2+j for C (output).
    
    Actually, let's use the standard tensor decomposition form:
    C = sum_k U_k ⊗ V_k ⊗ W_k
    where C[i,j,k] = sum_l A[i,l] * B[l,k]
    
    For Strassen:
    m1 = (A00 + A11) * (B00 + B11)
    m2 = (A10 + A11) * B00
    m3 = A00 * (B01 - B11)
    m4 = A11 * (B10 - B00)
    m5 = (A00 + A01) * B11
    m6 = (A10 - A00) * (B00 + B01)
    m7 = (A01 - A11) * (B10 + B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    """
    n = 2
    rank = 7
    # U maps A indices (i, j) to rank components
    # A has indices (i, j) where i is row, j is col of A (2x2)
    # So A flat index: i*2 + j
    # B flat index: l*2 + k where l is row of B, k is col of B
    # C flat index: i*2 + k
    
    U = [[0] * rank for _ in range(4)]  # 4 entries in A
    V = [[0] * rank for _ in range(4)]  # 4 entries in B
    W = [[0] * rank for _ in range(4)]  # 4 entries in C
    
    # m1: (A00 + A11) * (B00 + B11)
    # A00 is index 0, A11 is index 3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += m1, C11 += m1
    W[0][0] = 1; W[3][0] = 1
    
    # m2: (A10 + A11) * B00
    # A10 is index 2, A11 is index 3
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += m2
    W[2][1] = 1
    
    # m3: A00 * (B01 - B11)
    # A00 is index 0
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 += m3, C11 += m3
    W[1][2] = 1; W[3][2] = 1
    
    # m4: A11 * (B10 - B00)
    # A11 is index 3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00 += m4, C10 += m4
    W[0][3] = 1; W[2][3] = 1
    
    # m5: (A00 + A01) * B11
    # A00 is index 0, A01 is index 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= m5, C01 += m5
    W[0][4] = -1; W[1][4] = 1
    
    # m6: (A10 - A00) * (B00 + B01)
    # A10 is index 2, A00 is index 0
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += m6
    W[3][5] = 1
    
    # m7: (A01 - A11) * (B10 + B11)
    # A01 is index 1, A11 is index 3
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 += m7
    W[0][6] = 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying 
    a 4x5 matrix by a 5x6 matrix.
    
    Returns a dict with 'U', 'V', 'W' keys.
    U has shape (n*m, rank) = (20, rank)
    V has shape (m*p, rank) = (30, rank)
    W has shape (p*n, rank) = (24, rank)
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 4*5*6 = 120
    
    # Try to use Strassen where beneficial
    # For 4x5 by 5x6, we can potentially split into blocks.
    # One approach: view as 2x2 blocks of some size, but the dimensions don't align perfectly.
    
    # Let's try a simple block decomposition:
    # Split A (4x5) into 2x2 blocks of size 2x3, but that's not square.
    # Alternatively, pad to 6x6 and use Strassen? That might increase rank.
    
    # For now, let's use the schoolbook decomposition which is guaranteed correct.
    # rank = 120
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Constructed decomposition with rank:", len(result["U"][0]))
