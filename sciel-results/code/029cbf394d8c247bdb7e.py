import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=3
    # U shape: (n*m, rank) = (4, 11)
    # V shape: (m*p, rank) = (6, 11)
    # W shape: (p*n, rank) = (6, 11)
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Rank 7 decomposition for 2x2x2 (first 2 columns of B)
    # B columns 0 and 1 correspond to V rows:
    # j=0, k=0 -> row 0
    # j=0, k=1 -> row 1
    # j=1, k=0 -> row 3
    # j=1, k=1 -> row 4
    
    # C columns 0 and 1 correspond to W rows:
    # k=0, i=0 -> row 0
    # k=0, i=1 -> row 1
    # k=1, i=0 -> row 2
    # k=1, i=1 -> row 3
    
    # A rows:
    # i=0, j=0 -> row 0
    # i=0, j=1 -> row 1
    # i=1, j=0 -> row 2
    # i=1, j=1 -> row 3
    
    # Strassen's 7 terms
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10 + A11) B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3 = A00 (B01 - B11)
    U[0][2] += 1
    V[1][2] += 1; V[4][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11 (B10 - B00)
    U[3][3] += 1
    V[3][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A00 + A01) B11
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] -= 1
    
    # M6 = (A10 - A00) (B00 + B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A01 - A11) (B10 + B11)
    U[1][6] += 1; U[3][6] -= 1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1

    # Part 2: Rank 4 decomposition for 2x2x1 (3rd column of B)
    # B column 2 corresponds to V rows:
    # j=0, k=2 -> row 2
    # j=1, k=2 -> row 5
    
    # C column 2 corresponds to W rows:
    # k=2, i=0 -> row 4
    # k=2, i=1 -> row 5
    
    # Term 1: A00 B02 -> C02
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # Term 2: A01 B12 -> C02
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # Term 3: A10 B02 -> C12
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # Term 4: A11 B12 -> C12
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1

    return {"U": U, "V": V, "W": W}
