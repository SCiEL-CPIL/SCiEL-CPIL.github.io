import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_start, j_start, k_start):
        nonlocal col
        # A block: i_start..i_start+1, j_start..j_start+1
        # B block: j_start..j_start+1, k_start..k_start+1
        # C block: k_start..k_start+1, i_start..i_start+1
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        U[i_start*4 + j_start][col] += 1
        U[(i_start+1)*4 + (j_start+1)][col] += 1
        V[j_start*5 + k_start][col] += 1
        V[(j_start+1)*5 + (k_start+1)][col] += 1
        W[k_start*4 + i_start][col] += 1
        W[(k_start+1)*4 + (i_start+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_start+1)*4 + j_start][col] += 1
        U[(i_start+1)*4 + (j_start+1)][col] += 1
        V[j_start*5 + k_start][col] += 1
        W[k_start*4 + (i_start+1)][col] += 1
        W[(k_start+1)*4 + (i_start+1)][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i_start*4 + j_start][col] += 1
        V[j_start*5 + (k_start+1)][col] += 1
        V[(j_start+1)*5 + (k_start+1)][col] -= 1
        W[(k_start+1)*4 + i_start][col] += 1
        W[(k_start+1)*4 + (i_start+1)][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i_start+1)*4 + (j_start+1)][col] += 1
        V[(j_start+1)*5 + k_start][col] += 1
        V[j_start*5 + k_start][col] -= 1
        W[k_start*4 + i_start][col] += 1
        W[k_start*4 + (i_start+1)][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i_start*4 + j_start][col] += 1
        U[i_start*4 + (j_start+1)][col] += 1
        V[(j_start+1)*5 + (k_start+1)][col] += 1
        W[k_start*4 + i_start][col] -= 1
        W[(k_start+1)*4 + i_start][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_start+1)*4 + j_start][col] += 1
        U[i_start*4 + j_start][col] -= 1
        V[j_start*5 + k_start][col] += 1
        V[j_start*5 + (k_start+1)][col] += 1
        W[(k_start+1)*4 + (i_start+1)][col] += 1
        col += 1
        
        # M7 = (A11-A10)(B01+B11)
        U[(i_start+1)*4 + (j_start+1)][col] += 1
        U[(i_start+1)*4 + j_start][col] -= 1
        V[j_start*5 + (k_start+1)][col] += 1
        V[(j_start+1)*5 + (k_start+1)][col] += 1
        W[k_start*4 + i_start][col] += 1
        col += 1

    def add_naive_2x2x1(i_start, j_start, k_start):
        nonlocal col
        # A block: i_start..i_start+1, j_start..j_start+1
        # B block: j_start..j_start+1, k_start..k_start (width 1)
        # C block: k_start..k_start, i_start..i_start+1
        
        # C0 = A00*B0 + A01*B1
        U[i_start*4 + j_start][col] += 1
        V[j_start*5 + k_start][col] += 1
        W[k_start*4 + i_start][col] += 1
        col += 1
        
        U[i_start*4 + (j_start+1)][col] += 1
        V[(j_start+1)*5 + k_start][col] += 1
        W[k_start*4 + i_start][col] += 1
        col += 1
        
        # C1 = A10*B0 + A11*B1
        U[(i_start+1)*4 + j_start][col] += 1
        V[j_start*5 + k_start][col] += 1
        W[k_start*4 + (i_start+1)][col] += 1
        col += 1
        
        U[(i_start+1)*4 + (j_start+1)][col] += 1
        V[(j_start+1)*5 + k_start][col] += 1
        W[k_start*4 + (i_start+1)][col] += 1
        col += 1

    # A blocks:
    # A11: i=0, j=0
    # A12: i=0, j=2
    # A21: i=2, j=0
    # A22: i=2, j=2
    
    # B row blocks:
    # B_top: j=0
    # B_bot: j=2
    
    # C blocks:
    # C_top: i=0
    # C_bot: i=2
    
    # B col splits: 0-1 (2), 2-3 (2), 4 (1)
    
    # A11 * B_top
    add_strassen(0, 0, 0) # C00
    add_strassen(0, 0, 2) # C02
    add_naive_2x2x1(0, 0, 4) # C04
    
    # A12 * B_bot
    add_strassen(0, 2, 0) # C00
    add_strassen(0, 2, 2) # C02
    add_naive_2x2x1(0, 2, 4) # C04
    
    # A21 * B_top
    add_strassen(2, 0, 0) # C20
    add_strassen(2, 0, 2) # C22
    add_naive_2x2x1(2, 0, 4) # C24
    
    # A22 * B_bot
    add_strassen(2, 2, 0) # C20
    add_strassen(2, 2, 2) # C22
    add_naive_2x2x1(2, 2, 4) # C24
    
    return {"U": U, "V": V, "W": W}
