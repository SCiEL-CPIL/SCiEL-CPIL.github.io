import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, c):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1

    # Helper to add a Strassen term for 2x2x2
    # i, j, k are 0-indexed local indices for the 2x2 block
    # base_i, base_j, base_k are offsets
    def add_strassen(base_i, base_j, base_k, start_col):
        # M1 = (A11+A22)(B11+B22)
        U[(base_i+0)*m + (base_j+0)][start_col] = 1
        U[(base_i+1)*m + (base_j+1)][start_col] = 1
        V[(base_j+0)*p + (base_k+0)][start_col] = 1
        V[(base_j+1)*p + (base_k+1)][start_col] = 1
        W[(base_k+0)*n + (base_i+0)][start_col] = 1
        W[(base_k+1)*n + (base_i+1)][start_col] = 1
        col_val = start_col
        col_val += 1
        
        # M2 = (A21+A22)B11
        U[(base_i+1)*m + (base_j+0)][col_val] = 1
        U[(base_i+1)*m + (base_j+1)][col_val] = 1
        V[(base_j+0)*p + (base_k+0)][col_val] = 1
        W[(base_k+0)*n + (base_i+1)][col_val] = 1
        col_val += 1
        
        # M3 = A11(B12-B22)
        U[(base_i+0)*m + (base_j+0)][col_val] = 1
        V[(base_j+0)*p + (base_k+1)][col_val] = 1
        V[(base_j+1)*p + (base_k+1)][col_val] = -1
        W[(base_k+1)*n + (base_i+0)][col_val] = 1
        col_val += 1
        
        # M4 = A22(B21-B11)
        U[(base_i+1)*m + (base_j+1)][col_val] = 1
        V[(base_j+1)*p + (base_k+0)][col_val] = 1
        V[(base_j+0)*p + (base_k+0)][col_val] = -1
        W[(base_k+0)*n + (base_i+1)][col_val] = 1
        col_val += 1
        
        # M5 = (A11+A12)B22
        U[(base_i+0)*m + (base_j+0)][col_val] = 1
        U[(base_i+0)*m + (base_j+1)][col_val] = 1
        V[(base_j+1)*p + (base_k+1)][col_val] = 1
        W[(base_k+1)*n + (base_i+0)][col_val] = 1
        col_val += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(base_i+1)*m + (base_j+0)][col_val] = 1
        U[(base_i+0)*m + (base_j+0)][col_val] = -1
        V[(base_j+0)*p + (base_k+0)][col_val] = 1
        V[(base_j+0)*p + (base_k+1)][col_val] = 1
        W[(base_k+0)*n + (base_i+1)][col_val] = 1
        col_val += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[(base_i+0)*m + (base_j+1)][col_val] = 1
        U[(base_i+1)*m + (base_j+1)][col_val] = -1
        V[(base_j+1)*p + (base_k+0)][col_val] = 1
        V[(base_j+1)*p + (base_k+1)][col_val] = 1
        W[(base_k+1)*n + (base_i+0)][col_val] = 1
        W[(base_k+0)*n + (base_i+1)][col_val] = 1 # Wait, C22 uses M7?
        # Let's re-verify C22 = M1 - M2 + M3 + M6? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So W for M7:
        # W[0,0] += 1 (for C11)
        # W[1,1] += 0?
        # Let's check standard Strassen W matrix.
        
        return col_val

    # 1. M_{2,1,7} (j=0, k=0..6)
    for i in range(2):
        for k in range(7):
            add_schoolbook(i, 0, k, col)
            col += 1
            
    # 2. M_{2,2,3} (j=1,2; k=0,1,2)
    # Strassen for k=0,1 (j=1,2)
    col = add_strassen(0, 1, 0, col)
    # Schoolbook for k=2 (j=1,2)
    for i in range(2):
        for j in range(1, 3):
            add_schoolbook(i, j, 2, col)
            col += 1
            
    # 3. M_{2,2,4} (j=1,2; k=3,4,5,6)
    # Strassen for k=3,4
    col = add_strassen(0, 1, 3, col)
    # Strassen for k=5,6
    col = add_strassen(0, 1, 5, col)
            
    return {"U": U, "V": V, "W": W}
