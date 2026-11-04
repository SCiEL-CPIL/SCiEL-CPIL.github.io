import time
import sys

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition:
    # 7 for 4x4x4 (cols 0-3)
    # 7 for 4x4x4 (cols 4-7)
    # 4 for 4x4x1 (col 8)
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 18
    V = [[0] * rank for _ in range(m * p)] # 36 x 18
    W = [[0] * rank for _ in range(p * n)] # 36 x 18
    
    # Helper to set Strassen block
    def add_strassen_block(b_offset, c_offset, term_offset):
        # b_offset: shift for B columns (0 or 4)
        # c_offset: shift for C columns (0 or 4)
        # term_offset: shift for term index (0 or 7)
        
        # Term 1: (A11+A22)(B11+B22)
        # U: A11 (0,0),(0,1),(1,0),(1,1) and A22 (2,2),(2,3),(3,2),(3,3)
        for i in [0, 1, 2, 3]:
            for j in [0, 1]:
                if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                    U[i * m + j][term_offset + 0] = 1
        # V: B11 and B22
        for i in [0, 1]:
            for j in [0, 1]:
                V[i * p + (j + b_offset)][term_offset + 0] = 1
        for i in [2, 3]:
            for j in [2, 3]:
                V[i * p + (j + b_offset)][term_offset + 0] = 1
        # W: C11 (+), C22 (+)
        # C11: k=0, i=0 -> idx (0+c_offset)*n + 0
        W[(0 + c_offset) * n + 0][term_offset + 0] = 1
        # C22: k=3, i=3 -> idx (3+c_offset)*n + 3
        W[(3 + c_offset) * n + 3][term_offset + 0] = 1
        
        # Term 2: (A21+A22)B11
        # U: A21 (rows 2,3 cols 0,1) and A22 (rows 2,3 cols 2,3)
        for i in [2, 3]:
            for j in [0, 1, 2, 3]:
                U[i * m + j][term_offset + 1] = 1
        # V: B11
        for i in [0, 1]:
            for j in [0, 1]:
                V[i * p + (j + b_offset)][term_offset + 1] = 1
        # W: C21 (+), C22 (-)
        # C21: k=2, i=0
        W[(2 + c_offset) * n + 0][term_offset + 1] = 1
        # C22: k=3, i=3
        W[(3 + c_offset) * n + 3][term_offset + 1] = -1
        
        # Term 3: A11(B12-B22)
        # U: A11
        for i in [0, 1]:
            for j in [0, 1]:
                U[i * m + j][term_offset + 2] = 1
        # V: B12 (1), B22 (-1)
        for i in [0, 1]:
            for j in [2, 3]:
                V[i * p + (j + b_offset)][term_offset + 2] = 1
        for i in [2, 3]:
            for j in [2, 3]:
                V[i * p + (j + b_offset)][term_offset + 2] = -1
        # W: C12 (+), C22 (+)
        # C12: k=0, i=2
        W[(0 + c_offset) * n + 2][term_offset + 2] = 1
        # C22: k=3, i=3
        W[(3 + c_offset) * n + 3][term_offset + 2] = 1
        
        # Term 4: A22(B21-B11)
        # U: A22
        for i in [2, 3]:
            for j in [2, 3]:
                U[i * m + j][term_offset + 3] = 1
        # V: B21 (1), B11 (-1)
        for i in [2, 3]:
            for j in [0, 1]:
                V[i * p + (j + b_offset)][term_offset + 3] = 1
        for i in [0, 1]:
            for j in [0, 1]:
                V[i * p + (j + b_offset)][term_offset + 3] = -1
        # W: C11 (+), C21 (+)
        # C11: k=0, i=0
        W[(0 + c_offset) * n + 0][term_offset + 3] = 1
        # C21: k=2, i=0
        W[(2 + c_offset) * n + 0][term_offset + 3] = 1
        
        # Term 5: (A11+A12)B22
        # U: A11, A12
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                U[i * m + j][term_offset + 4] = 1
        # V: B22
        for i in [2, 3]:
            for j in [2, 3]:
                V[i * p + (j + b_offset)][term_offset + 4] = 1
        # W: C11 (-), C12 (+)
        # C11: k=0, i=0
        W[(0 + c_offset) * n + 0][term_offset + 4] = -1
        # C12: k=0, i=2
        W[(0 + c_offset) * n + 2][term_offset + 4] = 1
        
        # Term 6: (A21-A11)(B11+B12)
        # U: A21 (1), A11 (-1)
        for i in [2, 3]:
            for j in [0, 1]:
                U[i * m + j][term_offset + 5] = 1
        for i in [0, 1]:
            for j in [0, 1]:
                U[i * m + j][term_offset + 5] = -1
        # V: B11 (1), B12 (1)
        for i in [0, 1]:
            for j in [0, 1]:
                V[i * p + (j + b_offset)][term_offset + 5] = 1
        for i in [0, 1]:
            for j in [2, 3]:
                V[i * p + (j + b_offset)][term_offset + 5] = 1
        # W: C22 (+)
        # C22: k=3, i=3
        W[(3 + c_offset) * n + 3][term_offset + 5] = 1
        
        # Term 7: (A12-A22)(B21+B22)
        # U: A12 (1), A22 (-1)
        for i in [0, 1]:
            for j in [2, 3]:
                U[i * m + j][term_offset + 6] = 1
        for i in [2, 3]:
            for j in [2, 3]:
                U[i * m + j][term_offset + 6] = -1
        # V: B21 (1), B22 (1)
        for i in [2, 3]:
            for j in [0, 1]:
                V[i * p + (j + b_offset)][term_offset + 6] = 1
        for i in [2, 3]:
            for j in [2, 3]:
                V[i * p + (j + b_offset)][term_offset + 6] = 1
        # W: C11 (+)
        # C11: k=0, i=0
        W[(0 + c_offset) * n + 0][term_offset + 6] = 1

    # Add first 4x4x4 block (cols 0-3)
    add_strassen_block(0, 0, 0)
    
    # Add second 4x4x4 block (cols 4-7)
    add_strassen_block(4, 4, 7)
    
    # Add 4x4x1 block (col 8)
    # Schoolbook: C[i, 8] = sum_j A[i, j] B[j, 8]
    # 4 terms, one for each j
    for j in range(4):
        t = 14 + j
        # U: A[:, j]
        for i in range(4):
            U[i * m + j][t] = 1
        # V: B[j, 8]
        V[j * p + 8][t] = 1
        # W: C[:, 8]
        for i in range(4):
            W[8 * n + i][t] = 1
            
    return {"U": U, "V": V, "W": W}
