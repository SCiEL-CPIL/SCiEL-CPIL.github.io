import time

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset, j_offset, k_offset are the starting indices for the 2x2 block
        # A rows: i_offset, i_offset+1
        # A cols: j_offset, j_offset+1
        # B rows: j_offset, j_offset+1
        # B cols: k_offset, k_offset+1
        # C rows: i_offset, i_offset+1
        # C cols: k_offset, k_offset+1
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[i_offset * m + j_offset][col] += 1
        U[(i_offset+1) * m + (j_offset+1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] += 1
        # Output to all 4 C entries
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset+1) * n + i_offset][col] += 1
        W[k_offset * n + (i_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        # M2 = A11 * (B10 - B00)
        U[(i_offset+1) * m + (j_offset+1)][col] += 1
        V[(j_offset+1) * p + k_offset][col] += 1
        V[j_offset * p + k_offset][col] -= 1
        # Output to C[1, 0:2] (local) -> Global C[i_offset+1, k_offset : k_offset+2]
        W[k_offset * n + (i_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        # M3 = (A00 - A10) * B11
        U[i_offset * m + j_offset][col] += 1
        U[(i_offset+1) * m + j_offset][col] -= 1
        V[(j_offset+1) * p + (k_offset+1)][col] += 1
        # Output to all 4
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset+1) * n + i_offset][col] += 1
        W[k_offset * n + (i_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        # M4 = A00 * (B01 - B11)
        U[i_offset * m + j_offset][col] += 1
        V[j_offset * p + (k_offset+1)][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] -= 1
        # Output to C[0, 0:2] (local) -> Global C[i_offset, k_offset : k_offset+2]
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset+1) * n + i_offset][col] += 1
        col += 1
        
        # M5 = (A01 + A11) * (B00 + B01)
        U[i_offset * m + (j_offset+1)][col] += 1
        U[(i_offset+1) * m + (j_offset+1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        V[j_offset * p + (k_offset+1)][col] += 1
        # Output to all 4
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset+1) * n + i_offset][col] += 1
        W[k_offset * n + (i_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        # M6 = (A00 - A10) * (B10 + B11)
        U[i_offset * m + j_offset][col] += 1
        U[(i_offset+1) * m + j_offset][col] -= 1
        V[(j_offset+1) * p + k_offset][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] += 1
        # Output to C[1, 0:2] (local) -> Global C[i_offset+1, k_offset : k_offset+2]
        W[k_offset * n + (i_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B00 - B10)
        U[i_offset * m + (j_offset+1)][col] += 1
        U[(i_offset+1) * m + (j_offset+1)][col] -= 1
        V[j_offset * p + k_offset][col] += 1
        V[(j_offset+1) * p + k_offset][col] -= 1
        # Output to all 4
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset+1) * n + i_offset][col] += 1
        W[k_offset * n + (i_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1

    def add_schoolbook_block(i_range, j_range, k_range):
        nonlocal col
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] += 1
                    V[j * p + k][col] += 1
                    W[k * n + i][col] += 1
                    col += 1

    # Group 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # Group 2: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    add_schoolbook_block([0, 1], [2], [0, 1])
    
    # Group 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 2)
    
    # Group 4: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    add_schoolbook_block([0, 1], [0, 1], [4])
    
    # Group 5: A[0:2, 2] * B[2, 2:5] -> C[0:2, 2:5]
    add_schoolbook_block([0, 1], [2], [2, 3, 4])
    
    return {"U": U, "V": V, "W": W}
