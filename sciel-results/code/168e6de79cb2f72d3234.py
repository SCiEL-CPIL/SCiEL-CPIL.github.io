import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    n, m, p = DIMENSIONS
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # Term 1: (A00 + A11) * (B00 + B11)
        U[i_offset * m + 0][col] += 1
        U[i_offset * m + 1][col] += 1
        V[0 * p + j_offset][col] += 1
        V[1 * p + j_offset][col] += 1
        W[0 * n + i_offset][col] += 1
        W[1 * n + i_offset + 1][col] += 1
        col += 1
        
        # Term 2: (A10 + A11) * B00
        U[i_offset * m + 1][col] += 1
        U[i_offset * m + 1 + 1][col] += 1
        V[0 * p + j_offset][col] += 1
        W[1 * n + i_offset][col] += 1
        col += 1
        
        # Term 3: A00 * (B01 - B11)
        U[i_offset * m + 0][col] += 1
        V[0 * p + j_offset + 1][col] += 1
        V[1 * p + j_offset + 1][col] += -1
        W[0 * n + i_offset + 1][col] += 1
        col += 1
        
        # Term 4: A11 * (B10 - B00)
        U[i_offset * m + 1 + 1][col] += 1
        V[1 * p + j_offset][col] += 1
        V[0 * p + j_offset][col] += -1
        W[1 * n + i_offset][col] += 1
        col += 1
        
        # Term 5: (A00 + A01) * B11
        U[i_offset * m + 0][col] += 1
        U[i_offset * m + 0 + 1][col] += 1
        V[1 * p + j_offset + 1][col] += 1
        W[0 * n + i_offset + 1][col] += 1
        W[1 * n + i_offset + 1][col] += -1
        col += 1
        
        # Term 6: (A10 - A00) * (B00 + B01)
        U[i_offset * m + 1][col] += 1
        U[i_offset * m + 0][col] += -1
        V[0 * p + j_offset][col] += 1
        V[0 * p + j_offset + 1][col] += 1
        W[1 * n + i_offset + 1][col] += 1
        col += 1
        
        # Term 7: (A01 - A11) * (B10 + B11)
        U[i_offset * m + 0 + 1][col] += 1
        U[i_offset * m + 1 + 1][col] += -1
        V[1 * p + j_offset][col] += 1
        V[1 * p + j_offset + 1][col] += 1
        W[0 * n + i_offset + 1][col] += 1
        W[1 * n + i_offset + 1][col] += 1
        col += 1

    # Block 1: A1 (cols 0-1) * B1 (rows 0-1)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    
    # Block 2: A2 (cols 2-3) * B2 (rows 2-3)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    
    # Block 3: A3 (col 4) * B3 (row 4)
    for k in range(6):
        U[0 * m + 4][col] = 1
        U[1 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    if time.time() > deadline:
        # Fallback to schoolbook if somehow this takes too long (it won't)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
