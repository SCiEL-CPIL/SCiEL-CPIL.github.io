"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(matrix, row, col, val):
        matrix[row][col] = val
        
    # Block 1: i in {0, 1}, k in {0..3}, r in {0..7}
    # M_{2,4,4}
    for i in [0, 1]:
        for j in range(m):
            for r in range(8):
                # Local r'
                if i == 0:
                    r_local = r
                else:
                    r_local = r - 4
                if 0 <= r_local < m:
                    U[i * m + j][r] = 1 if j == r_local else 0
                    
    for j in range(m):
        for k in range(4):
            for r in range(8):
                V[j * p + k][r] = 1
                
    for k in range(4):
        for i in [0, 1]:
            for r in range(8):
                if i == 0:
                    if r < 4:
                        W[k * n + i][r] = 1
                else:
                    if 4 <= r < 8:
                        W[k * n + i][r] = 1
                        
    # Block 2: i = 2, k in {0..3}, r in {8..11}
    # M_{1,4,4}
    for j in range(m):
        for r in range(8, 12):
            r_local = r - 8
            if 0 <= r_local < m:
                U[2 * m + j][r] = 1 if j == r_local else 0
                
    for j in range(m):
        for k in range(4):
            for r in range(8, 12):
                V[j * p + k][r] = 1
                
    for k in range(4):
        for r in range(8, 12):
            W[k * n + 2][r] = 1

    # Block 3: i in {0, 1}, k in {4..7}, r in {12..19}
    # M_{2,4,4}
    for i in [0, 1]:
        for j in range(m):
            for r in range(12, 20):
                r_local = r - 12
                if i == 0:
                    r_use = r_local
                else:
                    r_use = r_local - 4
                if 0 <= r_use < m:
                    U[i * m + j][r] = 1 if j == r_use else 0
                    
    for j in range(m):
        for k in range(4, 8):
            for r in range(12, 20):
                V[j * p + k][r] = 1
                
    for k in range(4, 8):
        for i in [0, 1]:
            for r in range(12, 20):
                if i == 0:
                    if 12 <= r < 16:
                        W[k * n + i][r] = 1
                else:
                    if 16 <= r < 20:
                        W[k * n + i][r] = 1

    # Block 4: i = 2, k in {4..7}, r in {20..23}
    # M_{1,4,4}
    for j in range(m):
        for r in range(20, 24):
            r_local = r - 20
            if 0 <= r_local < m:
                U[2 * m + j][r] = 1 if j == r_local else 0
                
    for j in range(m):
        for k in range(4, 8):
            for r in range(20, 24):
                V[j * p + k][r] = 1
                
    for k in range(4, 8):
        for r in range(20, 24):
            W[k * n + 2][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
