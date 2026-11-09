"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill
    
    # Use block-wise Strassen approach
    # Split 4x4 x 4x6 into 2x2 x 2x2 blocks
    # n_blocks=2, m_blocks=2, p_blocks=3
    # Each 2x2x2 block uses Strassen rank-7
    # Total rank = 2*2*3*7 = 84
    
    # 2x2 Strassen factors
    # U: 4x7 (rows indexed by (i,j) in {0,1}^2, row = i*2+j)
    # V: 4x7 (rows indexed by (j,k) in {0,1}^2, row = j*2+k)
    # W: 4x7 (rows indexed by (k,i) in {0,1}^2, row = k*2+i)
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    S_U = [
        [1, 1, 1, 0, 1, -1, 0],   # A00 (row 0)
        [0, 0, 0, 0, 0, 0, 1],    # A01 (row 1)
        [0, 1, 0, 0, 1, 1, 0],    # A10 (row 2)
        [1, 1, 0, 1, 0, 0, -1],   # A11 (row 3)
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00 (row 0)
        [0, 0, 1, 0, 0, 1, 0],    # B01 (row 1)
        [0, 0, 0, 1, 0, 0, 1],    # B10 (row 2)
        [1, 0, -1, 0, 1, 0, 1],   # B11 (row 3)
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, -1],  # C00 (row 0)
        [0, 1, 0, 1, 0, 0, 0],    # C01 (row 1)
        [0, 0, 1, 0, 1, 0, 0],    # C10 (row 2)
        [1, -1, 1, 0, 0, 1, 1],   # C11 (row 3)
    ]
    
    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2
    
    total_rank = n_blocks * m_blocks * p_blocks * 7  # 84
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    # U: rows (i,j) where i = bi*2+di, j = bj*2+dj
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = bi * block_size + di
                            j = bj * block_size + dj
                            U[i * m + j][col] = S_U[di * block_size + dj][c]
                    
                    # V: rows (j,k) where j = bj*2+dj, k = bk*2+dk
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = bj * block_size + dj
                            k = bk * block_size + dk
                            V[j * p + k][col] = S_V[dj * block_size + dk][c]
                    
                    # W: rows (k,i) where k = bk*2+dk, i = bi*2+di
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = bk * block_size + dk
                            i = bi * block_size + di
                            W[k * n + i][col] = S_W[dk * block_size + di][c]
                
                col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
