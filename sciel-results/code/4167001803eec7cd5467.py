"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    def add_strassen_block(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_i0, c_i1, c_k0, c_k1, col_start):
        X00_u = c_i0 * m + a_j0
        X01_u = c_i0 * m + a_j1
        X10_u = c_i1 * m + a_j0
        X11_u = c_i1 * m + a_j1
        Y00_v = b_j0 * p + b_k0
        Y01_v = b_j0 * p + b_k1
        Y10_v = b_j1 * p + b_k0
        Y11_v = b_j1 * p + b_k1
        Z00_w = c_k0 * n + c_i0
        Z01_w = c_k1 * n + c_i0
        Z10_w = c_k0 * n + c_i1
        Z11_w = c_k1 * n + c_i1
        
        col = col_start + 0
        U[X00_u][col] += 1
        U[X11_u][col] += 1
        V[Y00_v][col] += 1
        V[Y11_v][col] += 1
        W[Z00_w][col] += 1
        W[Z11_w][col] += 1
        
        col = col_start + 1
        U[X10_u][col] += 1
        U[X11_u][col] += 1
        V[Y00_v][col] += 1
        W[Z10_w][col] += 1
        
        col = col_start + 2
        U[X00_u][col] += 1
        V[Y01_v][col] += 1
        V[Y11_v][col] += -1
        W[Z01_w][col] += 1
        W[Z11_w][col] += 1
        
        col = col_start + 3
        U[X11_u][col] += 1
        V[Y10_v][col] += 1
        V[Y00_v][col] += -1
        W[Z00_w][col] += 1
        W[Z10_w][col] += 1
        
        col = col_start + 4
        U[X00_u][col] += 1
        U[X01_u][col] += 1
        V[Y11_v][col] += 1
        W[Z01_w][col] += 1
        W[Z00_w][col] += -1
        
        col = col_start + 5
        U[X10_u][col] += 1
        U[X00_u][col] += -1
        V[Y00_v][col] += 1
        V[Y01_v][col] += 1
        W[Z11_w][col] += 1
        
        col = col_start + 6
        U[X01_u][col] += 1
        U[X11_u][col] += -1
        V[Y10_v][col] += 1
        V[Y11_v][col] += 1
        W[Z00_w][col] += 1
        W[Z11_w][col] += -1
    
    add_strassen_block(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
    add_strassen_block(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 7)
    add_strassen_block(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 14)
    add_strassen_block(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 21)
    
    if time.time() > deadline:
        pass  # Already done, just return
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
