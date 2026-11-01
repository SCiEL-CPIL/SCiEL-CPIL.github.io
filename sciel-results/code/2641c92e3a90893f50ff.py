import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 50s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen-based decomposition: 6 blocks of 2x2x2, rank 7 each = 42 total
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen factors for 2x2x2 matrix multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        
        # B indices for this block (V matrix: row = j*p + k)
        v_b00 = 0 * p + 2 * t      # B[0][2t]
        v_b01 = 0 * p + 2 * t + 1  # B[0][2t+1]
        v_b10 = 1 * p + 2 * t      # B[1][2t]
        v_b11 = 1 * p + 2 * t + 1  # B[1][2t+1]
        
        # C indices for this block (W matrix: row = k*n + i)
        w_c00 = (2 * t) * n + 0    # C[2t][0]
        w_c01 = (2 * t + 1) * n + 0 # C[2t+1][0]
        w_c10 = (2 * t) * n + 1    # C[2t][1]
        w_c11 = (2 * t + 1) * n + 1 # C[2t+1][1]
        
        # U indices: row = i*m + j
        u_a00 = 0 * m + 0  # A[0][0]
        u_a01 = 0 * m + 1  # A[0][1]
        u_a10 = 1 * m + 0  # A[1][0]
        u_a11 = 1 * m + 1  # A[1][1]
        
        # s=0: M1 = (A00 + A11) * (B00 + B11)
        c = base + 0
        U[u_a00][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1
        W[w_c11][c] = 1
        
        # s=1: M2 = (A10 + A11) * B00
        c = base + 1
        U[u_a10][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c10][c] = 1
        W[w_c11][c] = -1
        
        # s=2: M3 = A00 * (B01 - B11)
        c = base + 2
        U[u_a00][c] = 1
        V[v_b01][c] = 1
        V[v_b11][c] = -1
        W[w_c01][c] = 1
        W[w_c11][c] = 1
        
        # s=3: M4 = A11 * (B10 - B00)
        c = base + 3
        U[u_a11][c] = 1
        V[v_b10][c] = 1
        V[v_b00][c] = -1
        W[w_c00][c] = 1
        W[w_c10][c] = 1
        
        # s=4: M5 = (A00 + A01) * B11
        c = base + 4
        U[u_a00][c] = 1
        U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1
        W[w_c01][c] = 1
        
        # s=5: M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[u_a10][c] = 1
        U[u_a00][c] = -1
        V[v_b00][c] = 1
        V[v_b01][c] = 1
        W[w_c11][c] = 1
        
        # s=6: M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[u_a01][c] = 1
        U[u_a11][c] = -1
        V[v_b10][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
