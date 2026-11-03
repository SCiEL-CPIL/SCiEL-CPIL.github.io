"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(u_rows, v_rows, w_rows, offset):
        nonlocal col
        # Strassen coefficients
        # M1 = A00(B00+B11)
        # M2 = A10(B00+B01)
        # M3 = A01(B10-B11)
        # M4 = A11(B11-B00)
        # M5 = (A00+A11)(B01+B10)
        # M6 = (A01-A11)(B10+B11)
        # M7 = (A00+A10)(B00-B10)
        
        # u_rows: indices in U for A00, A01, A10, A11
        # v_rows: indices in V for B00, B01, B10, B11
        # w_rows: indices in W for C00, C01, C10, C11
        
        # Term 1
        U[u_rows[0]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[1]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1
        W[w_rows[1]][col] += 1
        W[w_rows[2]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # Term 2
        U[u_rows[2]][col] += 1
        V[v_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        W[w_rows[0]][col] += 1
        W[w_rows[2]][col] += 1
        col += 1
        
        # Term 3
        U[u_rows[1]][col] += 1
        V[v_rows[2]][col] += 1
        V[v_rows[3]][col] -= 1
        W[w_rows[1]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # Term 4
        U[u_rows[3]][col] += 1
        V[v_rows[0]][col] -= 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # Term 5
        U[u_rows[0]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[1]][col] += 1
        V[v_rows[2]][col] += 1
        W[w_rows[1]][col] += 1
        W[w_rows[2]][col] += 1
        col += 1
        
        # Term 6
        U[u_rows[1]][col] += 1
        U[u_rows[3]][col] -= 1
        V[v_rows[2]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[1]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # Term 7
        U[u_rows[0]][col] += 1
        U[u_rows[2]][col] += 1
        V[v_rows[0]][col] += 1
        V[v_rows[2]][col] -= 1
        W[w_rows[0]][col] += 1
        W[w_rows[2]][col] += 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # U rows: i*m + j
    # A00: 0*3+0=0, A01: 0*3+1=1, A10: 1*3+0=3, A11: 1*3+1=4
    u_rows_1 = [0, 1, 3, 4]
    # V rows: j*p + k
    # B00: 0*5+0=0, B01: 0*5+1=1, B10: 1*5+0=5, B11: 1*5+1=6
    v_rows_1 = [0, 1, 5, 6]
    # W rows: k*n + i
    # C00: 0*2+0=0, C01: 0*2+1=1, C10: 1*2+0=2, C11: 1*2+1=3
    w_rows_1 = [0, 1, 2, 3]
    add_strassen_block(u_rows_1, v_rows_1, w_rows_1, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    u_rows_2 = [0, 1, 3, 4]
    # B02: 0*5+2=2, B03: 0*5+3=3, B12: 1*5+2=7, B13: 1*5+3=8
    v_rows_2 = [2, 3, 7, 8]
    # C02: 2*2+0=4, C03: 2*2+1=5, C12: 3*2+0=6, C13: 3*2+1=7
    w_rows_2 = [4, 5, 6, 7]
    add_strassen_block(u_rows_2, v_rows_2, w_rows_2, 0)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5]
    # Rank 4
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04 -> C04
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1 # C04: k=4, i=0 -> 4*2+0=8
    col += 1
    
    # Term 2: A01 B14 -> C04
    U[1][col] = 1
    V[9][col] = 1 # B14: j=1, k=4 -> 1*5+4=9
    W[8][col] = 1
    col += 1
    
    # Term 3: A10 B04 -> C14
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1 # C14: k=4, i=1 -> 4*2+1=9
    col += 1
    
    # Term 4: A11 B14 -> C14
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5]
    # Rank 10
    # Ci_k = Ai_2 B_2_k
    for k in range(5):
        for i in range(2):
            # U index: i*m + 2 = i*3 + 2
            u_idx = i * 3 + 2
            # V index: 2*p + k = 2*5 + k = 10 + k
            v_idx = 10 + k
            # W index: k*n + i = k*2 + i
            w_idx = k * 2 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
