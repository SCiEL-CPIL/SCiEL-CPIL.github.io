"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices for U: i*m+j -> 0:a, 1:b, 2:c, 3:d
    # Indices for V: j*p'+k' -> 0:e, 1:f, 2:g, 3:h (p'=2)
    # Indices for W: k'*n+i -> 0:C00, 1:C01, 2:C10, 3:C11
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V_S = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Linear decomposition for 2x2x1
    # U: 4x4, V: 2x4, W: 2x4
    # P1=ae, P2=bg, P3=ce, P4=dg
    U_L = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    V_L = [
        [1, 0, 1, 0], # j=0: e, 0, e, 0
        [0, 1, 0, 1]  # j=1: 0, g, 0, g
    ]
    
    W_L = [
        [1, 1, 0, 0], # k=0, i=0: C00 gets a, b
        [0, 0, 1, 1]  # k=0, i=1: C10 gets c, d
    ]

    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # 4 Strassen blocks for cols 0-1, 2-3, 4-5, 6-7
    for k_start in [0, 2, 4, 6]:
        for c in range(7):
            # U
            for r in range(4):
                U[r][col_offset + c] = U_S[r][c]
            
            # V
            # Local V rows: 0 (j=0, k'=0), 1 (j=0, k'=1), 2 (j=1, k'=0), 3 (j=1, k'=1)
            # Global V rows:
            # k'=0 -> k_start, k'=1 -> k_start+1
            # j=0 -> 0*9 + k, j=1 -> 1*9 + k
            V[k_start][col_offset + c] = V_S[0][c]
            V[k_start + 1][col_offset + c] = V_S[1][c]
            V[9 + k_start][col_offset + c] = V_S[2][c]
            V[9 + k_start + 1][col_offset + c] = V_S[3][c]
            
            # W
            # Local W rows: 0 (k'=0, i=0), 1 (k'=0, i=1), 2 (k'=1, i=0), 3 (k'=1, i=1)
            # Global W rows:
            # k'=0 -> 2*k_start, 2*k_start+1
            # k'=1 -> 2*(k_start+1), 2*(k_start+1)+1
            W[2 * k_start][col_offset + c] = W_S[0][c]
            W[2 * k_start + 1][col_offset + c] = W_S[1][c]
            W[2 * (k_start + 1)][col_offset + c] = W_S[2][c]
            W[2 * (k_start + 1) + 1][col_offset + c] = W_S[3][c]
            
        col_offset += 7
        
    # 1 Linear block for col 8
    k_start = 8
    for c in range(4):
        # U
        for r in range(4):
            U[r][col_offset + c] = U_L[r][c]
        
        # V
        # Local V rows: 0 (j=0), 1 (j=1)
        V[k_start][col_offset + c] = V_L[0][c]
        V[9 + k_start][col_offset + c] = V_L[1][c]
        
        # W
        # Local W rows: 0 (i=0), 1 (i=1)
        W[2 * k_start][col_offset + c] = W_L[0][c]
        W[2 * k_start + 1][col_offset + c] = W_L[1][c]
        
        col_offset += 4

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
