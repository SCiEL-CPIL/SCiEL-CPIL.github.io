"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Strassen's 7-term decomposition for 2x2 x 2x2
    # Applied to each pair of columns of B (4 pairs for p=8)
    # Total rank = 4 * 7 = 28
    
    # U coefficients indexed by [i*m+j] = [2i+j]
    # i=0,j=0 (a): M1=1,M2=0,M3=1,M4=0,M5=1,M6=-1,M7=0
    # i=0,j=1 (b): M1=0,M2=0,M3=1,M4=0,M5=1,M6=0,M7=1
    # i=1,j=0 (c): M1=0,M2=1,M3=0,M4=0,M5=0,M6=1,M7=0
    # i=1,j=1 (d): M1=1,M2=1,M3=0,M4=1,M5=0,M6=-1,M7=-1
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1],
    ]
    
    # V coefficients indexed by [j*p+k] locally [2j+k_local]
    # j=0,k=0 (e): M1=1,M2=1,M3=0,M4=-1,M5=0,M6=1,M7=0
    # j=0,k=1 (f): M1=0,M2=0,M3=1,M4=0,M5=0,M6=1,M7=0
    # j=1,k=0 (g): M1=0,M2=0,M3=0,M4=1,M5=0,M6=0,M7=1
    # j=1,k=1 (h): M1=1,M2=0,M3=-1,M4=0,M5=1,M6=0,M7=1
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    
    # W coefficients indexed by [k*n+i] locally [2*k_local+i]
    # k=0,i=0 (C[0,0]): M1=1,M2=0,M3=0,M4=1,M5=-1,M6=0,M7=1
    # k=0,i=1 (C[1,0]): M1=0,M2=1,M3=0,M4=1,M5=0,M6=0,M7=0
    # k=1,i=0 (C[0,1]): M1=0,M2=0,M3=1,M4=0,M5=1,M6=0,M7=0
    # k=1,i=1 (C[1,1]): M1=1,M2=-1,M3=1,M4=0,M5=0,M6=1,M7=0
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    num_blocks = p // 2  # 4 blocks
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    for block in range(num_blocks):
        col_offset = block * 7
        
        # U: A is shared across all blocks
        for u_idx in range(n * m):
            for c in range(7):
                U[u_idx][col_offset + c] = U_strassen[u_idx][c]
        
        # V: each block uses different columns of B
        for j in range(m):
            for k_local in range(2):
                k_global = 2 * block + k_local
                v_idx = j * p + k_global
                v_local_idx = j * 2 + k_local
                for c in range(7):
                    V[v_idx][col_offset + c] = V_strassen[v_local_idx][c]
        
        # W: each block uses different columns of C
        for k_local in range(2):
            k_global = 2 * block + k_local
            for i in range(n):
                w_idx = k_global * n + i
                w_local_idx = k_local * n + i
                for c in range(7):
                    W[w_idx][col_offset + c] = W_strassen[w_local_idx][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
