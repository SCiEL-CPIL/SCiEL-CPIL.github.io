"""Strassen-based decomposition for 4x4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 coefficients
    # U_strassen: maps A (4 entries) to 7 linear combinations
    # V_strassen: maps B (4 entries) to 7 linear combinations
    # W_strassen: maps C (4 entries) to 7 linear combinations
    #
    # M1 = (a-d)(e+g)  -> U: [1,0,-1,0], V: [1,1,0,0], W: [1,0,0,1]
    # M2 = (c+d)e      -> U: [0,0,1,1], V: [1,0,0,0], W: [0,0,1,0]
    # M3 = a(f-g)      -> U: [1,0,0,0], V: [0,1,0,-1], W: [0,1,0,0]
    # M4 = d(h-f)      -> U: [0,0,0,1], V: [0,0,-1,1], W: [0,0,0,1]
    # M5 = (a+b)h      -> U: [1,1,0,0], V: [0,0,0,1], W: [1,0,0,0]
    # M6 = (c-a)(e+f)  -> U: [0,0,1,-1], V: [1,1,0,0], W: [0,1,0,0]
    # M7 = (b-d)(g+h)  -> U: [0,1,0,-1], V: [0,0,1,1], W: [0,0,1,0]
    
    U_s = [
        [1,  0, -1,  0],
        [0,  0,  1,  1],
        [1,  0,  0,  0],
        [0,  0,  0,  1],
        [1,  1,  0,  0],
        [0,  0,  1, -1],
        [0,  1,  0, -1],
    ]
    V_s = [
        [1,  1,  0,  0],
        [1,  0,  0,  0],
        [0,  1,  0, -1],
        [0,  0, -1,  1],
        [0,  0,  0,  1],
        [1,  1,  0,  0],
        [0,  0,  1,  1],
    ]
    W_s = [
        [1,  0,  0,  1],
        [0,  0,  1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1],
        [1,  0,  0,  0],
        [0,  1,  0,  0],
        [0,  0,  1,  0],
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # For each block pair (ii, jj) where ii in {0,1}, jj in {0,1}
    # A block is at rows ii*2..ii*2+1, cols jj*2..jj*2+1 of A (4x4)
    # B block is at rows jj*2..jj*2+1, cols kk*4..kk*4+3 of B (4x8)
    # C block is at rows ii*2..ii*2+1, cols kk*4..kk*4+3 of C (4x8)
    # 
    # For each kk in {0,1}, we do Strassen on 2x2x2
    # For kk=0: B cols 0,1; C cols 0,1
    # For kk=1: B cols 2,3; C cols 2,3
    
    col_offset = 0
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                for s in range(7):
                    # U: A block (ii, jj) local indices
                    # A[i,j] where i = ii*2 + local_i, j = jj*2 + local_j
                    for li in range(2):
                        for lj in range(2):
                            global_i = ii * 2 + li
                            global_j = jj * 2 + lj
                            U[global_i * m + global_j][col_offset] += U_s[s][li * 2 + lj]
                    
                    # V: B block (jj, kk) local indices
                    # B[j,k] where j = jj*2 + local_j, k = kk*4 + local_k (local_k in 0,1)
                    for lj in range(2):
                        for lk in range(2):
                            global_j = jj * 2 + lj
                            global_k = kk * 4 + lk  # cols 0,1 or 2,3
                            V[global_j * p + global_k][col_offset] += V_s[s][lj * 2 + lk]
                    
                    # W: C block (ii, kk) local indices
                    # C[i,k] where i = ii*2 + local_i, k = kk*4 + local_k (local_k in 0,1)
                    for li in range(2):
                        for lk in range(2):
                            global_i = ii * 2 + li
                            global_k = kk * 4 + lk
                            # W index: k*n + i
                            W[global_k * n + global_i][col_offset] += W_s[s][li * 2 + lk]
                    
                    col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
