"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6 column-pairs, each using Strassen rank-7
    num_blocks = p // 2  # 6
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)
    
    # Strassen U vectors (for A[i,j], index i*2+j)
    U_s = [
        [1, 0, 1, 0],   # m1: a+d
        [0, 0, 1, 1],   # m2: c+d
        [1, 0, 0, 0],   # m3: a
        [0, 0, 0, 1],   # m4: d
        [1, 1, 0, 0],   # m5: a+b
        [0, -1, 1, 0],  # m6: c-a
        [0, 1, 0, -1],  # m7: b-d
    ]
    
    # Strassen V vectors (for B[j,k], index j*2+k)
    V_s = [
        [1, 0, 1, 0],   # m1: e+h
        [1, 0, 0, 0],   # m2: e
        [0, 1, 0, -1],  # m3: f-h
        [0, 0, 1, -1],  # m4: g-e
        [0, 0, 0, 1],   # m5: h
        [1, 1, 0, 0],   # m6: e+f
        [0, 0, 1, 1],   # m7: g+h
    ]
    
    # Strassen W vectors (for C[i,k], index k*2+i)
    W_s = [
        [1, 0, 0, 1],   # m1: C[0,0]+1, C[1,1]+1
        [0, 1, 0, -1],  # m2: C[1,0]+1, C[1,1]-1
        [0, 0, 1, 1],   # m3: C[0,1]+1, C[1,1]+1
        [1, 1, 0, 0],   # m4: C[0,0]+1, C[1,0]+1
        [-1, 0, 1, 0],  # m5: C[0,0]-1, C[0,1]+1
        [0, 0, 0, 1],   # m6: C[1,1]+1
        [1, 0, 0, -1],  # m7: C[0,0]+1, C[1,1]-1
    ]
    
    for s in range(num_blocks):
        for t in range(7):
            col = s * 7 + t
            # U
            for idx in range(4):
                U[idx][col] = U_s[t][idx]
            # V: B[j, 2s+local_k]
            for j in range(2):
                for local_k in range(2):
                    V[j * p + (2 * s + local_k)][col] = V_s[t][j * 2 + local_k]
            # W: C[i, 2s+local_k] -> index (2s+local_k)*n + i
            for i in range(2):
                for local_k in range(2):
                    W[(2 * s + local_k) * n + i][col] = W_s[t][local_k * 2 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
