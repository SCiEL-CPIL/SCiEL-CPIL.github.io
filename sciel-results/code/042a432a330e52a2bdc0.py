"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Schoolbook decomposition: for each (i, j, k), add rank-1 term
    # U index: i*m + j (A[i,j])
    # V index: j*p + k (B[j,k])
    # W index: k*n + i (C[k,i]) per trace convention W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
