"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    # Block decomposition
    # I-blocks: [0,1], [2,3] (sizes 2,2)
    # J-blocks: [0,1], [2,3], [4] (sizes 2,2,1)
    # K-blocks: [0,1], [2,3], [4] (sizes 2,2,1)
    
    i_blocks = [(0, 2), (2, 2)]  # (start, size)
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]
    
    # Strassen decomposition for 2x2x2
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    terms = []
    
    for i0, i_sz in i_blocks:
        for j0, j_sz in j_blocks:
            for k0, k_sz in k_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    for u_local, v_local, w_local in strassen_terms:
                        U_vec = [0] * (n * m)
                        V_vec = [0] * (m * p)
                        W_vec = [0] * (p * n)
                        for lu in range(4):
                            U_vec[(i0 + lu // 2) * m + (j0 + lu % 2)] = u_local[lu]
                        for lv in range(4):
                            V_vec[(j0 + lv // 2) * p + (k0 + lv % 2)] = v_local[lv]
                        for lw in range(4):
                            W_vec[(k0 + lw // 2) * n + (i0 + lw % 2)] = w_local[lw]
                        terms.append((U_vec, V_vec, W_vec))
                else:
                    U_vec = [0] * (n * m)
                    V_vec = [0] * (m * p)
                    W_vec = [0] * (p * n)
                    for i in range(i_sz):
                        for j in range(j_sz):
                            U_vec[(i0 + i) * m + (j0 + j)] = 1
                    for j in range(j_sz):
                        for k in range(k_sz):
                            V_vec[(j0 + j) * p + (k0 + k)] = 1
                    for k in range(k_sz):
                        for i in range(i_sz):
                            W_vec[(k0 + k) * n + (i0 + i)] = 1
                    terms.append((U_vec, V_vec, W_vec))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        U_vec, V_vec, W_vec = terms[c]
        for i in range(n * m):
            if U_vec[i] != 0:
                U[i][c] = U_vec[i]
        for i in range(m * p):
            if V_vec[i] != 0:
                V[i][c] = V_vec[i]
        for i in range(p * n):
            if W_vec[i] != 0:
                W[i][c] = W_vec[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
