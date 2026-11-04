"""Matrix multiplication 5x5x5 with 2+2+1 Strassen-based block decomposition."""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    i_parts = [[0, 1], [2, 3], [4]]
    k_parts = [[0, 1], [2, 3], [4]]
    j_parts = [[0, 1], [2, 3], [4]]
    
    U_entries = []
    V_entries = []
    W_entries = []
    col = 0
    
    def add_strassen_2x2x2(I, K, J):
        nonlocal col
        i0, i1 = I
        k0, k1 = K
        j0, j1 = J
        
        # M1 = (A[i0,k0] + A[i1,k1]) * (B[k0,j0] + B[k1,j1])
        # C[i0,j0] += M1, C[i1,j1] += M1
        U_entries.append((i0*m + k0, 1))
        U_entries.append((i1*m + k1, 1))
        V_entries.append((k0*p + j0, 1))
        V_entries.append((k1*p + j1, 1))
        W_entries.append((j0*n + i0, 1))
        W_entries.append((j1*n + i1, 1))
        col += 1
        
        # M2 = (A[i1,k0] + A[i1,k1]) * B[k0,j0]
        # C[i1,j0] += M2, C[i1,j1] -= M2
        U_entries.append((i1*m + k0, 1))
        U_entries.append((i1*m + k1, 1))
        V_entries.append((k0*p + j0, 1))
        W_entries.append((j0*n + i1, 1))
        W_entries.append((j1*n + i1, -1))
        col += 1
        
        # M3 = A[i0,k0] * (B[k0,j1] - B[k1,j1])
        # C[i0,j1] += M3, C[i1,j1] += M3
        U_entries.append((i0*m + k0, 1))
        V_entries.append((k0*p + j1, 1))
        V_entries.append((k1*p + j1, -1))
        W_entries.append((j1*n + i0, 1))
        W_entries.append((j1*n + i1, 1))
        col += 1
        
        # M4 = A[i1,k1] * (B[k1,j0] - B[k0,j0])
        # C[i0,j0] += M4, C[i1,j0] += M4
        U_entries.append((i1*m + k1, 1))
        V_entries.append((k1*p + j0, 1))
        V_entries.append((k0*p + j0, -1))
        W_entries.append((j0*n + i0, 1))
        W_entries.append((j0*n + i1, 1))
        col += 1
        
        # M5 = (A[i0,k0] + A[i0,k1]) * B[k1,j1]
        # C[i0,j0] -= M5, C[i0,j1] += M5
        U_entries.append((i0*m + k0, 1))
        U_entries.append((i0*m + k1, 1))
        V_entries.append((k1*p + j1, 1))
        W_entries.append((j0*n + i0, -1))
        W_entries.append((j1*n + i0, 1))
        col += 1
        
        # M6 = (A[i1,k0] - A[i0,k0]) * (B[k0,j0] + B[k0,j1])
        # C[i1,j1] += M6
        U_entries.append((i1*m + k0, 1))
        U_entries.append((i0*m + k0, -1))
        V_entries.append((k0*p + j0, 1))
        V_entries.append((k0*p + j1, 1))
        W_entries.append((j1*n + i1, 1))
        col += 1
        
        # M7 = (A[i0,k1] - A[i1,k1]) * (B[k1,j0] + B[k1,j1])
        # C[i0,j0] += M7
        U_entries.append((i0*m + k1, 1))
        U_entries.append((i1*m + k1, -1))
        V_entries.append((k1*p + j0, 1))
        V_entries.append((k1*p + j1, 1))
        W_entries.append((j0*n + i0, 1))
        col += 1
    
    def add_schoolbook_block(I, K, J):
        nonlocal col
        for i in I:
            for k in K:
                for j in J:
                    U_entries.append((i*m + k, 1))
                    V_entries.append((k*p + j, 1))
                    W_entries.append((j*n + i, 1))
                    col += 1
    
    for i_b in range(3):
        for k_b in range(3):
            for j_b in range(3):
                I = i_parts[i_b]
                K = k_parts[k_b]
                J = j_parts[j_b]
                if len(I) == 2 and len(K) == 2 and len(J) == 2:
                    add_strassen_2x2x2(I, K, J)
                else:
                    add_schoolbook_block(I, K, J)
    
    rank = col
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, (pos, val) in enumerate(U_entries):
        U[pos][idx] = val
    for idx, (pos, val) in enumerate(V_entries):
        V[pos][idx] = val
    for idx, (pos, val) in enumerate(W_entries):
        W[pos][idx] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
