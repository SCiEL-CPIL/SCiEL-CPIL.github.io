import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    # Helper to add standard rank-1 terms for a block
    def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[u_idx(i, j)] = 1
                    v[v_idx(j, k)] = 1
                    w[w_idx(k, i)] = 1
                    add_term(u, v, w)

    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen_2x2x2(i0, j0, k0):
        # Indices are (i0, i0+1), (j0, j0+1), (k0, k0+1)
        # M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        # M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1
        w[w_idx(k0, i0+1)] = 1; w[w_idx(k0+1, i0+1)] = -1
        add_term(u, v, w)
        
        # M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1
        v[v_idx(j0, k0+1)] = 1; v[v_idx(j0+1, k0+1)] = -1
        w[w_idx(k0+1, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        # M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0+1, k0)] = 1; v[v_idx(j0, k0)] = -1
        w[w_idx(k0, i0)] = 1; w[w_idx(k0, i0+1)] = 1
        add_term(u, v, w)
        
        # M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1; u[u_idx(i0, j0+1)] = 1
        v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = -1; w[w_idx(k0+1, i0)] = 1
        add_term(u, v, w)
        
        # M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1; u[u_idx(i0, j0)] = -1
        v[v_idx(j0, k0)] = 1; v[v_idx(j0, k0+1)] = 1
        w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        # M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0+1)] = 1; u[u_idx(i0+1, j0+1)] = -1
        v[v_idx(j0+1, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1
        add_term(u, v, w)

    # Split 5 into 2 (S1) and 3 (S2)
    S1 = [0, 1]
    S2 = [2, 3, 4]
    
    # 1. Block S1 x S1 x S1 (2x2x2)
    add_strassen_2x2x2(0, 0, 0)
    
    # 2. Block S2 x S2 x S2 (3x3x3)
    # Use 1+2 split for 3x3x3
    # Local indices for S2: 0,1,2 corresponding to global 2,3,4
    # Split local 3 into 1 (local 0) and 2 (local 1,2)
    # Global: S2_1 = [2], S2_2 = [3,4]
    
    # Sub-block S2_1 x S2_1 x S2_1 (1x1x1)
    add_standard_block(2, 2, 2, 2, 2, 2)
    
    # Sub-block S2_2 x S2_2 x S2_2 (2x2x2)
    add_strassen_2x2x2(3, 3, 3)
    
    # Cross terms within S2xS2xS2
    # S2_1 x S2_1 x S2_2 (1x1x2)
    add_standard_block(2, 2, 2, 2, 3, 4)
    # S2_1 x S2_2 x S2_1 (1x2x1)
    add_standard_block(2, 2, 3, 4, 2, 2)
    # S2_1 x S2_2 x S2_2 (1x2x2)
    add_standard_block(2, 2, 3, 4, 3, 4)
    # S2_2 x S2_1 x S2_1 (2x1x1)
    add_standard_block(3, 4, 2, 2, 2, 2)
    # S2_2 x S2_1 x S2_2 (2x1x2)
    add_standard_block(3, 4, 2, 2, 3, 4)
    # S2_2 x S2_2 x S2_1 (2x2x1)
    add_standard_block(3, 4, 3, 4, 2, 2)
    
    # 3. Cross terms between S1 and S2
    # S1 x S1 x S2 (2x2x3)
    add_standard_block(0, 1, 0, 1, 2, 4)
    # S1 x S2 x S1 (2x3x2)
    add_standard_block(0, 1, 2, 4, 0, 1)
    # S1 x S2 x S2 (2x3x3)
    add_standard_block(0, 1, 2, 4, 2, 4)
    # S2 x S1 x S1 (3x2x2)
    add_standard_block(2, 4, 0, 1, 0, 1)
    # S2 x S1 x S2 (3x2x3)
    add_standard_block(2, 4, 0, 1, 2, 4)
    # S2 x S2 x S1 (3x3x2)
    add_standard_block(2, 4, 2, 4, 0, 1)
    
    rank = len(U)
    # Convert to list of lists if needed, but U,V,W are already lists of lists (vectors)
    # The interface expects {'U', 'V', 'W'} where U is (n*m, rank)
    # Currently U is a list of rows? No, U is a list of column vectors.
    # I need to transpose.
    
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            if U[r][i] != 0:
                U_mat[i][r] = U[r][i]
        for j in range(m*p):
            if V[r][j] != 0:
                V_mat[j][r] = V[r][j]
        for k in range(p*n):
            if W[r][k] != 0:
                W_mat[k][r] = W[r][k]
                
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
