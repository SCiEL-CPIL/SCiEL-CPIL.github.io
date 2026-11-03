import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_entries = n * m * p  # 60
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][c] = 1
            V[v_idx][c] = 1
            W[w_idx][c] = 1
            c += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10
    # Split B2 into 5 blocks of 2x2
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # W indices for outputs
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][c] += 1
        U[u_22][c] += 1
        V[v_11][c] += 1
        V[v_22][c] += 1
        W[w_00][c] += 1
        W[w_11][c] += 1
        c += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][c] += 1
        U[u_22][c] += 1
        V[v_11][c] += 1
        W[w_10][c] += 1
        W[w_11][c] += -1
        c += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][c] += 1
        V[v_12][c] += 1
        V[v_22][c] += -1
        W[w_01][c] += 1
        W[w_11][c] += 1
        c += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][c] += 1
        V[v_21][c] += 1
        V[v_11][c] += -1
        W[w_00][c] += 1
        W[w_10][c] += 1
        c += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][c] += 1
        U[u_12][c] += 1
        V[v_22][c] += 1
        W[w_00][c] += -1
        W[w_01][c] += 1
        c += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][c] += 1
        U[u_11][c] += -1
        V[v_11][c] += 1
        V[v_12][c] += 1
        W[w_11][c] += 1
        c += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][c] += 1
        U[u_22][c] += -1
        V[v_21][c] += 1
        V[v_22][c] += 1
        W[w_00][c] += 1
        c += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
