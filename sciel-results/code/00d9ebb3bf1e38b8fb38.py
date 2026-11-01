"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition based on 3 blocks of Strassen (2x2x2)
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices:
    # a: 0,0 -> U[0]
    # b: 0,1 -> U[1]
    # c: 1,0 -> U[2]
    # d: 1,1 -> U[3]
    
    # B indices:
    # B[j, k] -> V[j*p + k]
    # For block b (0,1,2), columns are 2b, 2b+1
    # e_b = B[0, 2b] -> V[0*p + 2b]
    # f_b = B[0, 2b+1] -> V[0*p + 2b+1]
    # g_b = B[1, 2b] -> V[1*p + 2b]
    # h_b = B[1, 2b+1] -> V[1*p + 2b+1]
    
    # C indices:
    # C[i, k] -> W[k*n + i]
    # For block b, columns 2b, 2b+1
    # C[0, 2b] -> W[2b*n + 0]
    # C[1, 2b] -> W[2b*n + 1]
    # C[0, 2b+1] -> W[(2b+1)*n + 0]
    # C[1, 2b+1] -> W[(2b+1)*n + 1]

    for b in range(3):
        base_col = b * 7
        
        # P1: (a+d)(e+h)
        col = base_col + 0
        U[0][col] = 1; U[3][col] = 1
        V[0*p + 2*b][col] = 1; V[1*p + 2*b + 1][col] = 1
        W[2*b*n + 0][col] = 1
        W[(2*b+1)*n + 1][col] = 1
        
        # P2: (c+d)(g+h)
        col = base_col + 1
        U[2][col] = 1; U[3][col] = 1
        V[1*p + 2*b][col] = 1; V[1*p + 2*b + 1][col] = 1
        W[2*b*n + 1][col] = 1
        W[(2*b+1)*n + 1][col] = -1
        
        # P3: a(f-h)
        col = base_col + 2
        U[0][col] = 1
        V[0*p + 2*b + 1][col] = 1; V[1*p + 2*b + 1][col] = -1
        W[(2*b+1)*n + 0][col] = 1
        W[(2*b+1)*n + 1][col] = 1
        
        # P4: d(g-e)
        col = base_col + 3
        U[3][col] = 1
        V[1*p + 2*b][col] = 1; V[0*p + 2*b][col] = -1
        W[2*b*n + 0][col] = 1
        
        # P5: (a+b)e
        col = base_col + 4
        U[0][col] = 1; U[1][col] = 1
        V[0*p + 2*b][col] = 1
        W[2*b*n + 0][col] = -1
        W[(2*b+1)*n + 0][col] = 1
        
        # P6: (c+d)f
        col = base_col + 5
        U[2][col] = 1; U[3][col] = 1
        V[0*p + 2*b + 1][col] = 1
        W[2*b*n + 1][col] = 1
        W[(2*b+1)*n + 1][col] = 1
        
        # P7: (a-b)h
        col = base_col + 6
        U[0][col] = 1; U[1][col] = -1
        V[1*p + 2*b + 1][col] = 1
        W[2*b*n + 0][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
