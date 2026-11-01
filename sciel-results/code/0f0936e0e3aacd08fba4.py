"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of M_{2,2,8} as a direct sum of 4 copies of M_{2,2,2}.
    # Each M_{2,2,2} is decomposed using Strassen's algorithm (rank 7).
    # Total rank = 4 * 7 = 28.
    
    n, m, p = DIMENSIONS
    # p = 8, so we have 4 independent 2x2 blocks along the p dimension.
    # Each block has p_block = 2.
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Dimensions of U, V, W
    U_dim = n * m  # 2 * 2 = 4
    V_dim = m * p  # 2 * 8 = 16
    W_dim = p * n  # 8 * 2 = 16
    
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    # Strassen's 7 products for a 2x2 matrix multiplication:
    # For C = A * B where A, B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C = [[P1+P4-P5+P7, P3+P5],
    #      [P2+P4,         P1-P2+P3+P6]]
    #
    # Indices for A (input to U): A[i, j] -> index i*m + j
    #   a = A[0,0] -> index 0
    #   b = A[0,1] -> index 1
    #   c = A[1,0] -> index 2
    #   d = A[1,1] -> index 3
    #
    # Indices for B (input to V): B[j, k] -> index j*p + k
    #   For a specific block q, k_global = 2*q + k_local
    #   e = B[0, 2q] -> index 0*8 + 2q = 2q
    #   f = B[0, 2q+1] -> index 0*8 + 2q+1 = 2q+1
    #   g = B[1, 2q] -> index 1*8 + 2q = 8+2q
    #   h = B[1, 2q+1] -> index 1*8 + 2q+1 = 8+2q+1
    #
    # Indices for C (output W): C[k, i] -> index k*n + i
    #   For a specific block q, k_global = 2*q + k_local
    #   C[2q, 0] -> index 2q*2 + 0 = 4q
    #   C[2q, 1] -> index 2q*2 + 1 = 4q+1
    #   C[2q+1, 0] -> index (2q+1)*2 + 0 = 4q+2
    #   C[2q+1, 1] -> index (2q+1)*2 + 1 = 4q+3
    #
    # Let's define the 7 Strassen factors for a single block:
    # Each factor r corresponds to one column in U, V, W.
    
    # For each block q, we add 7 columns to U, V, W.
    for q in range(num_blocks):
        base_col = q * 7
        e_idx = 2 * q          # B[0, 2q]
        f_idx = 2 * q + 1      # B[0, 2q+1]
        g_idx = 8 + 2 * q      # B[1, 2q]
        h_idx = 8 + 2 * q + 1  # B[1, 2q+1]
        
        c00_idx = 4 * q        # C[2q, 0]
        c01_idx = 4 * q + 1    # C[2q, 1]
        c10_idx = 4 * q + 2    # C[2q+1, 0]
        c11_idx = 4 * q + 3    # C[2q+1, 1]
        
        # Factor 1: P1 = (a+d)(e+h)
        # U: a=1, d=1 -> U[0][col]=1, U[3][col]=1
        # V: e=1, h=1 -> V[e_idx][col]=1, V[h_idx][col]=1
        # W: contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        # C[0,0] += P1, C[0,1] -= P1? No, let's check the formula.
        # C = [[P1+P4-P5+P7, P3+P5],
        #      [P2+P4,         P1-P2+P3+P6]]
        # So:
        # C[0,0] += P1 -> W[c00_idx][col] += 1
        # C[0,1] += 0  (from P1)
        # C[1,0] += 0  (from P1)
        # C[1,1] += P1 -> W[c11_idx][col] += 1
        col = base_col + 0
        U[0][col] += 1
        U[3][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        
        # Factor 2: P2 = (c+d)e
        # U: c=1, d=1 -> U[2][col]=1, U[3][col]=1
        # V: e=1 -> V[e_idx][col]=1
        # W: C[1,0] += P2, C[1,1] -= P2
        col = base_col + 1
        U[2][col] += 1
        U[3][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        
        # Factor 3: P3 = a(f-h)
        # U: a=1 -> U[0][col]=1
        # V: f=1, h=-1 -> V[f_idx][col]=1, V[h_idx][col]=-1
        # W: C[0,1] += P3, C[1,1] += P3
        col = base_col + 2
        U[0][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        
        # Factor 4: P4 = d(g-e)
        # U: d=1 -> U[3][col]=1
        # V: g=1, e=-1 -> V[g_idx][col]=1, V[e_idx][col]=-1
        # W: C[0,0] += P4, C[1,0] += P4
        col = base_col + 3
        U[3][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        
        # Factor 5: P5 = (a+b)h
        # U: a=1, b=1 -> U[0][col]=1, U[1][col]=1
        # V: h=1 -> V[h_idx][col]=1
        # W: C[0,0] -= P5, C[0,1] += P5
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        
        # Factor 6: P6 = (c-a)(e+f)
        # U: c=1, a=-1 -> U[2][col]=1, U[0][col]=-1
        # V: e=1, f=1 -> V[e_idx][col]=1, V[f_idx][col]=1
        # W: C[1,1] += P6
        col = base_col + 5
        U[2][col] += 1
        U[0][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        
        # Factor 7: P7 = (b-d)(g+h)
        # U: b=1, d=-1 -> U[1][col]=1, U[3][col]=-1
        # V: g=1, h=1 -> V[g_idx][col]=1, V[h_idx][col]=1
        # W: C[0,0] += P7
        col = base_col + 6
        U[1][col] += 1
        U[3][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
