"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    col = 0
    
    # Strassen 2x2x2 template
    # Y = A_block (2x2), Z = B_block (2x2)
    # y00, y01, y10, y11 are (i, j) global indices
    # z00, z01, z10, z11 are (j, k) global indices
    # w00, w01, w10, w11 are (k, i) global indices for C_block
    
    def strassen_block(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        col += 1

    # Schoolbook 2x2x1 template (Rank 4)
    # Y = A_block (2x2), Z = B_block (2x1)
    # y00, y01, y10, y11 are (i, j) global indices
    # z0, z1 are (j, k) global indices (k is fixed)
    # w0, w1 are (k, i) global indices for C_block (k is fixed)
    
    def schoolbook_2x2x1(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z0, z1 = z_indices
        w0, w1 = w_indices
        
        # M1 = y00 * z0
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z0[0], z0[1])][col] += 1
        W[w_idx(w0[0], w0[1])][col] += 1
        col += 1
        
        # M2 = y01 * z1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z1[0], z1[1])][col] += 1
        W[w_idx(w0[0], w0[1])][col] += 1
        col += 1
        
        # M3 = y10 * z0
        U[u_idx(y10[0], y10[1])][col] += 1
        V[v_idx(z0[0], z0[1])][col] += 1
        W[w_idx(w1[0], w1[1])][col] += 1
        col += 1
        
        # M4 = y11 * z1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z1[0], z1[1])][col] += 1
        W[w_idx(w1[0], w1[1])][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1) * B_1 (rows 0,1, cols 0,1) -> C (cols 0,1)
    # A_L: (0,0), (0,1), (1,0), (1,1)
    # B_1: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1) -> w_idx(k, i)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)] # w00=(k=0,i=0), w01=(k=1,i=0), w10=(k=0,i=1), w11=(k=1,i=1)
    # Note: w_indices order in strassen_block is w00, w01, w10, w11 corresponding to C00, C01, C10, C11
    # C00 is i=0, k=0. C01 is i=0, k=1. C10 is i=1, k=0. C11 is i=1, k=1.
    # w_idx(k, i).
    # C00 -> k=0, i=0.
    # C01 -> k=1, i=0.
    # C10 -> k=0, i=1.
    # C11 -> k=1, i=1.
    strassen_block(y1, z1, [(0,0), (1,0), (0,1), (1,1)])
    
    # Block 2: A_R (cols 2,3) * B_2 (rows 2,3, cols 0,1) -> C (cols 0,1)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_1 (rows 0,1, cols 2,3) -> C (cols 2,3)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    # C cols 2,3.
    # C00 (local) -> global i=0, k=2.
    # C01 (local) -> global i=0, k=3.
    # C10 (local) -> global i=1, k=2.
    # C11 (local) -> global i=1, k=3.
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_2 (rows 2,3, cols 2,3) -> C (cols 2,3)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_1 (rows 0,1, col 4) -> C (col 4)
    y5 = [(0,0), (0,1), (1,0), (1,1)]
    z5 = [(0,4), (1,4)]
    # C col 4.
    # C0 (local) -> global i=0, k=4.
    # C1 (local) -> global i=1, k=4.
    w5 = [(4,0), (4,1)]
    schoolbook_2x2x1(y5, z5, w5)
    
    # Block 6: A_R (cols 2,3) * B_2 (rows 2,3, col 4) -> C (col 4)
    y6 = [(0,2), (0,3), (1,2), (1,3)]
    z6 = [(2,4), (3,4)]
    w6 = [(4,0), (4,1)]
    schoolbook_2x2x1(y6, z6, w6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
