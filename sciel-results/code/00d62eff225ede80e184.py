"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition for 2x4 * 4x4 -> 2x4 matrix multiplication.
    # A is (2,4), B is (4,4), C = A*B is (2,4).
    # Indices: A[i,j] i in 0..1, j in 0..3
    #           B[j,k] j in 0..3, k in 0..3
    #           C[i,k] i in 0..1, k in 0..3
    #
    # We use a block decomposition. Split A into two 2x2 blocks: A0 = A[:, 0:2], A1 = A[:, 2:4].
    # Split B into two 2x2 blocks per column: B0 = B[:, 0:2], B1 = B[:, 2:4].
    # Then C = A0*B0 + A1*B1.
    #
    # For each 2x2 * 2x2 -> 2x2 multiplication, we use Strassen's algorithm (rank 7).
    # There are 4 such block products (for each pair of column-blocks of B and the two row-blocks of A? No.)
    # Wait: A is 2x4, B is 4x4.
    # A = [A_L | A_R] where A_L, A_R are 2x2.
    # B = [[B_TL, B_TR], [B_BL, B_BR]] where each is 2x2.
    # C = A * B = A_L * [B_TL; B_BL] + A_R * [B_TR; B_BR].
    # Let's define sub-matrices:
    # B_0 = [[B_TL, B_TR]]? No, B is 4x4.
    # Rows of B correspond to columns of A.
    # A_L multiplies the first 2 columns of B? No, A_L is the first 2 columns of A.
    # C[i, k] = sum_{j=0}^3 A[i, j] B[j, k].
    # C[i, k] = sum_{j=0}^1 A[i, j] B[j, k] + sum_{j=2}^3 A[i, j] B[j, k].
    #
    # Let's group by j-blocks.
    # Block 1: j in {0, 1}. A_block1 = A[:, 0:2] (2x2), B_block1 = B[0:2, :] (2x4).
    # Block 2: j in {2, 3}. A_block2 = A[:, 2:4] (2x2), B_block2 = B[2:4, :] (2x4).
    # C = A_block1 * B_block1 + A_block2 * B_block2.
    #
    # Each term is (2x2) * (2x4) -> (2x4).
    # We can split the (2x4) B into two (2x2) blocks: B_left and B_right.
    # So B_block1 = [B1L, B1R] where B1L = B[0:2, 0:2], B1R = B[0:2, 2:4].
    # B_block2 = [B2L, B2R] where B2L = B[2:4, 0:2], B2R = B[2:4, 2:4].
    #
    # C = A1 * [B1L, B1R] + A2 * [B2L, B2R]
    #   = [A1*B1L, A1*B1R] + [A2*B2L, A2*B2R]
    #   = [A1*B1L + A2*B2L, A1*B1R + A2*B2R]
    #
    # Let C_L = C[:, 0:2] and C_R = C[:, 2:4].
    # C_L = A1*B1L + A2*B2L.
    # C_R = A1*B1R + A2*B2R.
    #
    # Each of these is a sum of two (2x2)*(2x2) products.
    # Total 4 (2x2)*(2x2) products.
    # Each (2x2)*(2x2) can be done with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    #
    # Let's construct the U, V, W matrices for rank 28.
    # U shape: (2*4, 28) = (8, 28)
    # V shape: (4*4, 28) = (16, 28)
    # W shape: (4*2, 28) = (8, 28)  (Wait, output is 2x4, so p*n = 4*2 = 8. W[k*n+i, r])
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen for 2x2 * 2x2 -> 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[p, q], [r, s]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(g-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # p = M1 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M3 + M4 - M6
    #
    # We will define a helper to add one Strassen block.
    # The block corresponds to multiplying a 2x2 submatrix of A by a 2x2 submatrix of B,
    # contributing to a 2x2 submatrix of C.
    #
    # A_sub indices: (i_off + di, j_off + dj) for di, dj in {0, 1}
    # B_sub indices: (j_off + dj, k_off + dk) for dj, dk in {0, 1}
    # C_sub indices: (i_off + di, k_off + dk) for di, dk in {0, 1}
    #
    # U row index: i*m + j = (i_off + di)*4 + (j_off + dj)
    # V row index: j*p + k = (j_off + dj)*4 + (k_off + dk)
    # W row index: k*n + i = (k_off + dk)*2 + (i_off + di)

    def add_strassen_block(i_off, j_off, k_off, col_start):
        # col_start: starting column index in U, V, W
        # Returns the next column index
        col = col_start
        
        # M1 = (a+d)(e+h)
        # a = A[i_off, j_off], d = A[i_off+1, j_off+1]
        # e = B[j_off, k_off], h = B[j_off+1, k_off+1]
        u_idx_a = i_off * 4 + j_off
        u_idx_d = (i_off + 1) * 4 + (j_off + 1)
        v_idx_e = j_off * 4 + k_off
        v_idx_h = (j_off + 1) * 4 + (k_off + 1)
        
        U[u_idx_a][col] = 1
        U[u_idx_d][col] = 1
        V[v_idx_e][col] = 1
        V[v_idx_h][col] = 1
        # M1 contributes to p (C[i_off, k_off]) and s (C[i_off+1, k_off+1])
        # p: C[i_off, k_off] -> W[k_off*2 + i_off] += 1
        # s: C[i_off+1, k_off+1] -> W[(k_off+1)*2 + (i_off+1)] += 1
        W[k_off * 2 + i_off][col] = 1
        W[(k_off + 1) * 2 + (i_off + 1)][col] = 1
        col += 1

        # M2 = (c+d)e
        # c = A[i_off+1, j_off], d = A[i_off+1, j_off+1]
        # e = B[j_off, k_off]
        u_idx_c = (i_off + 1) * 4 + j_off
        v_idx_e2 = j_off * 4 + k_off
        
        U[u_idx_c][col] = 1
        U[u_idx_d][col] = 1
        V[v_idx_e2][col] = 1
        # M2 contributes to r (C[i_off+1, k_off])
        # r: C[i_off+1, k_off] -> W[k_off*2 + (i_off+1)] += 1
        W[k_off * 2 + (i_off + 1)][col] = 1
        col += 1

        # M3 = a(f-g)
        # a = A[i_off, j_off]
        # f = B[j_off, k_off+1], g = B[j_off+1, k_off]
        u_idx_a2 = i_off * 4 + j_off
        v_idx_f = j_off * 4 + (k_off + 1)
        v_idx_g = (j_off + 1) * 4 + k_off
        
        U[u_idx_a2][col] = 1
        V[v_idx_f][col] = 1
        V[v_idx_g][col] = -1
        # M3 contributes to q (C[i_off, k_off+1]) and s (C[i_off+1, k_off+1]) with -1
        # q: C[i_off, k_off+1] -> W[(k_off+1)*2 + i_off] += 1
        # s: C[i_off+1, k_off+1] -> W[(k_off+1)*2 + (i_off+1)] += -1
        W[(k_off + 1) * 2 + i_off][col] = 1
        W[(k_off + 1) * 2 + (i_off + 1)][col] = -1
        col += 1

        # M4 = d(g-h)
        # d = A[i_off+1, j_off+1]
        # g = B[j_off+1, k_off], h = B[j_off+1, k_off+1]
        u_idx_d2 = (i_off + 1) * 4 + (j_off + 1)
        v_idx_g2 = (j_off + 1) * 4 + k_off
        v_idx_h2 = (j_off + 1) * 4 + (k_off + 1)
        
        U[u_idx_d2][col] = 1
        V[v_idx_g2][col] = 1
        V[v_idx_h2][col] = -1
        # M4 contributes to r (C[i_off+1, k_off]) and s (C[i_off+1, k_off+1])
        # r: C[i_off+1, k_off] -> W[k_off*2 + (i_off+1)] += 1
        # s: C[i_off+1, k_off+1] -> W[(k_off+1)*2 + (i_off+1)] += 1
        W[k_off * 2 + (i_off + 1)][col] = 1
        W[(k_off + 1) * 2 + (i_off + 1)][col] = 1
        col += 1

        # M5 = (a+b)h
        # a = A[i_off, j_off], b = A[i_off, j_off+1]
        # h = B[j_off+1, k_off+1]
        u_idx_b = i_off * 4 + (j_off + 1)
        v_idx_h3 = (j_off + 1) * 4 + (k_off + 1)
        
        U[u_idx_a2][col] = 1
        U[u_idx_b][col] = 1
        V[v_idx_h3][col] = 1
        # M5 contributes to p (C[i_off, k_off]) with -1 and q (C[i_off, k_off+1])
        # p: C[i_off, k_off] -> W[k_off*2 + i_off] += -1
        # q: C[i_off, k_off+1] -> W[(k_off+1)*2 + i_off] += 1
        W[k_off * 2 + i_off][col] = -1
        W[(k_off + 1) * 2 + i_off][col] = 1
        col += 1

        # M6 = (c-a)(e+f)
        # c = A[i_off+1, j_off], a = A[i_off, j_off]
        # e = B[j_off, k_off], f = B[j_off, k_off+1]
        u_idx_c2 = (i_off + 1) * 4 + j_off
        v_idx_e3 = j_off * 4 + k_off
        v_idx_f2 = j_off * 4 + (k_off + 1)
        
        U[u_idx_c2][col] = 1
        U[u_idx_a2][col] = -1
        V[v_idx_e3][col] = 1
        V[v_idx_f2][col] = 1
        # M6 contributes to s (C[i_off+1, k_off+1]) with -1
        # s: C[i_off+1, k_off+1] -> W[(k_off+1)*2 + (i_off+1)] += -1
        W[(k_off + 1) * 2 + (i_off + 1)][col] = -1
        col += 1

        # M7 = (b-d)(g+h)
        # b = A[i_off, j_off+1], d = A[i_off+1, j_off+1]
        # g = B[j_off+1, k_off], h = B[j_off+1, k_off+1]
        u_idx_b2 = i_off * 4 + (j_off + 1)
        v_idx_g3 = (j_off + 1) * 4 + k_off
        v_idx_h4 = (j_off + 1) * 4 + (k_off + 1)
        
        U[u_idx_b2][col] = 1
        U[u_idx_d2][col] = -1
        V[v_idx_g3][col] = 1
        V[v_idx_h4][col] = 1
        # M7 contributes to p (C[i_off, k_off])
        # p: C[i_off, k_off] -> W[k_off*2 + i_off] += 1
        W[k_off * 2 + i_off][col] = 1
        col += 1

        return col

    # Block 1: A1 * B1L
    # i_off = 0, j_off = 0, k_off = 0
    col = add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A2 * B2L
    # i_off = 0, j_off = 2, k_off = 0
    col = add_strassen_block(0, 2, 0, col)
    
    # Block 3: A1 * B1R
    # i_off = 0, j_off = 0, k_off = 2
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 4: A2 * B2R
    # i_off = 0, j_off = 2, k_off = 2
    col = add_strassen_block(0, 2, 2, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
