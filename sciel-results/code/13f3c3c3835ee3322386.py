DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank-56 decomposition based on Strassen-style 2x2x2 blocks.
    # The 2x4x8 tensor is decomposed into 8 blocks of 2x2x2, each of rank 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 decomposition (7 terms)
    # For a 2x2 matrix A times 2x2 matrix B giving 2x2 result C
    # A is indexed by (i,j) with i in {0,1}, j in {0,1}
    # B is indexed by (j,k) with j in {0,1}, k in {0,1}
    # C is indexed by (i,k) with i in {0,1}, k in {0,1}
    #
    # The 7 rank-1 terms are:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # Result:
    # C[0,0] = M1 + M4 - M5 - M6
    # C[1,0] = M2 + M6
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M7
    
    # For each rank-1 term r, we store the coefficients for U (A factors), V (B factors), W (C coefficients)
    # U[r] has 4 entries (for i=0,1 and j=0,1)
    # V[r] has 4 entries (for j=0,1 and k=0,1)
    # W[r] has 4 entries (for i=0,1 and k=0,1)
    
    # Term 0: M1
    u0 = [(0,1), (3,1)]       # A[0,0], A[1,1]
    v0 = [(0,1), (3,1)]       # B[0,0], B[1,1]
    w0 = [(0,1), (3,1)]       # C[0,0], C[1,1]
    
    # Term 1: M2
    u1 = [(2,1), (3,1)]       # A[1,0], A[1,1]
    v1 = [(0,1)]              # B[0,0]
    w1 = [(2,1), (3,-1)]      # C[1,0], C[1,1]
    
    # Term 2: M3
    u2 = [(0,1)]              # A[0,0]
    v2 = [(1,1), (3,-1)]      # B[0,1], -B[1,1]
    w2 = [(1,1), (3,1)]       # C[0,1], C[1,1]
    
    # Term 3: M4
    u3 = [(3,1)]              # A[1,1]
    v3 = [(2,1), (0,-1)]      # B[1,0], -B[0,0]
    w3 = [(0,1), (2,1)]       # C[0,0], C[1,0]
    
    # Term 4: M5
    u4 = [(0,1), (1,1)]       # A[0,0], A[0,1]
    v4 = [(3,1)]              # B[1,1]
    w4 = [(0,-1), (1,1)]      # -C[0,0], C[0,1]
    
    # Term 5: M6
    u5 = [(2,1), (0,-1)]      # A[1,0], -A[0,0]
    v5 = [(0,1), (1,1)]       # B[0,0], B[0,1]
    w5 = [(0,-1), (2,1)]      # -C[0,0], C[1,0]
    
    # Term 6: M7
    u6 = [(1,1), (3,-1)]      # A[0,1], -A[1,1]
    v6 = [(2,1), (3,1)]       # B[1,0], B[1,1]
    w6 = [(3,1)]              # C[1,1]
    
    strassen_U = [u0, u1, u2, u3, u4, u5, u6]
    strassen_V = [v0, v1, v2, v3, v4, v5, v6]
    strassen_W = [w0, w1, w2, w3, w4, w5, w6]
    
    # Now we have a 2x4x8 matrix multiplication.
    # We split the 4 columns of A and the 8 columns of B into blocks.
    # A is 2x4: we split into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # B is 4x8: we split rows 0-1 (matching A_left) and rows 2-3 (matching A_right)
    # B rows 0-1: 2x8, split into four 2x2 blocks
    # B rows 2-3: 2x8, split into four 2x2 blocks
    # So we have 8 total 2x2x2 blocks, each of rank 7 = 56 total
    
    def add_block(base_rank, i_off, j_off, k_off):
        """Add a 2x2x2 Strassen block at given offsets.
        i_off: row offset in A (0 or 0, since A has 2 rows)
        j_off: column offset in A (0 or 2)
        k_off: column offset in B (0, 2, 4, or 6)
        """
        for r in range(7):
            current_rank = base_rank + r
            
            # Update U (A factors)
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)  # idx = i*2 + j
                global_i = local_i + i_off
                global_j = local_j + j_off
                U[global_i * 4 + global_j][current_rank] += val
            
            # Update V (B factors)
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)  # idx = j*2 + k
                global_j = local_j + j_off
                global_k = local_k + k_off
                V[global_j * 8 + global_k][current_rank] += val
            
            # Update W (C coefficients)
            # W is indexed by [k*n + i] per the trace convention
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)  # idx = k*2 + i
                global_k = local_k + k_off
                global_i = local_i + i_off
                W[global_k * 2 + global_i][current_rank] += val
    
    # Block assignments:
    # A_left (i:0-1, j:0-1) * B_top (j:0-1, k:0-7)
    add_block(0,  0, 0, 0)   # A_left * B[0:2, 0:2]
    add_block(7,  0, 0, 2)   # A_left * B[0:2, 2:4]
    add_block(14, 0, 0, 4)   # A_left * B[0:2, 4:6]
    add_block(21, 0, 0, 6)   # A_left * B[0:2, 6:8]
    
    # A_right (i:0-1, j:2-3) * B_bottom (j:2-3, k:0-7)
    add_block(28, 0, 2, 0)   # A_right * B[2:4, 0:2]
    add_block(35, 0, 2, 2)   # A_right * B[2:4, 2:4]
    add_block(42, 0, 2, 4)   # A_right * B[2:4, 4:6]
    add_block(49, 0, 2, 6)   # A_right * B[2:4, 6:8]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
