"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 45 decomposition
    # Split m=5 into J1={1,2} and J2={3,4,5}
    # Split p=6 into K1={1,2}, K2={3,4}, K3={5,6}
    # For each K block:
    #   J1 part: 2x2x2 (Strassen, rank 7)
    #   J2 part: 2x3x2 (Strassen 2x2x2 rank 7 + 2x1x2 rank 1 = rank 8)
    # Total rank = 3 * (7 + 8) = 45
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # U indices: (i, j') where i in 0..1, j' in 0..1
    # V indices: (j', k') where j' in 0..1, k' in 0..1
    # W indices: (k', i) where k' in 0..1, i in 0..1
    
    # M1: (a11+a22)(b11+b22)
    U_M1 = [(0,0,1), (1,1,1)]
    V_M1 = [(0,0,1), (1,1,1)]
    W_M1 = [(0,0,1), (1,1,-1), (0,1,1), (1,0,1)] # c11=1, c12=0, c21=0, c22=-1? 
    # Wait, let's use the explicit formulas derived earlier.
    # c11 (k=0, i=0): M1 + M4 - M5 + M7
    # c12 (k=0, i=1): M3 + M5
    # c21 (k=1, i=0): M2 + M4
    # c22 (k=1, i=1): M1 - M2 + M3 + M6
    
    # Let's define the 7 patterns explicitly for U, V, W
    # U[r][i*m + j]
    # V[r][j*p + k]
    # W[r][k*n + i]
    
    # Pattern definitions: list of (index, value) for each matrix
    # Indices are local (0..1)
    
    U_pats = [
        [(0, 1), (3, 1)],       # M1: a11, a22
        [(2, 1), (3, 1)],       # M2: a21, a22
        [(0, 1)],               # M3: a11
        [(3, 1)],               # M4: a22
        [(0, 1), (1, 1)],       # M5: a11, a12
        [(2, 1), (0, -1)],      # M6: a21, -a11
        [(1, 1), (3, -1)]       # M7: a12, -a22
    ]
    
    V_pats = [
        [(0, 1), (3, 1)],       # M1: b11, b22
        [(0, 1)],               # M2: b11
        [(1, 1), (3, -1)],      # M3: b12, -b22
        [(2, 1), (0, -1)],      # M4: b21, -b11
        [(3, 1)],               # M5: b22
        [(0, 1), (1, 1)],       # M6: b11, b12
        [(2, 1), (3, 1)]        # M7: b21, b22
    ]
    
    # W patterns: W[k*n + i]
    # k=0, i=0 -> idx 0
    # k=0, i=1 -> idx 1
    # k=1, i=0 -> idx 2
    # k=1, i=1 -> idx 3
    
    W_pats = [
        [(0, 1), (2, 1), (1, -1), (3, 1)],  # M1: c11=1, c21=1, c12=-1, c22=1 ? 
        # Let's re-verify Strassen output.
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # So for M1: c11+=1, c22+=1.
        # Indices: c11 is k=0,i=0 (0). c22 is k=1,i=1 (3).
        [(0, 1), (3, 1)],
        # M2: c21+=1, c22+=-1.
        # c21 is k=1,i=0 (2). c22 is k=1,i=1 (3).
        [(2, 1), (3, -1)],
        # M3: c12+=1, c22+=1.
        # c12 is k=0,i=1 (1). c22 is k=1,i=1 (3).
        [(1, 1), (3, 1)],
        # M4: c11+=1, c21+=1.
        # c11 is 0. c21 is 2.
        [(0, 1), (2, 1)],
        # M5: c11+=-1, c12+=1.
        # c11 is 0. c12 is 1.
        [(0, -1), (1, 1)],
        # M6: c22+=1.
        [(3, 1)],
        # M7: c11+=1.
        [(0, 1)]
    ]
    
    # Helper to add a block
    def add_strassen_block(j_offset, k_offset, start_col):
        for m_idx in range(7):
            c = start_col + m_idx
            # U
            for (local_idx, val) in U_pats[m_idx]:
                # local_idx is i*2 + j
                i = local_idx // 2
                j = local_idx % 2
                global_u_idx = i * m + (j + j_offset)
                U[global_u_idx][c] += val
            # V
            for (local_idx, val) in V_pats[m_idx]:
                # local_idx is j*2 + k
                j = local_idx // 2
                k = local_idx % 2
                global_v_idx = (j + j_offset) * p + (k + k_offset)
                V[global_v_idx][c] += val
            # W
            for (local_idx, val) in W_pats[m_idx]:
                # local_idx is k*2 + i
                k = local_idx // 2
                i = local_idx % 2
                global_w_idx = (k + k_offset) * n + i
                W[global_w_idx][c] += val
        return start_col + 7

    def add_rank1_block(j_idx, k_offset, start_col):
        c = start_col
        # U: i in 0..1, j = j_idx
        for i in range(n):
            U[i * m + j_idx][c] += 1
        # V: j = j_idx, k in 0..1
        for k in range(2):
            V[j_idx * p + (k + k_offset)][c] += 1
        # W: k in 0..1, i in 0..1
        for k in range(2):
            for i in range(n):
                W[(k + k_offset) * n + i][c] += 1
        return c + 1

    k_offsets = [0, 2, 4]
    
    for k_off in k_offsets:
        # Part 1: J1 = {0, 1} (global 0, 1)
        col = add_strassen_block(0, k_off, col)
        
        # Part 2: J2 = {2, 3, 4} (global 2, 3, 4)
        # Sub-part 2a: j in {2, 3} (local 0, 1 relative to offset 2)
        col = add_strassen_block(2, k_off, col)
        
        # Sub-part 2b: j = 4
        col = add_rank1_block(4, k_off, col)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
