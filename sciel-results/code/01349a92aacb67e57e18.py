import time
import sys

# EVOLVE-BLOCK-START

def get_strassen_2x2_2x2_UVW():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7
    Indices for 2x2 matrix [r, c] are r*2+c.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen steps
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def construct():
    # Dimensions
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x3 Schoolbook (Rank 48)
    
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Strassen 4x4x4 ---
    base_U, base_V, base_W = get_strassen_2x2_2x2_UVW()
    
    # Block definitions for 4x4 matrices
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00, B01, B10, B11 similarly
    # C blocks: C00, C01, C10, C11 similarly
    
    # Strassen steps mapping
    # Each step s in 0..6 involves specific A and B blocks with coefficients.
    # We define the linear combinations for A and B for each step.
    
    # A_combinations[s] is a list of (block_id, coeff)
    # B_combinations[s] is a list of (block_id, coeff)
    
    # Block IDs: 0=A00, 1=A01, 2=A10, 3=A11
    # Global indices:
    # A_idx(block, r, c) = (block//2)*2 + r, (block%2)*2 + c -> global (row, col) -> row*m + col
    # B_idx(block, r, c) = (block//2)*2 + r, (block%2)*2 + c -> global (row, col) -> row*p + col (p=4 for this part)
    # C_idx(block, r, c) = (block//2)*2 + r, (block%2)*2 + c -> global (row, col) -> col*n + row (W index k*n+i)
    
    def a_global(block, r, c):
        row = (block // 2) * 2 + r
        col = (block % 2) * 2 + c
        return row * m + col
        
    def b_global(block, r, c):
        row = (block // 2) * 2 + r
        col = (block % 2) * 2 + c
        # Note: p=4 for the first 4 columns of B
        return row * 4 + col 
        
    def c_global(block, r, c):
        row = (block // 2) * 2 + r
        col = (block % 2) * 2 + c
        # W index: k*n + i. k is col index (0..3), i is row index (0..3)
        return col * n + row

    # Strassen definitions
    # s=0: M1 = (A00+A11)(B00+B11)
    # s=1: M2 = (A10+A11)B00
    # s=2: M3 = A00(B01-B11)
    # s=3: M4 = A11(B10-B00)
    # s=4: M5 = (A00+A01)B11
    # s=5: M6 = (A10-A00)(B00+B01)
    # s=6: M7 = (A01-A11)(B10+B11)
    
    a_blocks = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, 1)],   # M2
        [(0, 1)],           # M3
        [(3, 1)],           # M4
        [(0, 1), (1, 1)],   # M5
        [(2, 1), (0, -1)],  # M6
        [(1, 1), (3, -1)]   # M7
    ]
    
    b_blocks = [
        [(0, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(1, 1), (3, -1)],  # M3
        [(2, 1), (0, -1)],  # M4
        [(3, 1)],           # M5
        [(0, 1), (1, 1)],   # M6
        [(2, 1), (3, 1)]    # M7
    ]
    
    # C combination for each M_i
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_contrib[s][block] = coeff
    # Block 0: C00, Block 1: C01, Block 2: C10, Block 3: C11
    
    c_contrib = [
        {0: 1, 3: 1},       # M1 contributes to C00, C11? No.
        # Let's map carefully.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # s=0 (M1): C00(+1), C11(+1)
        # s=1 (M2): C10(+1), C11(-1)
        # s=2 (M3): C01(+1), C11(+1)
        # s=3 (M4): C00(+1), C10(+1)
        # s=4 (M5): C00(-1), C01(+1)
        # s=5 (M6): C11(+1)
        # s=6 (M7): C00(-1)
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: -1}             # M7
    ]
    
    # Generate the 49 terms
    term_idx = 0
    for s in range(7):
        # For each of the 7 base Strassen terms
        for base_t in range(7):
            # U part
            for r in range(2):
                for c in range(2):
                    local_idx = r * 2 + c
                    coeff_u = base_U[local_idx][base_t]
                    if coeff_u == 0:
                        continue
                    for ab_id, ab_coeff in a_blocks[s]:
                        g_idx = a_global(ab_id, r, c)
                        U[g_idx][term_idx] += coeff_u * ab_coeff
                        
            # V part
            for r in range(2):
                for c in range(2):
                    local_idx = r * 2 + c
                    coeff_v = base_V[local_idx][base_t]
                    if coeff_v == 0:
                        continue
                    for bb_id, bb_coeff in b_blocks[s]:
                        g_idx = b_global(bb_id, r, c)
                        V[g_idx][term_idx] += coeff_v * bb_coeff
                        
            # W part
            for r in range(2):
                for c in range(2):
                    local_idx = r * 2 + c
                    coeff_w = base_W[local_idx][base_t]
                    if coeff_w == 0:
                        continue
                    for cb_id, cb_coeff in c_contrib[s].items():
                        g_idx = c_global(cb_id, r, c)
                        W[g_idx][term_idx] += coeff_w * cb_coeff
                        
            term_idx += 1
            
    # --- Schoolbook 4x4x3 ---
    # Columns 4, 5, 6 of B
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][term_idx] = 1
                V[v_idx][term_idx] = 1
                W[w_idx][term_idx] = 1
                term_idx += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
