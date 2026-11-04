import time
import sys

# EVOLVE-BLOCK-START

def get_strassen_2x2_2x2_UVW():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen steps
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    #           0=B00, 1=B01, 2=B10, 3=B11
    #           0=C00, 1=C01, 2=C10, 3=C11
    
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
    
    # We will build the rank-1 terms.
    terms = []
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # We use the 7 Strassen steps for 2x2 blocks.
    # Each step involves linear combinations of A and B blocks.
    # The product is a 2x2 matrix.
    # We decompose the 2x2x2 multiplication into 7 rank-1 terms.
    
    base_U, base_V, base_W = get_strassen_2x2_2x2_UVW()
    
    # Define the Strassen steps for 4x4x4
    # a_blocks: list of (block_row, block_col, coeff)
    # b_blocks: list of (block_row, block_col, coeff)
    # c_blocks: list of (block_row, block_col, coeff)
    
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]
        },
        # M2 = (A10+A11)B00
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1)]
        },
        # M3 = A00(B01-B11)
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1)]
        },
        # M4 = A11(B10-B00)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]
        },
        # M5 = (A00+A01)B11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,1,1), (1,1,-1)]
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,-1)]
        }
    ]
    
    for step in strassen_steps:
        a_blocks = step['a']
        b_blocks = step['b']
        c_blocks = step['c']
        
        # For each of the 7 rank-1 terms of the base 2x2x2 Strassen
        for t in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 28
            w_vec = [0] * 28
            
            # Map A
            for br, bc, coeff_a in a_blocks:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        coeff_base = base_U[local_idx][t]
                        if coeff_base == 0: continue
                        global_r = br * 2 + lr
                        global_c = bc * 2 + lc
                        global_idx = global_r * 4 + global_c
                        u_vec[global_idx] += coeff_a * coeff_base
                        
            # Map B (left part, cols 0..3)
            for br, bc, coeff_b in b_blocks:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        coeff_base = base_V[local_idx][t]
                        if coeff_base == 0: continue
                        global_r = br * 2 + lr
                        global_c = bc * 2 + lc
                        global_idx = global_r * 7 + global_c
                        v_vec[global_idx] += coeff_b * coeff_base
                        
            # Map C (left part, cols 0..3)
            for br, bc, coeff_c in c_blocks:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        coeff_base = base_W[local_idx][t]
                        if coeff_base == 0: continue
                        global_r = br * 2 + lr
                        global_c = bc * 2 + lc
                        # W index: k*n + i -> col * 4 + row
                        global_idx = global_c * 4 + global_r
                        w_vec[global_idx] += coeff_c * coeff_base
                        
            terms.append((u_vec, v_vec, w_vec))
            
    # Part 2: 4x4x3 Schoolbook (Rank 48)
    # B_right is cols 4..6.
    # C_right is cols 4..6.
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_vec = [0] * 16
                v_vec = [0] * 28
                w_vec = [0] * 28
                
                u_vec[i * 4 + j] = 1
                v_vec[j * 7 + k] = 1
                w_vec[k * 4 + i] = 1
                
                terms.append((u_vec, v_vec, w_vec))
                
    total_rank = len(terms)
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(16):
            U[i][r] = u[i]
        for j in range(28):
            V[j][r] = v[j]
        for k in range(28):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
