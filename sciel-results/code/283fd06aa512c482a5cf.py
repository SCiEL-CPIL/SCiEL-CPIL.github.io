import time
import sys

# EVOLVE-BLOCK-START

def get_strassen_2x2_2x2_UVW():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7
    Indices for 2x2 matrices:
    A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    C: 00, 01, 10, 11 -> 0, 1, 2, 3
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11) -> C01 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    # We split B (4x7) into B1 (4x4) and B2 (4x3).
    # C = A * B = A * B1 + A * B2
    # C1 = A * B1 (4x4) computed via Strassen (Rank 49)
    # C2 = A * B2 (4x3) computed via Schoolbook (Rank 48)
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    
    # Get base 2x2 Strassen UVW
    base_U, base_V, base_W = get_strassen_2x2_2x2_UVW()
    
    # Helper to get global index for A, B1, C1 elements
    # A is 4x4. Index i*m + j
    # B1 is 4x4 (cols 0-3 of B). Index j*p + k (where k is 0..3)
    # C1 is 4x4 (cols 0-3 of C). Index k*n + i (where k is 0..3)
    
    def a_idx(global_r, global_c):
        return global_r * m + global_c
        
    def b1_idx(global_r, global_c):
        return global_r * p + global_c
        
    def c1_idx(global_r, global_c):
        # W index is k*n + i, where i is row, k is col.
        # In C1, row is global_r, col is global_c.
        return global_c * n + global_r

    # Define the 7 outer Strassen steps for 4x4x4
    # Each step involves linear combinations of 2x2 blocks of A and B1.
    # A blocks: A00 (r0-1, c0-1), A01 (r0-1, c2-3), A10 (r2-3, c0-1), A11 (r2-3, c2-3)
    # B1 blocks: B00 (r0-1, c0-1), B01 (r0-1, c2-3), B10 (r2-3, c0-1), B11 (r2-3, c2-3)
    
    # The 7 products M1..M7 are 2x2 matrices.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C blocks reconstruction:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # However, we are constructing rank-1 tensors directly.
    # Each outer step s (0..6) corresponds to one of the M_s.
    # Each M_s is a product of two 2x2 matrices X and Y.
    # X is formed from A blocks, Y from B1 blocks.
    # We use the base 2x2 Strassen decomposition (7 terms) to decompose X*Y.
    # For each outer step s and each inner base term b (0..6), we have a global rank-1 term.
    
    # The coefficient for a specific global A element in U is the sum of contributions from the blocks it belongs to.
    # Similarly for V and W.
    
    # Let's iterate over the 7 outer steps
    outer_steps = [
        # (a_blocks, b_blocks)
        # a_blocks: dict { (br, bc): coeff }
        # b_blocks: dict { (br, bc): coeff }
        {
            'a': { (0,0): 1, (1,1): 1 },
            'b': { (0,0): 1, (1,1): 1 }
        },
        {
            'a': { (1,0): 1, (1,1): 1 },
            'b': { (0,0): 1 }
        },
        {
            'a': { (0,0): 1 },
            'b': { (0,1): 1, (1,1): -1 }
        },
        {
            'a': { (1,1): 1 },
            'b': { (1,0): 1, (0,0): -1 }
        },
        {
            'a': { (0,0): 1, (0,1): 1 },
            'b': { (1,1): 1 }
        },
        {
            'a': { (1,0): 1, (0,0): -1 },
            'b': { (0,0): 1, (0,1): 1 }
        },
        {
            'a': { (0,1): 1, (1,1): -1 },
            'b': { (1,0): 1, (1,1): 1 }
        }
    ]
    
    # We need to be careful with the W matrix.
    # The base Strassen W gives the contribution to the 2x2 result matrix M.
    # But M is added to specific blocks of C with specific signs.
    # Instead of pre-computing M and then adding to C, we can incorporate the signs into the W coefficients of the rank-1 tensors.
    
    # Let's define the contribution of each M_s to the C blocks.
    # C00 gets: M1 (+1), M4 (+1), M5 (-1), M7 (+1) -- Wait, M7 is -M7 in C11, +M7 in C00?
    # Let's re-verify Strassen 4x4 formulas.
    # Standard Strassen for 4x4:
    # M1 = (A11+A22)(B11+B22)  -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11        -> C21 += M2
    # M3 = A11(B12-B22)        -> C12 += M3
    # M4 = A22(B21-B11)        -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22        -> C12 += M5, C22 -= M5
    # M6 = (A21-A11)(B11+B12)  -> C22 += M6
    # M7 = (A12-A22)(B21+B22)  -> C11 += M7, C22 -= M7
    
    # Note: My block indices are 0-based (0,0), (0,1), (1,0), (1,1).
    # A11 in 1-based is A00 in 0-based.
    # A22 in 1-based is A11 in 0-based.
    # B11 in 1-based is B00 in 0-based.
    # B22 in 1-based is B11 in 0-based.
    
    # Let's map 1-based to 0-based:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B01-B11)       -> C01 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    
    # So the coefficients for C blocks from each M_s are:
    # M1: C00:1, C11:1
    # M2: C10:1
    # M3: C01:1
    # M4: C00:1, C10:1
    # M5: C01:1, C11:-1
    # M6: C11:1
    # M7: C00:1, C11:-1
    
    # We will iterate over outer steps s (0..6) and inner base terms b (0..6).
    # For each pair (s, b), we construct a global rank-1 tensor.
    
    for s_idx, step in enumerate(outer_steps):
        a_blocks = step['a']
        b_blocks = step['b']
        
        # Determine C block coefficients for this M_s
        c_coeff_blocks = {}
        if s_idx == 0: # M1
            c_coeff_blocks = { (0,0): 1, (1,1): 1 }
        elif s_idx == 1: # M2
            c_coeff_blocks = { (1,0): 1 }
        elif s_idx == 2: # M3
            c_coeff_blocks = { (0,1): 1 }
        elif s_idx == 3: # M4
            c_coeff_blocks = { (0,0): 1, (1,0): 1 }
        elif s_idx == 4: # M5
            c_coeff_blocks = { (0,1): 1, (1,1): -1 }
        elif s_idx == 5: # M6
            c_coeff_blocks = { (1,1): 1 }
        elif s_idx == 6: # M7
            c_coeff_blocks = { (0,0): 1, (1,1): -1 }
            
        for b_idx in range(7):
            global_t = s_idx * 7 + b_idx
            
            # Initialize local vectors for this rank-1 term
            # We accumulate into the global U, V, W
            
            # U part
            for lr in range(2):
                for lc in range(2):
                    local_a_idx = lr * 2 + lc
                    coeff_u_base = base_U[local_a_idx][b_idx]
                    if coeff_u_base == 0:
                        continue
                    
                    # This local element corresponds to position (lr, lc) in the 2x2 matrix X.
                    # X is formed from A blocks.
                    # X[lr, lc] = sum_{(br, bc) in a_blocks} coeff_a[br, bc] * A_block(br, bc)[lr, lc]
                    for (br, bc), coeff_a in a_blocks.items():
                        if coeff_a == 0:
                            continue
                        # Global A index
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        ga_idx = a_idx(gr, gc)
                        U[ga_idx][global_t] += coeff_u_base * coeff_a
                        
            # V part
            for lr in range(2):
                for lc in range(2):
                    local_b_idx = lr * 2 + lc
                    coeff_v_base = base_V[local_b_idx][b_idx]
                    if coeff_v_base == 0:
                        continue
                    
                    for (br, bc), coeff_b in b_blocks.items():
                        if coeff_b == 0:
                            continue
                        # Global B1 index
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        gb_idx = b1_idx(gr, gc)
                        V[gb_idx][global_t] += coeff_v_base * coeff_b
                        
            # W part
            for lr in range(2):
                for lc in range(2):
                    local_c_idx = lr * 2 + lc
                    coeff_w_base = base_W[local_c_idx][b_idx]
                    if coeff_w_base == 0:
                        continue
                    
                    # The result of the 2x2 product M is a 2x2 matrix.
                    # The element M[lr, lc] contributes to the C blocks.
                    # Which C blocks does M[lr, lc] belong to?
                    # M is the result of the s-th Strassen product.
                    # M is added to specific C blocks.
                    # The local position (lr, lc) in M corresponds to the local position in the target C block.
                    # So if M is added to C_block(br, bc) with coefficient coeff_c,
                    # then M[lr, lc] contributes to C_global(br*2+lr, bc*2+lc) with coeff_c * coeff_w_base.
                    
                    for (br, bc), coeff_c in c_coeff_blocks.items():
                        if coeff_c == 0:
                            continue
                        # Global C index
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        gc_idx = c1_idx(gr, gc)
                        W[gc_idx][global_t] += coeff_w_base * coeff_c
                        
    # --- Part 2: 4x4x3 Schoolbook (Rank 48) ---
    
    # B2 is cols 4,5,6 of B.
    # C2 is cols 4,5,6 of C.
    # Trivial decomposition:
    # For each i in 0..3, j in 0..3, k in 4..6:
    # Term t corresponds to (i, j, k).
    # U[i*m+j][t] = 1
    # V[j*p+k][t] = 1
    # W[k*n+i][t] = 1
    
    t_offset = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                t = t_offset + (i * m + j) * 3 + (k - 4) # Just a unique index
                # Actually, let's just use a counter
                pass
    
    # Simpler loop for schoolbook
    t_idx = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][t_idx] = 1
                V[v_idx][t_idx] = 1
                W[w_idx][t_idx] = 1
                t_idx += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
