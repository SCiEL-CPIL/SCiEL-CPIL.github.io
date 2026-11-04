"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4x4, duplicated for 4x4x8
    # 4x4x4 via Strassen has rank 7 (standard Strassen for 2x2x2 is rank 7, 
    # but for 4x4x4 we use a different approach)
    # Actually, let me use the known fact that 4x4x4 can be done with rank 21 
    # (Strassen's 7 * 3 = 21? No, that's not right)
    # 
    # Let me think more carefully. For 2x2x2, Strassen gives rank 7.
    # For 4x4x4, we can use Strassen recursively: 7 * 7 = 49? No.
    # Actually, the standard Strassen algorithm for 2x2 gives rank 7 for the 
    # multiplication tensor of size 2x2x2.
    #
    # For 4x4x4, one approach is to use the "Strassen scheme" which gives 
    # rank 7^2 = 49 for 4x4x4? No, that's not how it works.
    #
    # Let me use a simpler approach: use the known rank-21 decomposition for 
    # 4x4x4 (which is achievable via specific algebraic constructions).
    #
    # Actually, let me just use the straightforward Strassen approach:
    # Split 4x4 into 2x2 blocks. The 4x4x4 multiplication can be done with
    # 7 * (2x2x2 multiplications) where each 2x2x2 uses Strassen's 7-term
    # decomposition. But wait, that gives 7 * 7 = 49 terms, which is worse
    # than schoolbook's 64.
    #
    # Let me reconsider. The schoolbook for 4x4x4 is 64. 
    # Strassen for 2x2x2 is 7.
    # For 4x4x4, using Strassen: we need 7 "sub-multiplications" of 2x2 matrices,
    # each of which requires 7 scalar multiplications = 49 total.
    # So rank 49 < 64. That's an improvement!
    #
    # For 4x4x8: we can do this as 2 * (4x4x4), so rank 2 * 49 = 98.
    # Or we can do it more cleverly.
    #
    # Let me implement the Strassen 4x4x4 decomposition.
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # First, build the Strassen decomposition for 4x4x4
    # U, V, W for 4x4x4 with rank r_444
    
    # Strassen 2x2x2 has rank 7
    # For 4x4x4, we use 7 copies of 2x2x2 Strassen
    # Total rank = 7 * 7 = 49
    
    # Let me define the Strassen 2x2x2 terms first
    # For 2x2 matrices A, B, C = AB:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, split A, B into 2x2 blocks:
    # A = [A11 A12; A21 A22] where each Ai is 2x2
    # B = [B11 B12; B21 B22] where each Bj is 2x2
    # C = [C11 C12; C21 C22] where each Ck is 2x2
    
    # Using Strassen at the block level:
    # S1 = A11 + A22
    # S2 = A21 + A22
    # S3 = A11
    # S4 = A22
    # S5 = A11 + A12
    # S6 = A21 - A11
    # S7 = A12 - A22
    # T1 = B11 + B22
    # T2 = B11
    # T3 = B21 - B22
    # T4 = B12 - B11
    # T5 = B22
    # T6 = B11 + B12
    # T7 = B21 + B22
    
    # M1 = S1 * T1 (2x2 * 2x2)
    # M2 = S2 * T2
    # M3 = S3 * T3
    # M4 = S4 * T4
    # M5 = S5 * T5
    # M6 = S6 * T6
    # M7 = S7 * T7
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2*2x2 multiplication, which uses 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Let me build this systematically.
    # Indices for 4x4 matrix: row 0-3, col 0-3
    # A[i,j], B[j,k], C[i,k]
    
    # 2x2 block indices:
    # A11: rows 0,1 cols 0,1 -> A[0,0], A[0,1], A[1,0], A[1,1]
    # A12: rows 0,1 cols 2,3 -> A[0,2], A[0,3], A[1,2], A[1,3]
    # A21: rows 2,3 cols 0,1 -> A[2,0], A[2,1], A[3,0], A[3,1]
    # A22: rows 2,3 cols 2,3 -> A[2,2], A[2,3], A[3,2], A[3,3]
    
    # B11: rows 0,1 cols 0,1 -> B[0,0], B[0,1], B[1,0], B[1,1]
    # B12: rows 0,1 cols 2,3 -> B[0,2], B[0,3], B[1,2], B[1,3]
    # B21: rows 2,3 cols 0,1 -> B[2,0], B[2,1], B[3,0], B[3,1]
    # B22: rows 2,3 cols 2,3 -> B[2,2], B[2,3], B[3,2], B[3,3]
    
    # C11: rows 0,1 cols 0,1 -> C[0,0], C[0,1], C[1,0], C[1,1]
    # C12: rows 0,1 cols 2,3 -> C[0,2], C[0,3], C[1,2], C[1,3]
    # C21: rows 2,3 cols 0,1 -> C[2,0], C[2,1], C[3,0], C[3,1]
    # C22: rows 2,3 cols 2,3 -> C[2,2], C[2,3], C[3,2], C[3,3]
    
    # I'll build the 4x4x4 decomposition with rank 49.
    
    # Helper: create a Strassen 2x2x2 decomposition
    # Returns lists of (U_entries, V_entries, W_entries) for rank 7
    def strassen_2x2():
        # For 2x2x2: A[0,0], A[0,1], A[1,0], A[1,1], etc.
        # U has 4 rows (for A entries), V has 4 rows (for B entries), W has 4 rows (for C entries)
        # 7 terms, each with U, V, W vectors of length 4
        
        terms = []
        
        # Term 1: M1 = (a00+a11)(b00+b11)
        # U: [1, 0, 1, 0] for A
        # V: [1, 1, 0, 0] for B  
        # W: contributes to c00 (+1), c11 (+1)
        terms.append(([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]))
        
        # Term 2: M2 = (a10+a11)b00
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: contributes to c10 (+1)
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        
        # Term 3: M3 = a00(b10-b11)
        # U: [1, 0, 0, 0]
        # V: [0, -1, 1, 0]
        # W: contributes to c01 (+1)
        terms.append(([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0]))
        
        # Term 4: M4 = a11(b01-b00)
        # U: [0, 0, 0, 1]
        # V: [-1, 1, 0, 0]
        # W: contributes to c00 (+1), c10 (+1)
        terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]))
        
        # Term 5: M5 = (a00+a01)b11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: contributes to c00 (-1), c01 (+1)
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        
        # Term 6: M6 = (a10-a00)(b00+b01)
        # U: [-1, 0, 1, 0]
        # V: [1, 1, 0, 0]
        # W: contributes to c11 (+1)
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # Term 7: M7 = (a01-a11)(b10+b11)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: contributes to c00 (+1)
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return terms
    
    # Now build 4x4x4 using 7 copies of Strassen 2x2x2
    # Each copy handles a pair of 2x2 blocks
    
    # The 7 block-level terms:
    # (S_idx, T_idx, C_block_coeffs)
    # S1=T1 -> C11(+1), C22(+1)  [wait, let me re-derive]
    
    # Block Strassen:
    # S1 = A11 + A22, T1 = B11 + B22, M1 = S1*T1
    # S2 = A21 + A22, T2 = B11,    M2 = S2*T2
    # S3 = A11,        T3 = B21-B22, M3 = S3*T3
    # S4 = A22,        T4 = B12-B11, M4 = S4*T4
    # S5 = A11+A12,    T5 = B22,     M5 = S5*T5
    # S6 = A21-A11,    T6 = B11+B12, M6 = S6*T6
    # S7 = A12-A22,    T7 = B21+B22, M7 = S7*T7
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block term Mk, we need to:
    # 1. Form the linear combination of A-blocks (S_k)
    # 2. Form the linear combination of B-blocks (T_k)
    # 3. Multiply them using Strassen 2x2x2 (7 scalar mults)
    # 4. Add the result to the appropriate C-blocks with the right coefficients
    
    # A-block indices (in the 4x4 A matrix):
    # A11: rows [0,1], cols [0,1]
    # A12: rows [0,1], cols [2,3]
    # A21: rows [2,3], cols [0,1]
    # A22: rows [2,3], cols [2,3]
    
    # B-block indices (in the 4x4 B matrix):
    # B11: rows [0,1], cols [0,1]
    # B12: rows [0,1], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    # B22: rows [2,3], cols [2,3]
    
    # C-block indices (in the 4x4 C matrix):
    # C11: rows [0,1], cols [0,1]
    # C12: rows [0,1], cols [2,3]
    # C21: rows [2,3], cols [0,1]
    # C22: rows [2,3], cols [2,3]
    
    # Define the block-level Strassen terms:
    # Each term: (A_block_coeffs, B_block_coeffs)
    # A_block_coeffs: dict from block name to coefficient
    # B_block_coeffs: dict from block name to coefficient
    
    block_terms = [
        # (A_coeffs, B_coeffs)
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}),  # M1
        ({'A21': 1, 'A22': 1}, {'B11': 1}),              # M2
        ({'A11': 1}, {'B21': 1, 'B22': -1}),             # M3
        ({'A22': 1}, {'B12': 1, 'B11': -1}),             # M4
        ({'A11': 1, 'A12': 1}, {'B22': 1}),              # M5
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}),   # M6
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}),   # M7
    ]
    
    # C-block contributions from each M_k:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    c_contrib = {
        0: {'C11': 1, 'C22': 1},     # M1
        1: {'C21': 1},                # M2
        2: {'C12': 1},                # M3
        3: {'C11': 1, 'C21': 1},     # M4
        4: {'C11': -1, 'C12': 1},    # M5
        5: {'C22': 1},                # M6
        6: {'C11': 1},                # M7
    }
    
    # Now I need to map block indices to actual matrix indices
    # A-block to (row_start, col_start):
    a_block_pos = {
        'A11': (0, 0),
        'A12': (0, 2),
        'A21': (2, 0),
        'A22': (2, 2),
    }
    
    b_block_pos = {
        'B11': (0, 0),
        'B12': (0, 2),
        'B21': (2, 0),
        'B22': (2, 2),
    }
    
    c_block_pos = {
        'C11': (0, 0),
        'C12': (0, 2),
        'C21': (2, 0),
        'C22': (2, 2),
    }
    
    # Build the 4x4x4 decomposition
    # U: 16 rows (4x4 A entries)
    # V: 16 rows (4x4 B entries)  
    # W: 16 rows (4x4 C entries)
    # Rank: 7 * 7 = 49
    
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]
    
    col = 0
    for k in range(7):
        a_coeffs, b_coeffs = block_terms[k]
        strassen_terms = strassen_2x2()
        
        for st in strassen_terms:
            u_2x2, v_2x2, w_2x2 = st
            
            # u_2x2: [a00, a01, a10, a11] coefficients for the 2x2 A matrix
            # v_2x2: [b00, b01, b10, b11] coefficients for the 2x2 B matrix
            # w_2x2: [c00, c01, c10, c11] coefficients for the 2x2 C matrix
            
            # For each element in u_2x2, add to U
            for dr in range(2):
                for dc in range(2):
                    idx_in_block = dr * 2 + dc  # 0,1,2,3
                    coeff = u_2x2[idx_in_block]
                    if coeff == 0:
                        continue
                    # For each A-block that contributes to this term
                    for a_block_name, a_coeff in a_coeffs.items():
                        row_start, col_start = a_block_pos[a_block_name]
                        global_row = row_start + dr
                        global_col = col_start + dc
                        u_idx = global_row * 4 + global_col  # A is 4x4, row-major
                        U_444[u_idx][col] += coeff * a_coeff
            
            # For each element in v_2x2, add to V
            for dr in range(2):
                for dc in range(2):
                    idx_in_block = dr * 2 + dc
                    coeff = v_2x2[idx_in_block]
                    if coeff == 0:
                        continue
                    for b_block_name, b_coeff in b_coeffs.items():
                        row_start, col_start = b_block_pos[b_block_name]
                        global_row = row_start + dr
                        global_col = col_start + dc
                        v_idx = global_row * 4 + global_col  # B is 4x4, row-major
                        V_444[v_idx][col] += coeff * b_coeff
            
            # For each element in w_2x2, add to W
            # w_2x2 gives contributions to a 2x2 C matrix
            # But we need to distribute to the correct C-blocks
            for c_block_name, c_coeff in c_contrib[k].items():
                if c_coeff == 0:
                    continue
                row_start, col_start = c_block_pos[c_block_name]
                for dr in range(2):
                    for dc in range(2):
                        idx_in_block = dr * 2 + dc
                        coeff = w_2x2[idx_in_block]
                        if coeff == 0:
                            continue
                        global_row = row_start + dr
                        global_col = col_start + dc
                        w_idx = global_row * 4 + global_col  # C is 4x4, row-major
                        W_444[w_idx][col] += coeff * c_coeff
            
            col += 1
    
    # Now build 4x4x8 by duplicating the 4x4x4 decomposition
    # B is 4x8, so we split it into two 4x4 halves: B[:,0:4] and B[:,4:8]
    # C is 4x8, so C[:,0:4] = A * B[:,0:4] and C[:,4:8] = A * B[:,4:8]
    
    # For 4x4x8:
    # U: 16 rows (4x4 A entries)
    # V: 32 rows (4x8 B entries)
    # W: 32 rows (4x8 C entries)
    # Rank: 2 * 49 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # First half: B[:,0:4] -> C[:,0:4]
    for r in range(16):
        for c in range(49):
            U[r][c] = U_444[r][c]
            V[r][c] = V_444[r][c]
            W[r][c] = W_444[r][c]
    
    # Second half: B[:,4:8] -> C[:,4:8]
    for r in range(16):
        for c in range(49):
            U[r][c + 49] = U_444[r][c]
            # B indices: B[j, 4+k] for j in 0..3, k in 0..3
            # In V_444, index is j*4 + k
            # In V, index is j*8 + (4+k)
            V_idx_444 = r  # This is for B, which is 4x4
            # Wait, I need to be more careful.
            # V_444 has 16 rows for 4x4 B
            # V has 32 rows for 4x8 B
            # B_444[j, k] maps to V_444[j*4 + k]
            # B_48[j, k] for k in 0..7 maps to V[j*8 + k]
            # For the second half, B_48[j, 4+k] = B_444[j, k]
            # So V_444[j*4 + k] maps to V[j*8 + 4 + k]
            
    # Let me redo this more carefully
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # First half: columns 0-3 of B, columns 0-3 of C
    for r_u in range(16):
        for c in range(49):
            U[r_u][c] = U_444[r_u][c]
    
    for r_v in range(16):
        # r_v corresponds to B_444[j, k] where j = r_v // 4, k = r_v % 4
        j = r_v // 4
        k = r_v % 4
        v_idx_48 = j * 8 + k  # B_48[j, k] for k in 0..3
        for c in range(49):
            V[v_idx_48][c] = V_444[r_v][c]
    
    for r_w in range(16):
        # r_w corresponds to C_444[i, k] where i = r_w // 4, k = r_w % 4
        i = r_w // 4
        k = r_w % 4
        w_idx_48 = k * 4 + i  # C_48[k, i] -> wait, the convention is W[k*n+i, r]
        # For 4x4x4: W[k*4+i, r] for k in 0..3, i in 0..3
        # For 4x4x8: W[k*4+i, r] for k in 0..7, i in 0..3
        # So C_444[i, k] (row i, col k) maps to W_444[k*4+i]
        # C_48[i, k] for k in 0..7 maps to W_48[k*4+i]
        # For the first half, k in 0..3: C_48[i, k] = C_444[i, k]
        # So W_444[k*4+i] maps to W_48[k*4+i]
        w_idx_444 = r_w  # This is already k*4+i
        for c in range(49):
            W[w_idx_444][c] = W_444[r_w][c]
    
    # Second half: columns 4-7 of B, columns 4-7 of C
    for r_u in range(16):
        for c in range(49):
            U[r_u][c + 49] = U_444[r_u][c]
    
    for r_v in range(16):
        j = r_v // 4
        k = r_v % 4
        v_idx_48 = j * 8 + (4 + k)  # B_48[j, 4+k]
        for c in range(49):
            V[v_idx_48][c + 49] = V_444[r_v][c]
    
    for r_w in range(16):
        # r_w is k*4+i for k in 0..3, i in 0..3
        # For the second half, k -> k+4
        w_idx_444 = r_w
        k = r_w // 4
        i = r_w % 4
        w_idx_48 = (4 + k) * 4 + i
        for c in range(49):
            W[w_idx_48][c + 49] = W_444[r_w][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
