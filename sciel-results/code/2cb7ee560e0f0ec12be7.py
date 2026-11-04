import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We will build the matrices column by column
    # Rank is expected to be 88
    # Let's collect the terms first
    terms = []
    
    # Helper to add a Strassen 2x2x2 block
    # A indices: (i, j) where i in {0,1}, j in {j_off, j_off+1}
    # B indices: (j, k) where j in {j_off, j_off+1}, k in {k_off, k_off+1}
    # C indices: (i, k) where i in {0,1}, k in {k_off, k_off+1}
    # Note: In our global indexing:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    def add_strassen_2x2x2(i_base, j_base, k_base):
        # i_base: 0 or 1 (for the 2x2 block in A)
        # j_base: 0, 1, 2, 3 (start of 2 cols in A, start of 2 rows in B)
        # k_base: 0, 1, 2, 3, 4, 5, 6, 7 (start of 2 cols in B, start of 2 cols in C)
        
        # Strassen products M1..M7
        # Each M is (A_comb) * (B_comb)
        # A_comb is a vector in U space (2x2 block)
        # B_comb is a vector in V space (2x2 block)
        # W contribution is determined by Strassen formulas
        
        # Local indices for 2x2 block:
        # A: (0,0), (0,1), (1,0), (1,1) -> Global U: (i_base*m + j_base), (i_base*m + j_base+1), ((i_base+1)*m + j_base), ((i_base+1)*m + j_base+1)
        # B: (0,0), (0,1), (1,0), (1,1) -> Global V: (j_base*p + k_base), (j_base*p + k_base+1), ((j_base+1)*p + k_base), ((j_base+1)*p + k_base+1)
        # C: (0,0), (0,1), (1,0), (1,1) -> Global W: (k_base*n + i_base), (k_base*n + i_base+1), ((k_base+1)*n + i_base), ((k_base+1)*n + i_base+1)
        
        # We define the 7 Strassen terms
        # U_vecs, V_vecs, W_vecs for each term
        
        # M1 = A11 * (B11 + B22)
        # U: A11 -> [1, 0, 0, 0] in local 2x2
        # V: B11+B22 -> [1, 0, 0, 1] in local 2x2 (B11 at 0, B22 at 3)
        # W: C11 += 1, C22 += 1
        
        # We will store (U_sparse, V_sparse, W_sparse) for each of the 7 terms
        # Sparse representation: list of (index, value)
        
        strassen_terms = [
            # M1
            {
                'U': [(0, 1)], # A11
                'V': [(0, 1), (3, 1)], # B11 + B22
                'W': [(0, 1), (3, 1)] # C11 + C22
            },
            # M2
            {
                'U': [(0, 1), (2, 1)], # A11 + A21
                'V': [(3, 1)], # B22
                'W': [(2, 1), (3, -1)] # C21 + C22 (M2 contributes to C21 and -C22? No, C22 = M1 - M2 ... so -1 for C22)
                # Wait, let's use the standard formulas carefully.
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
            },
            # M3
            {
                'U': [(1, 1), (3, 1)], # A12 + A22
                'V': [(0, 1)], # B11
                'W': [(1, 1), (3, 1)] # C12 + C22
            },
            # M4
            {
                'U': [(3, 1)], # A22
                'V': [(2, 1), (0, -1)], # B21 - B11
                'W': [(0, 1), (2, 1)] # C11 + C21
            },
            # M5
            {
                'U': [(0, 1), (1, 1)], # A11 + A12
                'V': [(2, 1)], # B21
                'W': [(0, -1), (1, 1)] # -C11 + C12
            },
            # M6
            {
                'U': [(2, 1), (0, -1)], # A21 - A11
                'V': [(0, 1), (1, 1)], # B11 + B12
                'W': [(3, 1)] # C22
            },
            # M7
            {
                'U': [(1, 1), (3, -1)], # A12 - A22
                'V': [(2, 1), (3, 1)], # B21 + B22
                'W': [(0, 1)] # C11
            }
        ]
        
        # Map local indices to global indices
        # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global U index = (i_base + local_i) * m + (j_base + local_j)
        u_map = {
            0: (i_base * m + j_base),
            1: (i_base * m + j_base + 1),
            2: ((i_base + 1) * m + j_base),
            3: ((i_base + 1) * m + j_base + 1)
        }
        
        # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global V index = (j_base + local_j) * p + (k_base + local_k)
        # Note: In Strassen, B is 2x2.
        # Local B (0,0) is j=0, k=0.
        v_map = {
            0: (j_base * p + k_base),
            1: (j_base * p + k_base + 1),
            2: ((j_base + 1) * p + k_base),
            3: ((j_base + 1) * p + k_base + 1)
        }
        
        # Local W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global W index = (k_base + local_k) * n + (i_base + local_i)
        w_map = {
            0: (k_base * n + i_base),
            1: (k_base * n + i_base + 1),
            2: ((k_base + 1) * n + i_base),
            3: ((k_base + 1) * n + i_base + 1)
        }
        
        for term in strassen_terms:
            u_sparse = [(u_map[idx], val) for idx, val in term['U']]
            v_sparse = [(v_map[idx], val) for idx, val in term['V']]
            w_sparse = [(w_map[idx], val) for idx, val in term['W']]
            terms.append((u_sparse, v_sparse, w_sparse))

    # Helper to add Schoolbook 1x4x4 block
    # A row i, B rows j, C cols k
    def add_schoolbook_1x4x4(i, j_base, k_base):
        # i is the row of A (and C)
        # j_base is start of 4 rows in B
        # k_base is start of 4 cols in B (and C)
        for dj in range(4):
            for dk in range(4):
                u_idx = i * m + (j_base + dj)
                v_idx = (j_base + dj) * p + (k_base + dk)
                w_idx = (k_base + dk) * n + i
                terms.append(
                    ([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                )

    # 1. A1 B1 (i=0,1; j=0..3; k=0..3)
    # Split into 4 2x2x2 blocks
    # j_blocks: 0, 2
    # k_blocks: 0, 2
    for jb in [0, 2]:
        for kb in [0, 2]:
            add_strassen_2x2x2(0, jb, kb) # i_base=0
            add_strassen_2x2x2(1, jb, kb) # i_base=1
            # Wait, A1 is 2x4. i=0,1.
            # My add_strassen_2x2x2 takes i_base which is the row index in the 2x2 A block.
            # But A1 is 2 rows.
            # So for A1, i is 0 and 1.
            # The Strassen block is 2x2x2.
            # It covers i in {i_base, i_base+1}?
            # No, the Strassen algorithm is for 2x2 matrices.
            # A is 2x2.
            # So for A1 (2x4), we split columns.
            # We do NOT split rows of A1 into two 1x2 blocks for Strassen.
            # We use the 2 rows of A1 as the 2x2 A matrix.
            # So i_base should be the start of the 2 rows?
            # In my add_strassen_2x2x2, i_base is used in u_map.
            # u_map uses i_base and i_base+1.
            # So if I call add_strassen_2x2x2(0, ...), it uses rows 0 and 1.
            # If I call add_strassen_2x2x2(1, ...), it uses rows 1 and 2.
            # But A1 only has rows 0 and 1.
            # So I should ONLY call add_strassen_2x2x2(0, ...) for the A1 part?
            # No, A1 is 2x4.
            # The Strassen block is 2x2x2.
            # It consumes 2 rows of A, 2 rows of B, 2 cols of B.
            # So for A1 (rows 0,1), I should use i_base=0.
            # I should NOT use i_base=1 because that would involve row 2 (which is A2).
            
            # Correction:
            # A1 is rows 0,1.
            # So for the 4 blocks of B (2x2), we use the SAME A1 rows (0,1).
            # So we call add_strassen_2x2x2(0, jb, kb) for each jb, kb.
            # We do NOT call it for i_base=1.
            
    # Re-doing the loop logic above:
    # The previous loop `add_strassen_2x2x2(0...)` and `add_strassen_2x2x2(1...)` was WRONG.
    # It would double count or use wrong rows.
    
    # Correct Logic for A1 B1:
    # A1 is rows 0,1.
    # B1 is cols 0-3.
    # Split B1 into 4 2x2 blocks.
    # For each block, use A1 (rows 0,1) and B_block (2x2).
    # This produces 4 Strassen decompositions.
    # Each uses i_base=0.
    
    for jb in [0, 2]:
        for kb in [0, 2]:
            add_strassen_2x2x2(0, jb, kb)

    # 2. A1 B2 (i=0,1; j=0..3; k=4..7)
    for jb in [0, 2]:
        for kb in [4, 6]:
            add_strassen_2x2x2(0, jb, kb)

    # 3. A2 B1 (i=2; j=0..3; k=0..3)
    add_schoolbook_1x4x4(2, 0, 0)

    # 4. A2 B2 (i=2; j=0..3; k=4..7)
    add_schoolbook_1x4x4(2, 0, 4)

    # Now construct U, V, W
    R = len(terms)
    U = [[0] * R for _ in range(U_rows)]
    V = [[0] * R for _ in range(V_rows)]
    W = [[0] * R for _ in range(W_rows)]
    
    for c, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, val in u_sp:
            U[idx][c] = val
        for idx, val in v_sp:
            V[idx][c] = val
        for idx, val in w_sp:
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
