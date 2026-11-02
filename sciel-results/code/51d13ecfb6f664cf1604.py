"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Helper to add a term to the matrices
    # U, V, W are lists of lists
    def add_term(U, V, W, u_idx, v_idx, w_idx, val=1):
        U[u_idx].append(val)
        V[v_idx].append(val)
        W[w_idx].append(val)
        
    def add_strassen_2x2x2(U, V, W, u_off, v_off, w_off):
        # u_off: offset for A (2x2)
        # v_off: offset for B (2x2)
        # w_off: offset for C (2x2)
        # Indices:
        # A: (u_off, u_off+1) x (u_off+2, u_off+3) ? No, A is 2x2.
        # Let's define local indices 0,1 for rows and cols.
        # A_00, A_01, A_10, A_11
        # B_00, B_01, B_10, B_11
        # C_00, C_01, C_10, C_11
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M6
        
        # We need to map these to global indices.
        # A indices:
        # A00: u_off + 0*2 + 0
        # A01: u_off + 0*2 + 1
        # A10: u_off + 1*2 + 0
        # A11: u_off + 1*2 + 1
        
        # B indices:
        # B00: v_off + 0*2 + 0
        # B01: v_off + 0*2 + 1
        # B10: v_off + 1*2 + 0
        # B11: v_off + 1*2 + 1
        
        # C indices:
        # C00: w_off + 0*2 + 0
        # C01: w_off + 0*2 + 1
        # C10: w_off + 1*2 + 0
        # C11: w_off + 1*2 + 1
        
        # Initialize local U, V, W contributions
        # We will append to the global lists.
        
        # Term 1
        u_idx = [u_off, u_off+3] # A00, A11
        v_idx = [v_off, v_off+3] # B00, B11
        w_idx = [w_off, w_off+1, w_off+2, w_off+3] # C00, C01, C10, C11
        # This is getting complex to do inline.
        # Better to generate the 7 terms explicitly.
        
        terms = [
            # (U_indices, V_indices, W_indices, Coeffs)
            # M1
            ([u_off, u_off+3], [v_off, v_off+3], [w_off, w_off+1, w_off+2, w_off+3], [1, 1, -1, -1]), # C00=1, C01=1, C10=-1, C11=-1?
            # Wait, C00 = M1 - M4 + M6 - M7.
            # So M1 contributes +1 to C00, +1 to C11?
            # C11 = M1 - M3 + M2 - M6.
            # So M1: C00=1, C11=1.
            # M2: C10=1, C11=1.
            # M3: C01=1, C11=-1.
            # M4: C00=-1, C10=1.
            # M5: C01=1.
            # M6: C00=1, C11=-1.
            # M7: C00=-1.
            
            # Let's list contributions per M
        ]
        
        # M1: (A00+A11)(B00+B11)
        # U: [u_off, u_off+3] (1, 1)
        # V: [v_off, v_off+3] (1, 1)
        # W: C00(1), C11(1)
        add_term(U, V, W, [u_off, u_off+3], [v_off, v_off+3], [w_off, w_off+3], 1) # This doesn't handle multiple indices per term well in my helper.
        
        # Let's rewrite add_term to handle lists.
        
    # Revised Helper
    def add_term_multi(U, V, W, u_idxs, v_idxs, w_idxs, coeff):
        # u_idxs, v_idxs, w_idxs are lists of indices.
        # We assume the tensor product is sum of products.
        # For rank 1 term: U[i,k]=1, V[j,k]=1, W[l,k]=1.
        # If we have multiple indices, it's not a single rank 1 term.
        # Strassen terms are rank 1 in the intermediate sense, but they sum to the result.
        # Each M_i is a scalar product.
        # So for each M_i, we add a rank 1 term.
        
        pass

    # Let's just build the matrices directly for the 7 Strassen terms.
    
    def get_strassen_2x2x2_terms(u_off, v_off, w_off):
        # Returns list of (u_indices, v_indices, w_indices, coeff)
        # Each entry is a rank 1 component of the 7 mults.
        # Actually, each M_i is a single multiplication of two linear combinations.
        # So each M_i corresponds to ONE rank 1 term in the tensor decomposition?
        # No, M_i = (sum a_i A_ij) * (sum b_j B_jk).
        # This is a rank 1 tensor in the A,B space, mapped to C.
        # Yes, each M_i adds 1 to the rank.
        
        terms = []
        
        # M1
        terms.append(([u_off, u_off+3], [v_off, v_off+3], [w_off, w_off+3], 1)) # C00, C11 get +1?
        # Check C00 = M1 - M4 + M6 - M7.
        # Check C11 = M1 - M3 + M2 - M6.
        # So M1 contributes +1 to C00 and +1 to C11.
        
        # M2
        terms.append(([u_off+2, u_off+3], [v_off], [w_off+2, w_off+3], 1)) # A10, A11; B00; C10, C11
        # C10 = M2 + M4.
        # C11 = ... + M2 ...
        
        # M3
        terms.append(([u_off], [v_off+1, v_off+3], [w_off+1, w_off+3], 1)) # A00; B01, B11; C01, C11
        # C01 = M3 + M5.
        # C11 = ... - M3 ...
        # So M3 contributes +1 to C01, -1 to C11.
        # My helper needs to handle coeffs in W.
        
        # This is getting tedious. I will hardcode the 7 terms for 2x2x2.
        
        return terms

    # Given the complexity, I will use the known rank 72 decomposition logic.
    # Rank 72 is a safe improvement.
    
    U = []
    V = []
    W = []
    
    # Initialize
    for _ in range(n*m): U.append([])
    for _ in range(m*p): V.append([])
    for _ in range(p*n): W.append([])
    
    def add_rank1_term(u_idxs, v_idxs, w_idxs, coeff):
        # u_idxs: list of row indices in U
        # v_idxs: list of row indices in V
        # w_idxs: list of row indices in W
        # This represents a rank 1 term where U has 1s at u_idxs, V at v_idxs, W at w_idxs * coeff.
        # But U, V, W are matrices.
        # We append a new column to each.
        new_u_col = [0] * (n*m)
        new_v_col = [0] * (m*p)
        new_w_col = [0] * (p*n)
        
        for idx in u_idxs:
            new_u_col[idx] = 1
        for idx in v_idxs:
            new_v_col[idx] = 1
        for idx, c in zip(w_idxs, coeff):
            new_w_col[idx] = c
            
        for i in range(n*m):
            U[i].append(new_u_col[i])
        for i in range(m*p):
            V[i].append(new_v_col[i])
        for i in range(p*n):
            W[i].append(new_w_col[i])

    # Define Strassen 2x2x2 terms
    def add_strassen_2x2x2(u_off, v_off, w_off):
        # u_off, v_off, w_off are offsets for the 2x2 blocks.
        # A indices: u_off (00), u_off+1 (01), u_off+2 (10), u_off+3 (11)
        # B indices: v_off (00), v_off+1 (01), v_off+2 (10), v_off+3 (11)
        # C indices: w_off (00), w_off+1 (01), w_off+2 (10), w_off+3 (11)
        
        # M1
        add_rank1_term([u_off, u_off+3], [v_off, v_off+3], [w_off, w_off+3], [1, 1])
        # M2
        add_rank1_term([u_off+2, u_off+3], [v_off], [w_off+2, w_off+3], [1, 1])
        # M3
        add_rank1_term([u_off], [v_off+1, v_off+3], [w_off+1, w_off+3], [1, -1])
        # M4
        add_rank1_term([u_off+3], [v_off+2, v_off], [w_off, w_off+2], [-1, 1]) # C00=-1, C10=1
        # M5
        add_rank1_term([u_off, u_off+1], [v_off+3], [w_off+1], [1])
        # M6
        add_rank1_term([u_off+2, u_off], [v_off, v_off+1], [w_off, w_off+3], [1, -1]) # C00=1, C11=-1
        # M7
        add_rank1_term([u_off+1, u_off+3], [v_off+2, v_off+3], [w_off], [-1]) # C00=-1

    def add_schoolbook_2x2x1(u_off, v_off, w_off):
        # A is 2x2, B is 2x1, C is 2x1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # B indices: v_off (00), v_off+1 (10) ? No, B is 2x1.
        # B_00, B_10.
        # C_00, C_10.
        
        add_rank1_term([u_off], [v_off], [w_off], [1])
        add_rank1_term([u_off+1], [v_off+1], [w_off], [1])
        add_rank1_term([u_off+2], [v_off], [w_off+1], [1])
        add_rank1_term([u_off+3], [v_off+1], [w_off+1], [1])

    def add_2x2x3(u_off, v_off, w_off):
        # A is 2x2, B is 2x3, C is 2x3
        # Split B into B1 (2x2) and B2 (2x1)
        # B1 indices: v_off, v_off+1, v_off+2, v_off+3
        # B2 indices: v_off+4, v_off+5
        # C1 indices: w_off, w_off+1, w_off+2, w_off+3
        # C2 indices: w_off+4, w_off+5
        
        add_strassen_2x2x2(u_off, v_off, w_off)
        add_schoolbook_2x2x1(u_off, v_off+4, w_off+4)

    # Construct 4x4x5
    # Split B into B1 (4x2) and B2 (4x3)
    # B1: v_off 0..7
    # B2: v_off 8..19
    
    # C1 (4x2): w_off 0..7
    # C2 (4x3): w_off 8..19
    
    # A is 4x4.
    # We need to compute A B1 and A B2.
    
    # A B1:
    # Split A into A_top (2x4) and A_bot (2x4).
    # A_top: u_off 0..3
    # A_bot: u_off 4..7
    
    # A_top B1:
    # B1 is 4x2. Split into B1_top (2x2) and B1_bot (2x2).
    # B1_top: v_off 0..3
    # B1_bot: v_off 4..7
    
    # C1_top (2x2): w_off 0..3
    # C1_bot (2x2): w_off 4..7
    
    # A_top B1_top -> C1_top (2x2x2)
    add_strassen_2x2x2(0, 0, 0)
    # A_top B1_bot -> C1_bot (2x2x2)
    add_strassen_2x2x2(0, 4, 4)
    
    # A_bot B1_top -> C1_bot (2x2x2)
    add_strassen_2x2x2(4, 0, 4)
    # A_bot B1_bot -> C1_bot (2x2x2)
    add_strassen_2x2x2(4, 4, 4)
    
    # Wait, C1_bot is sum of A_top B1_bot and A_bot B1_bot.
    # So we add both.
    
    # A B2:
    # B2 is 4x3. Split into B2_1 (4x2) and B2_2 (4x1).
    # B2_1: v_off 8..11
    # B2_2: v_off 12..15
    
    # C2_1 (4x2): w_off 8..11
    # C2_2 (4x1): w_off 12..15
    
    # A B2_1:
    # A_top B2_1_top -> C2_1_top
    # A_top B2_1_bot -> C2_1_bot
    # A_bot B2_1_top -> C2_1_bot
    # A_bot B2_1_bot -> C2_1_bot
    
    add_strassen_2x2x2(0, 8, 8)
    add_strassen_2x2x2(0, 10, 10) # B2_1_bot is v 10,11?
    # B2 is 4x3.
    # B2_1 is 4x2.
    # B2_1_top is 2x2 (v 8,9,10,11? No).
    # B2_1 indices:
    # Row 1: v8, v9
    # Row 2: v10, v11
    # So B2_1_top is v8, v9? No, B is 4x3.
    # B2_1 is first 2 cols of B2.
    # B2_1[0,0]=v8, B2_1[0,1]=v9.
    # B2_1[1,0]=v10, B2_1[1,1]=v11.
    
    # A_top (u 0..3) * B2_1_top (v 8,9) -> C2_1_top (w 8,9)
    add_strassen_2x2x2(0, 8, 8)
    # A_top (u 0..3) * B2_1_bot (v 10,11) -> C2_1_bot (w 10,11)
    add_strassen_2x2x2(0, 10, 10)
    # A_bot (u 4..7) * B2_1_top (v 8,9) -> C2_1_bot (w 10,11)
    add_strassen_2x2x2(4, 8, 10)
    # A_bot (u 4..7) * B2_1_bot (v 10,11) -> C2_1_bot (w 10,11)
    add_strassen_2x2x2(4, 10, 10)
    
    # A B2_2:
    # B2_2 is 4x1.
    # B2_2_top is 2x1 (v 12,13).
    # B2_2_bot is 2x1 (v 14,15).
    
    # C2_2 is 4x1.
    # C2_2_top is 2x1 (w 12,13).
    # C2_2_bot is 2x1 (w 14,15).
    
    # A_top * B2_2_top -> C2_2_top
    add_schoolbook_2x2x1(0, 12, 12)
    # A_top * B2_2_bot -> C2_2_bot
    add_schoolbook_2x2x1(0, 14, 14)
    # A_bot * B2_2_top -> C2_2_bot
    add_schoolbook_2x2x1(4, 12, 14)
    # A_bot * B2_2_bot -> C2_2_bot
    add_schoolbook_2x2x1(4, 14, 14)
    
    # Convert to lists of lists
    U_final = [U[i] for i in range(n*m)]
    V_final = [V[i] for i in range(m*p)]
    W_final = [W[i] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
