"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    n, m, p = DIMENSIONS
    
    # We will construct a rank 88 decomposition.
    # Structure:
    # Split p=8 into two blocks of 4.
    # For each 3x4x4 block, split m=4 into two blocks of 2.
    # This results in 8 blocks of 3x2x2.
    # Each 3x2x2 block is decomposed with rank 11 (7 for 2x2x2 part + 4 for 1x2x2 part).
    # Total rank = 8 * 11 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U: i*m + j (i in 0..2, j in 0..3)
    # Indices for V: j*p + k (j in 0..3, k in 0..7)
    # Indices for W: k*n + i (k in 0..7, i in 0..2)
    
    # We iterate over the 8 blocks of 3x2x2.
    # Block index b in 0..7
    # b determines:
    # - which half of p (k_offset): b // 4 (0 or 1) -> k in 0..3 or 4..7
    # - which half of m (j_offset): (b // 2) % 2 (0 or 1) -> j in 0..1 or 2..3
    # - which column of the 3x4x4 result (c_offset): b % 2 (0 or 1) -> determines which 2x2 block of B is used
    
    # Wait, the mapping of the 8 blocks to the 3x4x4 structure:
    # For a fixed 3x4x4 (p_half), we have 4 3x2x2 blocks.
    # A is split into A1 (j=0,1) and A2 (j=2,3).
    # B is split into B11, B12 (j=0,1) and B21, B22 (j=2,3).
    # C11 = A1 B11 + A2 B21
    # C12 = A1 B12 + A2 B22
    # So for C11, we need A1*B11 and A2*B21.
    # For C12, we need A1*B12 and A2*B22.
    
    # Let's define the 8 (A_block, B_block) pairs:
    # 1. A1 (j=0,1), B11 (j=0,1, k=0,1) -> contributes to C11
    # 2. A2 (j=2,3), B21 (j=2,3, k=0,1) -> contributes to C11
    # 3. A1 (j=0,1), B12 (j=0,1, k=2,3) -> contributes to C12
    # 4. A2 (j=2,3), B22 (j=2,3, k=2,3) -> contributes to C12
    # And the same for the second half of p (k=4..7).
    
    # Block definitions:
    # block_id: (j_start_A, k_start_B, j_start_B, target_k_in_C)
    # Actually, let's just iterate and map indices.
    
    blocks = []
    for p_half in range(2): # 0, 1
        k_base = p_half * 4
        for c_half in range(2): # 0, 1 (column of 3x4 result)
            # If c_half=0, C11. If c_half=1, C12.
            # C11 uses A1*B11 and A2*B21.
            # C12 uses A1*B12 and A2*B22.
            
            if c_half == 0:
                # A1 (j=0,1) with B11 (j=0,1, k=k_base..k_base+1)
                blocks.append((0, k_base, 0, k_base)) # A_j_start=0, B_k_start=k_base, B_j_start=0
                # A2 (j=2,3) with B21 (j=2,3, k=k_base..k_base+1)
                blocks.append((2, k_base, 2, k_base))
            else:
                # A1 (j=0,1) with B12 (j=0,1, k=k_base+2..k_base+3)
                blocks.append((0, k_base + 2, 0, k_base + 2))
                # A2 (j=2,3) with B22 (j=2,3, k=k_base+2..k_base+3)
                blocks.append((2, k_base + 2, 2, k_base + 2))
                
    # Each block has rank 11.
    # Total 8 blocks * 11 = 88.
    
    rank_offset = 0
    for b_idx in range(8):
        a_j_start, b_k_start, b_j_start, c_k_start = blocks[b_idx]
        
        # A indices: i in 0..2, j in a_j_start..a_j_start+1
        # B indices: j in b_j_start..b_j_start+1, k in b_k_start..b_k_start+1
        # C indices: i in 0..2, k in c_k_start..c_k_start+1
        
        # Decomposition of 3x2x2
        # Rows of A: 0, 1, 2 (local)
        # Cols of B: 0, 1 (local)
        # Rows of C: 0, 1, 2 (local)
        # Cols of C: 0, 1 (local)
        
        # 7 Strassen terms for first 2 rows
        # 4 Schoolbook terms for 3rd row
        
        # Term 1: (a00 + a11)(b00 + b11)
        # U: a00 (i=0, j=a_j_start), a11 (i=1, j=a_j_start+1)
        # V: b00 (j=b_j_start, k=b_k_start), b11 (j=b_j_start+1, k=b_k_start+1)
        # W: c00 (i=0, k=c_k_start), c11 (i=1, k=c_k_start+1)
        
        # We need to add contributions to U, V, W for each term.
        
        def add_term(u_indices, v_indices, w_indices, coeff):
            # u_indices: list of (row, col_in_U)
            # v_indices: list of (row, col_in_V)
            # w_indices: list of (row, col_in_W)
            # coeff: scalar multiplier for this term (1, -1, etc)
            pass

        # It's easier to just build the matrices directly.
        
        # Local indices mapping:
        # A local row r -> Global row r * m + (a_j_start + c_local) where c_local is 0 or 1?
        # No, A is 3x2. Rows are i=0,1,2. Cols are j=0,1.
        # Global A index: i * m + (a_j_start + j_local)
        
        # B local row r -> Global row (b_j_start + r) * p + (b_k_start + c_local)
        # B is 2x2. Rows j=0,1. Cols k=0,1.
        
        # C local row r -> Global row (c_k_start + c_local) * n + r
        # C is 3x2. Rows i=0,1,2. Cols k=0,1.
        
        # Strassen Terms (for A rows 0,1 and B cols 0,1)
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Schoolbook (for A row 2)
        # M8 = A20 B00
        # M9 = A20 B01
        # M10 = A21 B10
        # M11 = A21 B11
        
        # C20 = M8 + M10
        # C21 = M9 + M11
        
        # We will create 11 columns in U, V, W for this block.
        
        # Helper to get global indices
        def a_idx(i, j_loc):
            return i * m + (a_j_start + j_loc)
            
        def b_idx(j_loc, k_loc):
            return (b_j_start + j_loc) * p + (b_k_start + k_loc)
            
        def c_idx(i, k_loc):
            return (c_k_start + k_loc) * n + i

        # Term 1: (A00 + A11)(B00 + B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        U[a_idx(0, 0)][rank_offset] = 1
        U[a_idx(1, 1)][rank_offset] = 1
        V[b_idx(0, 0)][rank_offset] = 1
        V[b_idx(1, 1)][rank_offset] = 1
        W[c_idx(0, 0)][rank_offset] = 1
        W[c_idx(1, 1)][rank_offset] = 1
        rank_offset += 1
        
        # Term 2: (A10 + A11)B00
        # U: A10, A11
        # V: B00
        # W: C10
        U[a_idx(1, 0)][rank_offset] = 1
        U[a_idx(1, 1)][rank_offset] = 1
        V[b_idx(0, 0)][rank_offset] = 1
        W[c_idx(1, 0)][rank_offset] = 1
        rank_offset += 1
        
        # Term 3: A00(B01 - B11)
        # U: A00
        # V: B01, B11
        # W: C01
        U[a_idx(0, 0)][rank_offset] = 1
        V[b_idx(0, 1)][rank_offset] = 1
        V[b_idx(1, 1)][rank_offset] = -1
        W[c_idx(0, 1)][rank_offset] = 1
        rank_offset += 1
        
        # Term 4: A11(B10 - B00)
        # U: A11
        # V: B10, B00
        # W: C00, C10
        U[a_idx(1, 1)][rank_offset] = 1
        V[b_idx(1, 0)][rank_offset] = 1
        V[b_idx(0, 0)][rank_offset] = -1
        W[c_idx(0, 0)][rank_offset] = 1
        W[c_idx(1, 0)][rank_offset] = 1
        rank_offset += 1
        
        # Term 5: (A00 + A01)B11
        # U: A00, A01
        # V: B11
        # W: C00, C01
        U[a_idx(0, 0)][rank_offset] = 1
        U[a_idx(0, 1)][rank_offset] = 1
        V[b_idx(1, 1)][rank_offset] = 1
        W[c_idx(0, 0)][rank_offset] = -1
        W[c_idx(0, 1)][rank_offset] = 1
        rank_offset += 1
        
        # Term 6: (A10 - A00)(B00 + B01)
        # U: A10, A00
        # V: B00, B01
        # W: C11
        U[a_idx(1, 0)][rank_offset] = 1
        U[a_idx(0, 0)][rank_offset] = -1
        V[b_idx(0, 0)][rank_offset] = 1
        V[b_idx(0, 1)][rank_offset] = 1
        W[c_idx(1, 1)][rank_offset] = 1
        rank_offset += 1
        
        # Term 7: (A01 - A11)(B10 + B11)
        # U: A01, A11
        # V: B10, B11
        # W: C00
        U[a_idx(0, 1)][rank_offset] = 1
        U[a_idx(1, 1)][rank_offset] = -1
        V[b_idx(1, 0)][rank_offset] = 1
        V[b_idx(1, 1)][rank_offset] = 1
        W[c_idx(0, 0)][rank_offset] = 1
        rank_offset += 1
        
        # Term 8: A20 B00
        # U: A20
        # V: B00
        # W: C20
        U[a_idx(2, 0)][rank_offset] = 1
        V[b_idx(0, 0)][rank_offset] = 1
        W[c_idx(2, 0)][rank_offset] = 1
        rank_offset += 1
        
        # Term 9: A20 B01
        # U: A20
        # V: B01
        # W: C21
        U[a_idx(2, 0)][rank_offset] = 1
        V[b_idx(0, 1)][rank_offset] = 1
        W[c_idx(2, 1)][rank_offset] = 1
        rank_offset += 1
        
        # Term 10: A21 B10
        # U: A21
        # V: B10
        # W: C20
        U[a_idx(2, 1)][rank_offset] = 1
        V[b_idx(1, 0)][rank_offset] = 1
        W[c_idx(2, 0)][rank_offset] = 1
        rank_offset += 1
        
        # Term 11: A21 B11
        # U: A21
        # V: B11
        # W: C21
        U[a_idx(2, 1)][rank_offset] = 1
        V[b_idx(1, 1)][rank_offset] = 1
        W[c_idx(2, 1)][rank_offset] = 1
        rank_offset += 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
