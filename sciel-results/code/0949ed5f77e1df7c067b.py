"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 90.
    # We use a block decomposition to reduce rank.
    # Split m=5 into m1=2, m2=3.
    # A = [A1 (3x2), A2 (3x3)]
    # B = [B1 (2x6), B2 (3x6)]
    # C = A1 B1 + A2 B2
    
    # Part 1: A2 B2 (3x3 * 3x6)
    # Split B2 into two 3x3 blocks: B21, B22
    # Rank = 27 + 27 = 54
    
    # Part 2: A1 B1 (3x2 * 2x6)
    # Split B1 into three 2x2 blocks: B11, B12, B13
    # Each block is 3x2 * 2x2.
    # We use Rank 11 for each 3x2x2 block:
    #   - Row 1 (1x2 * 2x2) uses Rank 4 (schoolbook)
    #   - Rows 2-3 (2x2 * 2x2) uses Rank 7 (Strassen)
    # Rank per block = 4 + 7 = 11.
    # Total for Part 2 = 3 * 11 = 33.
    
    # Total Rank = 54 + 33 = 87.
    
    rank_333 = 27
    rank_322_schoolbook = 4 # 1x2x2 is 4
    rank_222_strassen = 7
    rank_322_optimized = rank_322_schoolbook + rank_222_strassen # 11
    
    total_rank = 2 * rank_333 + 3 * rank_322_optimized
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # --- Part 1: A2 B2 (Rows 2-3 of A? No, A is 3x5. A2 is 3x3 part) ---
    # A indices: 0..2 (rows), 0..4 (cols)
    # A2 corresponds to cols 2,3,4 of A (if we split 5 into 2+3, A1 is cols 0,1, A2 is cols 2,3,4)
    # Wait, standard split: A = [A1, A2]. A1 is 3x2 (cols 0,1). A2 is 3x3 (cols 2,3,4).
    # B = [B1; B2]. B1 is 2x6 (rows 0,1). B2 is 3x6 (rows 2,3,4).
    
    # B2 is split into B21 (cols 0,1,2) and B22 (cols 3,4,5)
    # B21 is 3x3, B22 is 3x3.
    
    # A2 B21: 3x3 * 3x3 (Schoolbook)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A2 row i, col j (global col j+2)
                # B21 row j (global row j+2), col k (global col k)
                add_term(i * 5 + (j + 2), 1,
                         (j + 2) * 6 + k, 1,
                         k * 3 + i, 1)
    
    # A2 B22: 3x3 * 3x3 (Schoolbook)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A2 row i, col j (global col j+2)
                # B22 row j (global row j+2), col k (global col k+3)
                add_term(i * 5 + (j + 2), 1,
                         (j + 2) * 6 + (k + 3), 1,
                         (k + 3) * 3 + i, 1)

    # --- Part 2: A1 B1 (3x2 * 2x6) ---
    # A1 is cols 0,1 of A.
    # B1 is rows 0,1 of B.
    # B1 split into B11 (cols 0,1), B12 (cols 2,3), B13 (cols 4,5).
    
    for b_idx in range(3):
        # b_idx 0 -> cols 0,1; 1 -> 2,3; 2 -> 4,5
        col_offset = b_idx * 2
        
        # Sub-part 2a: Row 1 of A1 (global row 0) * B1_block (2x2)
        # A1 row 0 is [A[0,0], A[0,1]]
        # B1_block is [[B[0,c0], B[0,c1]], [B[1,c0], B[1,c1]]]
        # Result Z[0, c0..c1]
        
        # Rank 4 Schoolbook for 1x2 * 2x2
        # z00 = x0 y00 + x1 y10
        # z01 = x0 y01 + x1 y11
        # Terms:
        # 1: x0 y00
        add_term(0 * 5 + 0, 1,
                 0 * 6 + col_offset, 1,
                 col_offset * 3 + 0, 1)
        # 2: x1 y10
        add_term(0 * 5 + 1, 1,
                 1 * 6 + col_offset, 1,
                 col_offset * 3 + 0, 1)
        # 3: x0 y01
        add_term(0 * 5 + 0, 1,
                 0 * 6 + (col_offset + 1), 1,
                 (col_offset + 1) * 3 + 0, 1)
        # 4: x1 y11
        add_term(0 * 5 + 1, 1,
                 1 * 6 + (col_offset + 1), 1,
                 (col_offset + 1) * 3 + 0, 1)

        # Sub-part 2b: Rows 2-3 of A1 (global rows 1,2) * B1_block (2x2)
        # A1_sub is 2x2. B1_block is 2x2.
        # Use Strassen Rank 7.
        
        # Strassen for 2x2:
        # m1 = (A00) * (B00 + B11) -> A[1,0] * (B[0,c0] + B[1,c1])? No.
        # Standard Strassen indices for A (2x2), B (2x2):
        # m1 = A00 * (B00 + B11)
        # m2 = (A00 + A01) * B11
        # m3 = (A10 + A11) * B00
        # m4 = A11 * (B00 - B10)
        # m5 = (A00 + A10) * (B11 - B10)
        # m6 = (A01 - A10) * (B00 + B10)
        # m7 = (A11 - A01) * (B01 + B11)
        
        # Our A_sub:
        # A00 = A[1,0], A01 = A[1,1]
        # A10 = A[2,0], A11 = A[2,1]
        
        # Our B_block:
        # B00 = B[0,c0], B01 = B[0,c1]
        # B10 = B[1,c0], B11 = B[1,c1]
        
        # Output Z_sub (2x2):
        # Z00 = m1 - m5 + m6 + m7
        # Z01 = m4 + m5
        # Z10 = m2 - m4 + m6
        # Z11 = m1 - m2 + m3 + m7
        
        # We need to add these terms to U, V, W.
        # Note: U index for A1_sub rows:
        # Row 1 of A1 is global row 1.
        # Row 2 of A1 is global row 2.
        
        # Term 1: A[1,0] * (B[0,c0] + B[1,c1])
        # U: A[1,0] -> idx 1*5+0 = 5. Val 1.
        # V: B[0,c0] -> idx 0*6+c0. Val 1.
        # V: B[1,c1] -> idx 1*6+c1. Val 1.
        # W: Z00 (global row 1, col c0) -> idx c0*3+1. Val 1.
        # W: Z11 (global row 2, col c1) -> idx c1*3+2. Val 1.
        # Wait, m1 contributes to Z00 and Z11.
        # Z00 is output row 1, col c0.
        # Z11 is output row 2, col c1.
        
        # We can't add two W indices in one term.
        # Strassen terms are scalar multiplications.
        # We need to distribute the contributions.
        
        # Let's just use the 7 rank-1 tensors of Strassen.
        # Each tensor is u * v * w.
        
        # m1 = A00 * (B00 + B11)
        # u: A[1,0] (1)
        # v: B[0,c0] (1) + B[1,c1] (1)
        # w: Z[1,c0] (1) + Z[2,c1] (1) -> NO, w must be a vector in output space.
        # The output of m1 is added to Z00 and Z11.
        # So the rank-1 term is:
        # u_vec = [0, 1, 0, 0, 0, 0] (only A[1,0] active)
        # v_vec = [0, 1, 0, 0, 0, 1, ...] (B[0,c0] and B[1,c1] active)
        # w_vec = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, ...] (Z[1,c0] and Z[2,c1] active)
        # This is a single rank-1 term in the full tensor space.
        
        def add_strassen_term(u_indices, v_indices, w_indices):
            nonlocal col
            for ui in u_indices:
                U[ui][col] = 1
            for vi in v_indices:
                V[vi][col] = 1
            for wi in w_indices:
                W[wi][col] = 1
            col += 1

        # m1
        add_strassen_term([1*5+0], [0*6+col_offset, 1*6+(col_offset+1)], [col_offset*3+1, (col_offset+1)*3+2])
        
        # m2 = (A00+A01) * B11
        # A: A[1,0], A[1,1]
        # B: B[1,c1]
        # Z: Z[2,c0] (Z10)
        # Wait, m2 contributes to Z10.
        # Z10 is row 2, col c0.
        add_strassen_term([1*5+0, 1*5+1], [1*6+(col_offset+1)], [col_offset*3+2])
        
        # m3 = (A10+A11) * B00
        # A: A[2,0], A[2,1]
        # B: B[0,c0]
        # Z: Z[2,c1] (Z11)
        add_strassen_term([2*5+0, 2*5+1], [0*6+col_offset], [(col_offset+1)*3+2])
        
        # m4 = A11 * (B00 - B10)
        # A: A[2,1]
        # B: B[0,c0] (1), B[1,c0] (-1)
        # Z: Z[1,c1] (Z01), Z[2,c0] (Z10)
        add_strassen_term([2*5+1], [0*6+col_offset, 1*6+col_offset], [(col_offset+1)*3+1, col_offset*3+2])
        # Handle -1 in V
        V[1*6+col_offset][col-1] = -1
        
        # m5 = (A00+A10) * (B11 - B10)
        # A: A[1,0], A[2,0]
        # B: B[1,c1] (1), B[1,c0] (-1)
        # Z: Z[1,c0] (Z00, -1), Z[1,c1] (Z01, 1)
        add_strassen_term([1*5+0, 2*5+0], [1*6+(col_offset+1), 1*6+col_offset], [col_offset*3+1, (col_offset+1)*3+1])
        V[1*6+col_offset][col-1] = -1
        W[col_offset*3+1][col-1] = -1
        
        # m6 = (A01-A10) * (B00+B10)
        # A: A[1,1] (1), A[2,0] (-1)
        # B: B[0,c0] (1), B[1,c0] (1)
        # Z: Z[1,c0] (Z00, 1), Z[2,c0] (Z10, 1)
        add_strassen_term([1*5+1, 2*5+0], [0*6+col_offset, 1*6+col_offset], [col_offset*3+1, col_offset*3+2])
        U[2*5+0][col-1] = -1
        
        # m7 = (A11-A01) * (B01+B11)
        # A: A[2,1] (1), A[1,1] (-1)
        # B: B[0,c1] (1), B[1,c1] (1)
        # Z: Z[1,c0] (Z00, 1), Z[2,c1] (Z11, 1)
        add_strassen_term([2*5+1, 1*5+1], [0*6+(col_offset+1), 1*6+(col_offset+1)], [col_offset*3+1, (col_offset+1)*3+2])
        U[1*5+1][col-1] = -1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
