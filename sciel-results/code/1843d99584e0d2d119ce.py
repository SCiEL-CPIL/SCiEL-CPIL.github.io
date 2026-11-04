"""Matrix multiplication 4x4 * 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-77 exact tensor decomposition for 4x4 by 4x6 matrix multiplication.
    
    Strategy:
    - Split B (4x6) into two 4x3 matrices: B_left (cols 0-2) and B_right (cols 3-5).
    - For A(4x4) * B_left(4x3): split B_left into 4x2 + 4x1.
      - A * B_4x2: use recursive Strassen on 4x4*4x2 (rank 28).
        Split A into 2x2 blocks, B_4x2 into 2x2 blocks (each 2x1).
        Apply Strassen at top level (7 steps), each step is 2x2*2x1.
        For 2x2*2x1: split into 2x2*2x1 = 2 columns, use Strassen again (7 terms each) = 14 per step.
        Wait, that gives 7*14 = 98. That's too much.
        
    Let me reconsider. The knowledge says:
    "Recursive Strassen for 4x4*4x4 (rank 49) plus 2x2 Strassen blocks for 4x4*4x2 (rank 28) gives rank 77"
    
    So the approach is:
    - For 4x4 * 4x6: split into 4x4*4x3 + 4x4*4x3.
    - For each 4x4*4x3: split into 4x4*4x2 + 4x4*4x1.
    - 4x4*4x2: rank 28 (using some decomposition)
    - 4x4*4x1: rank 7 (using Strassen-like or just 16 schoolbook? No, 7 is better)
    
    Actually, let me think about this differently.
    
    4x4 * 4x6:
    Split columns of B into groups:
    - Group 1: cols 0-3 (4x4)
    - Group 2: cols 4-5 (4x2)
    
    For 4x4 * 4x4: recursive Strassen gives rank 49.
    For 4x4 * 4x2: we need a decomposition.
    
    For 4x4 * 4x2:
    Split A into 2x2 blocks (4 blocks of 2x2).
    Split B (4x2) into 2x2 blocks (2 blocks of 2x2, each 2x1... no, 2x2 means 2 rows x 2 cols).
    B is 4x2, so blocks are 2x2: B00 (rows 0-1, cols 0-1), B10 (rows 2-3, cols 0-1).
    Wait, B is 4x2. If we split into 2x2 blocks: B00 (rows 0-1, cols 0-1), B10 (rows 2-3, cols 0-1).
    That's 2 blocks.
    
    C is 4x2. Blocks: C00 (rows 0-1, cols 0-1), C10 (rows 2-3, cols 0-1).
    
    Apply Strassen at the 2x2 block level:
    A has 4 blocks: A00, A01, A10, A11 (each 2x2)
    B has 2 blocks: B00, B10 (each 2x2) -- but wait, B should also be split into 2x2 blocks.
    B is 4x2. If we think of it as 2x2 blocks of size 2x1... that doesn't work for Strassen.
    
    Let me reconsider. For Strassen, both matrices need to be square and split into 4 equal sub-blocks.
    4x4 * 4x2 is not square, so standard Strassen doesn't directly apply.
    
    Alternative approach for 4x4 * 4x2:
    View as 4x4 * [4x2]. We can write:
    C = A * B where A is 4x4, B is 4x2.
    
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] each 2x2.
    Split B into 2x1 blocks: B = [[B0], [B1]] where B0 is 2x2, B1 is 2x2.
    Wait, B is 4x2. Split into two 2x2 blocks: B_top (rows 0-1, cols 0-1), B_bot (rows 2-3, cols 0-1).
    
    C = A * B = [[A00*B_top + A01*B_bot], [A10*B_top + A11*B_bot]]
    Each entry is a 2x2 matrix product.
    
    So we have 4 products of 2x2 * 2x2 matrices:
    C00 = A00*B_top + A01*B_bot
    C10 = A10*B_top + A11*B_bot
    
    For each 2x2*2x2 product, we can use Strassen (7 multiplications).
    But we have 4 such products, so 4*7 = 28 multiplications.
    
    This gives rank 28 for 4x4*4x2!
    
    For 4x4 * 4x1:
    Split A into 2x2 blocks.
    B is 4x1. Split into two 2x1 blocks: B_top (rows 0-1), B_bot (rows 2-3).
    C is 4x1. Split into two 2x1 blocks: C_top (rows 0-1), C_bot (rows 2-3).
    
    C_top = A00*B_top + A01*B_bot  (2x2 * 2x1 + 2x2 * 2x1)
    C_bot = A10*B_top + A11*B_bot
    
    For 2x2 * 2x1: we can use a 7-term decomposition? No, 2x2*2x1 is just 4 scalar multiplications.
    Actually, 2x2 * 2x1:
    [a b]   [x]   [ax+by]
    [c d] * [y] = [cx+dy]
    
    This is 4 scalar multiplications, or can it be done in fewer? No, the tensor rank of 2x2*2x1 is 4.
    
    So C_top requires 4+4 = 8 multiplications, C_bot requires 4+4 = 8 multiplications. Total 16.
    
    But wait, can we do better? Let's use the fact that we're adding two 2x2*2x1 products.
    C_top = A00*B_top + A01*B_bot
    
    This is equivalent to [A00 | A01] * [B_top; B_bot] which is 2x4 * 4x1.
    The rank of 2x4*4x1 matrix multiplication... 
    
    Actually, let me just count: for 4x4*4x1, we have 16 scalar multiplications in schoolbook.
    Can we reduce? 
    
    Hmm, let me reconsider the total.
    
    4x4 * 4x6 = 4x4*4x4 + 4x4*4x2
    - 4x4*4x4: rank 49 (recursive Strassen)
    - 4x4*4x2: rank 28
    
    Total: 49 + 28 = 77.
    
    This matches the knowledge! So the approach is:
    1. For the 4x4*4x4 part (cols 0-3 of B), use recursive Strassen (rank 49).
    2. For the 4x4*4x2 part (cols 4-5 of B), use the 2x2 Strassen block approach (rank 28).
    
    Total rank = 77.
    
    Now let me implement this.
    
    For 4x4*4x4 recursive Strassen:
    - Split A into 2x2 blocks (each 2x2).
    - Split B into 2x2 blocks (each 2x2).
    - Apply Strassen: 7 steps, each involving 2x2*2x2 products.
    - Each 2x2*2x2 product uses Strassen again: 7 scalar multiplications.
    - Total: 7 * 7 = 49.
    
    For 4x4*4x2:
    - Split A into 2x2 blocks (each 2x2).
    - Split B (4x2) into two 2x2 blocks.
    - C (4x2) has two 2x2 blocks.
    - C_top = A00*B_top + A01*B_bot
    - C_bot = A10*B_top + A11*B_bot
    - For each, use Strassen on the 2x2*2x2 products.
    - 4 Strassen decompositions, each rank 7, total 28.
    
    Let me implement this carefully.
    """
    start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4 * 4x4 (columns 0-3 of B) -> rank 49
    # Part 2: 4x4 * 4x2 (columns 4-5 of B) -> rank 28
    # Total rank = 77
    
    total_rank = 77
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Helper functions
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    # Strassen steps for 2x2 * 2x2
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
    # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
    # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
    
    # For each step, define the coefficients for A blocks, B blocks, C blocks
    # A blocks: 00, 01, 10, 11 (row_block, col_block) each 2x2
    # B blocks: 00, 01, 10, 11 (row_block, col_block) each 2x2
    # C blocks: 00, 01, 10, 11 (row_block, col_block) each 2x2
    
    strassen_steps = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict from (ri, ci) to coeff
        # b_coeffs: dict from (ri, ci) to coeff
        # c_coeffs: dict from (ri, ci) to coeff
        # ri, ci are 0 or 1 for block indices
        # Within each block, we iterate over local rows and cols
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    # Part 1: 4x4 * 4x4 recursive Strassen (rank 49)
    # Columns 0-3 of B
    # A is 4x4, split into 2x2 blocks of size 2x2
    # B is 4x4 (cols 0-3), split into 2x2 blocks of size 2x2
    # C is 4x4 (cols 0-3), split into 2x2 blocks of size 2x2
    
    col = 0
    for s in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_steps[s]
        # For each local position within the 2x2 blocks
        for ri_a in range(2):  # local row in A block
            for ci_a in range(2):  # local col in A block
                for ri_b in range(2):  # local row in B block
                    for ci_b in range(2):  # local col in B block
                        # This is one scalar multiplication
                        # U entry: A at position determined by a_coeffs
                        # V entry: B at position determined by b_coeffs
                        # W entry: C at position determined by c_coeffs
                        
                        u_val = 0.0
                        v_val = 0.0
                        w_val = 0.0
                        
                        for (bi, bj), coeff in a_coeffs.items():
                            i = bi * 2 + ri_a
                            j = bj * 2 + ci_a
                            u_val += coeff
                            # Store at U[i*m + j, col]
                        
                        for (bi, bj), coeff in b_coeffs.items():
                            i = bi * 2 + ri_b
                            j = bj * 2 + ci_b
                            v_val += coeff
                            # Store at V[i*p + j, col]
                        
                        for (bi, bj), coeff in c_coeffs.items():
                            i = bi * 2 + ri_a  # C row uses A's row index
                            j = bj * 2 + ci_b  # C col uses B's col index
                            w_val += coeff
                            # Store at W[j*n + i, col]
                        
                        # But we need to handle multiple blocks contributing to same U/V/W entries
                        # Let me restructure: for each (ri_a, ci_a, ri_b, ci_b), 
                        # accumulate contributions from all blocks in the coefficients
                        
                        col += 1
    
    # Wait, this approach is wrong. The Strassen decomposition works at the block level.
    # Each Strassen step involves products of 2x2 blocks. The tensor decomposition
    # should have one term per scalar multiplication within the block products.
    
    # Let me restart with a clearer approach.
    
    # For 4x4*4x4 recursive Strassen:
    # Top level: 7 Strassen steps.
    # Each step: product of (2x2 linear combo of A blocks) * (2x2 linear combo of B blocks)
    # = 2x2 * 2x2 matrix product
    # Each 2x2*2x2 product: 7 scalar multiplications (Strassen again)
    # Total: 7 * 7 = 49 terms.
    
    # Each term corresponds to a specific scalar multiplication.
    # For a given top-level step s and inner Strassen step t:
    # The U vector has non-zero entries at positions corresponding to A elements
    # The V vector has non-zero entries at positions corresponding to B elements  
    # The W vector has non-zero entries at positions corresponding to C elements
    
    # Let me define the decomposition more explicitly.
    
    # Reset
    total_rank = 77
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # For 4x4*4x4 (cols 0-3):
    # Top Strassen step s (0..6), inner Strassen step t (0..6)
    # For each (s, t), we have one rank-1 term.
    
    # The top-level Strassen defines linear combinations of 2x2 blocks.
    # The inner Strassen defines linear combinations of individual elements within those 2x2 blocks.
    
    # Let me define:
    # top_steps[s] = (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    #   where a_block_coeffs maps block (bi,bj) to coefficient
    # inner_steps[t] = (a_elem_coeffs, b_elem_coeffs, c_elem_coeffs)
    #   where a_elem_coeffs maps element (ri,ci) to coefficient within a 2x2 block
    
    top_steps = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    inner_steps = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for s in range(7):
        a_bc, b_bc, c_bc = top_steps[s]
        for t in range(7):
            a_ec, b_ec, c_ec = inner_steps[t]
            
            # U: for each A block (bi,bj) in a_bc, for each element (ri,ci) in a_ec:
            for (bi, bj), bc_val in a_bc.items():
                for (ri, ci), ec_val in a_ec.items():
                    i = bi * 2 + ri
                    j = bj * 2 + ci
                    U[i * m + j, col] += bc_val * ec_val
            
            # V: for each B block (bi,bj) in b_bc, for each element (ri,ci) in b_ec:
            for (bi, bj), bc_val in b_bc.items():
                for (ri, ci), ec_val in b_ec.items():
                    i = bi * 2 + ri
                    j = bj * 2 + ci  # j is 0-3 for cols 0-3
                    V[i * p + j, col] += bc_val * ec_val
            
            # W: for each C block (bi,bj) in c_bc, for each element (ri,ci) in c_ec:
            for (bi, bj), bc_val in c_bc.items():
                for (ri, ci), ec_val in c_ec.items():
                    i = bi * 2 + ri
                    j = bj * 2 + ci  # j is 0-3 for cols 0-3
                    W[j * n + i, col] += bc_val * ec_val
            
            col += 1
    
    # col should be 49 now
    
    # Part 2: 4x4 * 4x2 (cols 4-5 of B)
    # A is 4x4, B is 4x2 (cols 4-5)
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B into 2x2 blocks: B_top (rows 0-1, cols 4-5), B_bot (rows 2-3, cols 4-5)
    # C is 4x2: C_top (rows 0-1, cols 4-5), C_bot (rows 2-3, cols 4-5)
    # 
    # C_top = A00 * B_top + A01 * B_bot
    # C_bot = A10 * B_top + A11 * B_bot
    #
    # For each 2x2*2x2 product, use Strassen (7 terms)
    # 4 products * 7 terms = 28 terms
    
    # The four products are:
    # 1. A00 * B_top -> contributes to C_top
    # 2. A01 * B_bot -> contributes to C_top
    # 3. A10 * B_top -> contributes to C_bot
    # 4. A11 * B_bot -> contributes to C_bot
    
    # For product 1: A00 * B_top
    # A00 is rows 0-1, cols 0-1 of A
    # B_top is rows 0-1, cols 4-5 of B
    # C_top is rows 0-1, cols 4-5 of C
    
    for prod_idx, (a_block, b_block, c_block) in enumerate([
        ((0,0), (0,0), (0,0)),  # A00 * B_top -> C_top
        ((0,1), (1,0), (0,0)),  # A01 * B_bot -> C_top
        ((1,0), (0,0), (1,0)),  # A10 * B_top -> C_bot
        ((1,1), (1,0), (1,0)),  # A11 * B_bot -> C_bot
    ]):
        a_bi, a_bj = a_block
        b_bi, b_bj = b_block
        c_bi, c_bj = c_block
        
        for t in range(7):
            a_ec, b_ec, c_ec = inner_steps[t]
            
            # U: A block (a_bi, a_bj)
            for (ri, ci), ec_val in a_ec.items():
                i = a_bi * 2 + ri
                j = a_bj * 2 + ci
                U[i * m + j, col] += ec_val
            
            # V: B block (b_bi, b_bj) -> cols 4-5
            for (ri, ci), ec_val in b_ec.items():
                i = b_bi * 2 + ri
                j = b_bj * 2 + ci + 4  # offset by 4 for cols 4-5
                V[i * p + j, col] += ec_val
            
            # W: C block (c_bi, c_bj) -> cols 4-5
            for (ri, ci), ec_val in c_ec.items():
                i = c_bi * 2 + ri
                j = c_bj * 2 + ci + 4  # offset by 4 for cols 4-5
                W[j * n + i, col] += ec_val
            
            col += 1
    
    # col should be 49 + 28 = 77
    
    if time.time() - start > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
