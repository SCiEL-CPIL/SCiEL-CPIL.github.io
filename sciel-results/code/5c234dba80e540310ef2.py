import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We will construct a decomposition of rank 72.
    # Part 1: 4x4x4 part using Strassen (rank 56)
    # Part 2: 4x4x1 part using schoolbook (rank 16)
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2x2
    # a_idx: indices of A in the 2x2 block
    # b_idx: indices of B in the 2x2 block
    # c_idx: indices of C in the 2x2 block
    # global offset for A, B, C indices
    
    def add_strassen_block(a_base, b_base, c_base):
        nonlocal col
        # a_base is (row, col) offset in 4x4 A
        # b_base is (row, col) offset in 4x4 B (part 1)
        # c_base is (row, col) offset in 4x4 C (part 1)
        
        # Local indices 0,1 for rows/cols of the 2x2 block
        # A indices: (a_base[0]+r, a_base[1]+c)
        # B indices: (b_base[0]+r, b_base[1]+c)
        # C indices: (c_base[0]+r, c_base[1]+c)
        
        # Strassen factors for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the 7 products
        # Each product is a sum of 4 terms (2x2x2 rank 4 schoolbook)
        # But we combine them into 7 rank-1 terms for the whole 2x2x2 block?
        # No, Strassen reduces the 8 scalar multiplications to 7.
        # The "scalar" multiplications are actually the rank-1 terms in the tensor decomposition.
        
        # Let's define the 7 rank-1 terms explicitly.
        # Term 1: (A00+A11) * (B00+B11) -> contributes to C00, C11
        # Term 2: (A10+A11) * B00 -> contributes to C10, C11
        # Term 3: A00 * (B01-B11) -> contributes to C01, C11
        # Term 4: A11 * (B10-B00) -> contributes to C00, C10
        # Term 5: (A00+A01) * B11 -> contributes to C00, C01
        # Term 6: (A10-A00) * (B00+B01) -> contributes to C11
        # Term 7: (A01-A11) * (B10+B11) -> contributes to C00
        
        # We need to map these to U, V, W vectors.
        
        # A indices in global A (4x4):
        # A00: (a_base[0], a_base[1])
        # A01: (a_base[0], a_base[1]+1)
        # A10: (a_base[0]+1, a_base[1])
        # A11: (a_base[0]+1, a_base[1]+1)
        
        # B indices in global B (4x4 part, cols 0-3):
        # B00: (b_base[0], b_base[1])
        # B01: (b_base[0], b_base[1]+1)
        # B10: (b_base[0]+1, b_base[1])
        # B11: (b_base[0]+1, b_base[1]+1)
        
        # C indices in global C (4x4 part, cols 0-3):
        # C00: (c_base[0], c_base[1])
        # C01: (c_base[0], c_base[1]+1)
        # C10: (c_base[0]+1, c_base[1])
        # C11: (c_base[0]+1, c_base[1]+1)
        
        # Helper to get global index
        def a_idx(r, c): return (a_base[0]+r)*4 + (a_base[1]+c)
        def b_idx(r, c): return (b_base[0]+r)*5 + (b_base[1]+c) # Note: B is 4x5, so col stride is 5
        def c_idx(r, c): return (c_base[1]+c)*4 + (c_base[0]+r) # W[k*n+i, r], so row is k*4+i. Here c_base is (row, col) of C block.
        
        # Wait, C is n x p = 4 x 5.
        # The 4x4 part of C corresponds to cols 0-3.
        # So c_base[1] is the column offset in C (0 or 2? No, 0 or 2 for 2x2 blocks?
        # A is 4x4, B is 4x4.
        # Blocks are 2x2.
        # A blocks: (0,0), (0,1), (1,0), (1,1).
        # B blocks: (0,0), (0,1), (1,0), (1,1).
        # C blocks: (0,0), (0,1), (1,0), (1,1).
        
        # Let's iterate over the 8 products needed for 4x4x4.
        
        pass # This logic is inside the loop below.

    # List of (A_block_row, A_block_col, B_block_row, B_block_col, C_block_row, C_block_col)
    # C_rc = sum_k A_rk B_kc
    products = [
        (0, 0, 0, 0, 0, 0), # A00 B00 -> C00
        (0, 1, 1, 0, 0, 0), # A01 B10 -> C00
        (1, 0, 0, 0, 1, 0), # A10 B00 -> C10
        (1, 1, 1, 0, 1, 0), # A11 B10 -> C10
        (0, 0, 0, 1, 0, 1), # A00 B01 -> C01
        (0, 1, 1, 1, 0, 1), # A01 B11 -> C01
        (1, 0, 0, 1, 1, 1), # A10 B01 -> C11
        (1, 1, 1, 1, 1, 1), # A11 B11 -> C11
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        # Add Strassen for this 2x2x2 product
        # a_base = (ar*2, ac*2)
        # b_base = (br*2, bc*2)
        # c_base = (cr*2, cc*2)
        
        a_base = (ar*2, ac*2)
        b_base = (br*2, bc*2)
        c_base = (cr*2, cc*2)
        
        def a_idx(r, c): return (a_base[0]+r)*4 + (a_base[1]+c)
        def b_idx(r, c): return (b_base[0]+r)*5 + (b_base[1]+c)
        def c_idx(r, c): return (c_base[1]+c)*4 + (c_base[0]+r)
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11
        U[a_idx(0,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[a_idx(1,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        W[c_idx(1,1)][col] -= 1 # Wait, C11 = M1 - M2 ...
        # Let's check C11 formula: C11 = M1 - M2 + M3 + M6
        # So W[c_idx(1,1)] should be -1 for M2.
        # And W[c_idx(1,0)] = M2 + M4. So +1.
        col += 1
        
        # M3: A00(B01-B11)
        U[a_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] -= 1
        W[c_idx(0,1)][col] += 1 # C01 = M3 + M5
        W[c_idx(1,1)][col] += 1 # C11 = ... + M3
        col += 1
        
        # M4: A11(B10-B00)
        U[a_idx(1,1)][col] += 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(0,0)][col] -= 1
        W[c_idx(0,0)][col] += 1 # C00 = M1 + M4 ...
        W[c_idx(1,0)][col] += 1 # C10 = M2 + M4
        col += 1
        
        # M5: (A00+A01)B11
        U[a_idx(0,0)][col] += 1
        U[a_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] -= 1 # C00 = ... - M5
        W[c_idx(0,1)][col] += 1 # C01 = ... + M5
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[a_idx(1,0)][col] += 1
        U[a_idx(0,0)][col] -= 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1 # C11 = ... + M6
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[a_idx(0,1)][col] += 1
        U[a_idx(1,1)][col] -= 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1 # C00 = ... + M7
        col += 1

    # Part 2: 4x4x1 (rank 16)
    # A (4x4) * B_2 (4x1) -> C_2 (4x1)
    # B_2 is column 4 of B.
    # C_2 is column 4 of C.
    
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1 # B index: row j, col 4
            W[4*4 + i][col] = 1 # C index: k=4, i. W[k*n+i] = W[4*4+i]
            col += 1

    return {"U": U, "V": V, "W": W}
