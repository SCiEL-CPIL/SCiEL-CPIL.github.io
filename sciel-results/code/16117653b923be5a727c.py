"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C11, C22 get +1
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1 # C21, C22 get +1
    
    # M3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1 # C12 +1, C22 -1
    
    # M4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = -1; W2[2, 3] = 1 # C11 -1, C21 +1
    
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = -1 # C12 +1, C22 -1
    
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = 1 # C11 +1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11 +1
    
    # Check W2
    # C11: M1 - M4 + M6 + M7 -> 1 - (-1) + 1 + 1 = 4? No.
    # C11 = M1 - M4 + M6 + M7
    # W2[0,0]=1, W2[0,3]=-1, W2[0,5]=1, W2[0,6]=1.
    # Sum for C11: 1*M1 - 1*M4 + 1*M6 + 1*M7. Correct.
    
    # C12 = M3 + M5
    # W2[1,2]=1, W2[1,4]=1. Correct.
    
    # C21 = M2 + M4
    # W2[2,1]=1, W2[2,3]=1. Correct.
    
    # C22 = M1 - M3 + M2 - M5
    # W2[3,0]=1, W2[3,2]=-1, W2[3,1]=1, W2[3,4]=-1. Correct.

    # Construct 4x4x4 decomposition (rank 49)
    # We use the outer Strassen structure, where each scalar is replaced by a 2x2x2 Strassen decomposition.
    # U4 will be 16x49, V4 16x49, W4 16x49.
    # Indices:
    # Outer Strassen index k (0..6)
    # Inner Strassen index l (0..6)
    # Total index col = k * 7 + l
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Map outer indices to inner blocks
    # Outer A blocks: 0:A11, 1:A12, 2:A21, 3:A22
    # Outer B blocks: 0:B11, 1:B12, 2:B21, 3:B22
    # Outer C blocks: 0:C11, 1:C12, 2:C21, 3:C22
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            # U4 contribution
            for i in range(4):
                if U2[i, k] != 0:
                    # i is outer block index
                    # Map to 16 rows: block i has 4 rows
                    # Row offset = i * 4
                    # Inner row offset = l_row (from U2[:, l])
                    for r in range(4):
                        if U2[r, l] != 0:
                            U4[i * 4 + r, col] += U2[i, k] * U2[r, l]
            
            # V4 contribution
            for j in range(4):
                if V2[j, k] != 0:
                    for r in range(4):
                        if V2[r, l] != 0:
                            V4[j * 4 + r, col] += V2[j, k] * V2[r, l]
            
            # W4 contribution
            for c in range(4):
                if W2[c, k] != 0:
                    for r in range(4):
                        if W2[r, l] != 0:
                            W4[c * 4 + r, col] += W2[c, k] * W2[r, l]

    # Construct 4x4x2 decomposition (rank 28)
    # We need to compute C[:, 4:6]
    # This involves A (4x4) and B[:, 4:6] (4x2)
    # Split A into 4 2x2 blocks, B into 2 2x2 blocks (vertically)
    # Products:
    # 1. A11 * B_top (B[0:2, 4:6]) -> contributes to C_top
    # 2. A12 * B_bot (B[2:4, 4:6]) -> contributes to C_top
    # 3. A21 * B_top -> contributes to C_bot
    # 4. A22 * B_bot -> contributes to C_bot
    
    # We use Strassen for each 2x2x2 product.
    # Note: The B block here is 2x2.
    # In our U2/V2/W2, B is 2x2.
    # So we can reuse the same U2, V2, W2 logic.
    
    U_tail = np.zeros((16, 28), dtype=int)
    V_tail = np.zeros((24, 28), dtype=int) # V is m*p = 4*6 = 24
    W_tail = np.zeros((24, 28), dtype=int) # W is p*n = 6*4 = 24
    
    # We have 4 products, each with 7 multiplications.
    # Product 0: A11 * B_top
    # Product 1: A12 * B_bot
    # Product 2: A21 * B_top
    # Product 3: A22 * B_bot
    
    # A indices:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B indices (for the tail part, cols 4-5):
    # B_top: rows 0-1, cols 4-5
    # B_bot: rows 2-3, cols 4-5
    
    # C indices (for the tail part, cols 4-5):
    # C_top: rows 0-1, cols 4-5
    # C_bot: rows 2-3, cols 4-5
    
    products = [
        (0, 0), # A11, B_top -> C_top
        (1, 2), # A12, B_bot -> C_top
        (2, 0), # A21, B_top -> C_bot
        (3, 2), # A22, B_bot -> C_bot
    ]
    
    # Map block index to row/col offset in the full matrices
    # A is 4x4. Block (br, bc) is at row br*2, col bc*2.
    # B is 4x6. Block (br, bc) is at row br*2, col bc*2.
    # C is 4x6. Block (br, bc) is at row br*2, col bc*2.
    
    # For the tail, B and C blocks are at col offset 2 (since 4/2=2).
    # B_top is block (0, 2). B_bot is block (1, 2).
    # C_top is block (0, 2). C_bot is block (1, 2).
    
    for p_idx, (a_block, b_block) in enumerate(products):
        # a_block: 0,1,2,3
        # b_block: 0 (top), 2 (bot) -- wait, my products list has b_block as 0 or 2?
        # Let's clarify:
        # B_top is rows 0-1. In block index, it's br=0.
        # B_bot is rows 2-3. In block index, it's br=1.
        # But in the products list above, I used 0 and 2. Let's fix.
        # Product 0: A11 (br=0, bc=0) * B_top (br=0, bc=2) -> C_top (br=0, bc=2)
        # Product 1: A12 (br=0, bc=1) * B_bot (br=1, bc=2) -> C_top (br=0, bc=2)
        # Product 2: A21 (br=1, bc=0) * B_top (br=0, bc=2) -> C_bot (br=1, bc=2)
        # Product 3: A22 (br=1, bc=1) * B_bot (br=1, bc=2) -> C_bot (br=1, bc=2)
        
        a_br = a_block // 2
        a_bc = a_block % 2
        b_br = b_block // 2 # This logic is flawed in my list.
        
        # Let's redefine the products clearly
        pass

    # Let's rewrite the tail generation more carefully
    U_tail = np.zeros((16, 28), dtype=int)
    V_tail = np.zeros((24, 28), dtype=int)
    W_tail = np.zeros((24, 28), dtype=int)
    
    # Product definitions: (a_block_idx, b_block_idx, c_block_idx)
    # a_block_idx: 0..3 (br, bc)
    # b_block_idx: 0..1 (br, bc=2) -> actually just br, since bc is fixed at 2 for tail
    # c_block_idx: 0..1 (br, bc=2)
    
    # A blocks:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # B blocks (tail):
    # 0: (0,2), 1: (1,2)
    # C blocks (tail):
    # 0: (0,2), 1: (1,2)
    
    tail_products = [
        (0, 0, 0), # A11 * B_top -> C_top
        (1, 1, 0), # A12 * B_bot -> C_top
        (2, 0, 1), # A21 * B_top -> C_bot
        (3, 1, 1), # A22 * B_bot -> C_bot
    ]
    
    for p_idx, (a_idx, b_idx, c_idx) in enumerate(tail_products):
        # a_idx -> a_br, a_bc
        a_br = a_idx // 2
        a_bc = a_idx % 2
        
        # b_idx -> b_br (b_bc is 2)
        b_br = b_idx
        b_bc = 2
        
        # c_idx -> c_br (c_bc is 2)
        c_br = c_idx
        c_bc = 2
        
        for l in range(7):
            col = p_idx * 7 + l
            
            # U_tail
            for r in range(4):
                if U2[r, l] != 0:
                    # Map inner row r to full A row
                    # A block (a_br, a_bc) starts at row a_br*2, col a_bc*2
                    # Inner row r corresponds to local row r
                    # So global row = a_br*2 + r
                    # Global col = a_bc*2 + (r // 2) ? No.
                    # A is 4x4. U2 is 4x7.
                    # U2 row 0 -> A11 (local 0,0)
                    # U2 row 1 -> A12 (local 0,1)
                    # U2 row 2 -> A21 (local 1,0)
                    # U2 row 3 -> A22 (local 1,1)
                    # This mapping is NOT row-wise.
                    # U2 index 0 corresponds to A[0,0], A[0,1], A[1,0], A[1,1]?
                    # No, U2 is 4x7. The 4 rows correspond to the 4 elements of the 2x2 matrix.
                    # Row 0: A11
                    # Row 1: A12
                    # Row 2: A21
                    # Row 3: A22
                    # But A11 is a 2x2 matrix? No, in the 2x2x2 Strassen, A is 2x2.
                    # So U2 row 0 is A[0,0], U2 row 1 is A[0,1], U2 row 2 is A[1,0], U2 row 3 is A[1,1].
                    # Wait, in my U2 definition:
                    # U2[0,0]=1 (A11), U2[3,0]=1 (A22).
                    # This implies U2 row 0 is A[0,0], row 1 is A[0,1], row 2 is A[1,0], row 3 is A[1,1].
                    # Let's verify.
                    # M1 = (A11+A22)...
                    # A11 is A[0,0]. A22 is A[1,1].
                    # So U2[0] is A[0,0], U2[3] is A[1,1].
                    # U2[1] is A[0,1], U2[2] is A[1,0].
                    
                    # So for a 2x2 block at (a_br, a_bc) in the 4x4 matrix:
                    # Local (0,0) -> Global (a_br*2, a_bc*2)
                    # Local (0,1) -> Global (a_br*2, a_bc*2 + 1)
                    # Local (1,0) -> Global (a_br*2 + 1, a_bc*2)
                    # Local (1,1) -> Global (a_br*2 + 1, a_bc*2 + 1)
                    
                    # U2 row r corresponds to local index r.
                    # Global row = a_br*2 + (r // 2) ? No.
                    # r=0 -> (0,0)
                    # r=1 -> (0,1)
                    # r=2 -> (1,0)
                    # r=3 -> (1,1)
                    # Global row = a_br*2 + (r % 2) ? No.
                    # r=0 -> row 0
                    # r=1 -> row 0
                    # r=2 -> row 1
                    # r=3 -> row 1
                    # So Global row = a_br*2 + (r // 2)
                    # Global col = a_bc*2 + (r % 2)
                    
                    g_row = a_br * 2 + (r // 2)
                    g_col = a_bc * 2 + (r % 2)
                    U_tail[g_row * 4 + g_col, col] += U2[r, l]
            
            # V_tail
            for r in range(4):
                if V2[r, l] != 0:
                    g_row = b_br * 2 + (r // 2)
                    g_col = b_bc * 2 + (r % 2)
                    V_tail[g_row * 6 + g_col, col] += V2[r, l]
            
            # W_tail
            for r in range(4):
                if W2[r, l] != 0:
                    g_row = c_br * 2 + (r // 2)
                    g_col = c_bc * 2 + (r % 2)
                    W_tail[g_col * 4 + g_row, col] += W2[r, l] # W index is k*n+i -> col*n + row

    # Combine U4, V4, W4 with U_tail, V_tail, W_tail
    # U is 16x77
    # V is 24x77
    # W is 24x77
    
    U = np.zeros((16, 77), dtype=int)
    V = np.zeros((24, 77), dtype=int)
    W = np.zeros((24, 77), dtype=int)
    
    # First 49 columns
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    
    # Last 28 columns
    U[:, 49:] = U_tail
    V[:, 49:] = V_tail
    W[:, 49:] = W_tail
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
