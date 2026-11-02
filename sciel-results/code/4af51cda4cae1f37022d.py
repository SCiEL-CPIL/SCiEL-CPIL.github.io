import time
import numpy as np

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U row: i*2 + j (i,j in 0..1)
    V row: j*2 + k (j,k in 0..1)
    W row: k*2 + i (k,i in 0..1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen products
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    # Wait, W indices are k*2+i.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map W rows:
    # Row 0: k=0, i=0 -> C00
    # Row 1: k=0, i=1 -> C10 (which is C21 in 1-based? No, i is row index of C)
    # C is 2x2. C[i, k]? No, standard is C[i, j] = sum_k A[i, k] B[k, j].
    # Here V index is j*2+k? No, V index is j*p+k. For p=2, j*2+k.
    # W index is k*n+i. For n=2, k*2+i.
    # So W[0] is k=0, i=0 -> C[0,0].
    # W[1] is k=0, i=1 -> C[1,0].
    # W[2] is k=1, i=0 -> C[0,1].
    # W[3] is k=1, i=1 -> C[1,1].
    
    # M1 contributes to C00, C11?
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # C10 = M2 + M4
    # C01 = M3 + M5
    
    # So:
    # W[0, 0] = 1 (M1)
    # W[3, 0] = 1 (M1)
    # W[1, 0] = 0
    # W[2, 0] = 0
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C10
    W[0, 1] = 1 # C00
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[1, 3] = 1 # C10
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1 # C01
    W[0, 4] = -1 # C00
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    W[3, 6] = 1 # C11
    
    return U, V, W

def _strassen_4x4_factors():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    """
    U2, V2, W2 = _strassen_2x2_factors()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # We have 7 Strassen products for the 2x2 block structure.
    # Each product is a 2x2x2 multiplication.
    # The inputs to these 2x2x2 multiplications are linear combinations of the 2x2 blocks of A and B.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the 2x2x2 factors to the 4x4x4 factors.
    # U4 rows are i*4+j.
    # V4 rows are j*4+k.
    # W4 rows are k*4+i.
    
    # For each of the 7 products, we add the 2x2x2 factors scaled and shifted.
    
    # Product 1: M1
    # A part: A00 + A11. Rows 0-3 (A00) and 8-11 (A11).
    # B part: B00 + B11. Rows 0-3 (B00) and 8-11 (B11).
    # C part: C00 + C11? No, M1 contributes to C00 and C11.
    # In 2x2x2, W[0] is C00, W[3] is C11.
    # In 4x4x4, C00 is rows 0-3, C11 is rows 8-11.
    
    for r in range(7):
        # U contribution
        for i2 in range(2):
            for j2 in range(2):
                row_2x2 = i2 * 2 + j2
                val = U2[row_2x2, r]
                if val != 0:
                    # Map to 4x4 blocks
                    if r == 0: # M1: A00 + A11
                        # A00 is rows 0-3, A11 is rows 8-11
                        # i2=0, j2=0 -> A00[0,0] -> 4x4 row 0
                        # i2=0, j2=1 -> A00[0,1] -> 4x4 row 1
                        # i2=1, j2=0 -> A00[1,0] -> 4x4 row 4
                        # i2=1, j2=1 -> A00[1,1] -> 4x4 row 5
                        # A11 is rows 8-11
                        # i2=0, j2=0 -> A11[0,0] -> 4x4 row 8
                        # ...
                        # This logic is complex to hardcode.
                        pass
        
    # Better approach: Construct the 49x16 matrix for U, V, W by summing contributions.
    # Let's define the block mappings.
    # Blocks: 00, 01, 10, 11.
    # Block 00: rows 0-3 (U), 0-3 (V), 0-3 (W)
    # Block 01: rows 4-7 (U), 4-7 (V), 4-7 (W)
    # Block 10: rows 8-11 (U), 8-11 (V), 8-11 (W)
    # Block 11: rows 12-15 (U), 12-15 (V), 12-15 (W)
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (rows 0-3) + A11 (rows 12-15)
    # V: B00 (rows 0-3) + B11 (rows 12-15)
    # W: C00 (rows 0-3) + C11 (rows 12-15)
    
    # We will iterate through the 7 products and add the 2x2x2 factors.
    
    block_offsets = {
        0: 0,  # 00
        1: 4,  # 01
        2: 8,  # 10
        3: 12  # 11
    }
    
    # Product definitions: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_id, sign)
    # B_blocks: list of (block_id, sign)
    # C_blocks: list of (block_id, sign)
    
    products = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (0, 1)]), # C10, C00? 
        # Wait, M2 = (A10+A11)B00.
        # C10 = M2 + M4. C00 = M1 + M4 - M5 + M7.
        # So M2 contributes to C10 and C00?
        # In 2x2: C10 = M2 + M4. C00 = M1 + M4 - M5 + M7.
        # So M2 contributes to C10 and C00.
        # C10 is block 10? No, C is 2x2.
        # C00, C01, C10, C11.
        # C10 is block 10? No, C10 is row 1, col 0.
        # In 4x4, C10 is block 10 (rows 8-11, cols 0-3)?
        # No, C is 4x4.
        # C = [C00 C01; C10 C11].
        # C10 is the bottom-left block.
        # So M2 contributes to C10 and C00.
        # C10 is block 2 (offset 8). C00 is block 0 (offset 0).
        # So C_blocks for M2: [(2, 1), (0, 1)]
        
        # M3 = A00(B01-B11)
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 and C11.
        # C01 is block 1. C11 is block 3.
        # C_blocks: [(1, 1), (3, 1)]
        
        # M4 = A11(B10-B00)
        # C00 = ... + M4 ...
        # C10 = M2 + M4.
        # So M4 contributes to C00 and C10.
        # C_blocks: [(0, 1), (2, 1)]
        
        # M5 = (A00+A01)B11
        # C01 = M3 + M5.
        # C00 = ... - M5 ...
        # So M5 contributes to C01 and C00.
        # C_blocks: [(1, 1), (0, -1)]
        
        # M6 = (A10-A00)(B00+B01)
        # C11 = ... + M6 ...
        # So M6 contributes to C11.
        # C_blocks: [(3, 1)]
        
        # M7 = (A01-A11)(B10+B11)
        # C00 = ... + M7 ...
        # C11 = ... + M7 ...
        # So M7 contributes to C00 and C11.
        # C_blocks: [(0, 1), (3, 1)]
    ]
    
    # Corrected Product Definitions
    products = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (0, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
    ]
    
    for r in range(7):
        A_blocks, B_blocks, C_blocks = products[r]
        
        # Add U contributions
        for i2 in range(2):
            for j2 in range(2):
                val_2x2 = U2[i2 * 2 + j2, r]
                if val_2x2 == 0: continue
                for block_id, sign in A_blocks:
                    offset = block_offsets[block_id]
                    row_4x4 = offset + i2 * 2 + j2
                    U4[row_4x4, r] += val_2x2 * sign
                    
        # Add V contributions
        for j2 in range(2):
            for k2 in range(2):
                val_2x2 = V2[j2 * 2 + k2, r]
                if val_2x2 == 0: continue
                for block_id, sign in B_blocks:
                    offset = block_offsets[block_id]
                    row_4x4 = offset + j2 * 2 + k2
                    V4[row_4x4, r] += val_2x2 * sign
                    
        # Add W contributions
        for k2 in range(2):
            for i2 in range(2):
                val_2x2 = W2[k2 * 2 + i2, r]
                if val_2x2 == 0: continue
                for block_id, sign in C_blocks:
                    offset = block_offsets[block_id]
                    row_4x4 = offset + k2 * 2 + i2
                    W4[row_4x4, r] += val_2x2 * sign
                    
    return U4, V4, W4

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _strassen_4x4_factors()
    
    # Convert to lists of lists for the required format
    U = [[int(x) for x in row] for row in U4]
    V = [[int(x) for x in row] for row in V4]
    W = [[int(x) for x in row] for row in W4]
    
    # Now extend to 4x4x8
    # U is 16x49. We need 16x98.
    # V is 16x49. We need 32x98.
    # W is 16x49. We need 32x98.
    
    U_full = [[0] * 98 for _ in range(16)]
    V_full = [[0] * 98 for _ in range(32)]
    W_full = [[0] * 98 for _ in range(32)]
    
    # First block (k=0..3)
    for i in range(16):
        for r in range(49):
            U_full[i][r] = U[i][r]
            V_full[i][r] = V[i][r]
            W_full[i][r] = W[i][r]
            
    # Second block (k=4..7)
    # V rows 16-31 correspond to j*8+k where k=4..7.
    # This is the same as j*4+k' where k'=0..3, but shifted.
    # So V_full[16+i][49+r] = V[i][r]
    # W rows 16-31 correspond to k*4+i where k=4..7.
    # So W_full[16+i][49+r] = W[i][r]
    
    for i in range(16):
        for r in range(49):
            V_full[16 + i][49 + r] = V[i][r]
            W_full[16 + i][49 + r] = W[i][r]
            
    # Validate
    if _validate(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
