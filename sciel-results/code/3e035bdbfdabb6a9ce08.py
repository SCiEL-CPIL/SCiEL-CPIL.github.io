import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: 
    # Part 1: 4x4 * 4x4 (first 4 cols of B) using 2x2 Strassen blocks -> rank 28
    # Part 2: 4x4 * 4x2 (last 2 cols of B) using schoolbook -> rank 16
    # Total rank: 44
    
    # However, let's verify if we can do better.
    # The 4x4x4 multiplication can be decomposed into 4 blocks of 2x2x2.
    # Each 2x2x2 block multiplication has rank 7 (Strassen).
    # There are 4 such blocks (top-left, top-right, bottom-left, bottom-right).
    # So rank = 4 * 7 = 28 for the 4x4x4 part.
    
    # For the 4x4x2 part:
    # A is 4x4, B is 4x2, C is 4x2.
    # We can view this as 4 blocks of 2x2x2? No, B is 4x2, so it's 2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00], [B10]] (each 2x2)
    # C = [[C00], [C10]] (each 2x2)
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # This is 4 block multiplications of 2x2x2.
    # Each 2x2x2 multiplication has rank 7.
    # So rank = 4 * 7 = 28 for the 4x4x2 part?
    # Wait, the blocks are not independent in the same way.
    # Let's check the tensor structure.
    # The tensor for 4x4x2 is a sum of 4 terms:
    # T = A00*B00*C00 + A01*B10*C00 + A10*B00*C10 + A11*B10*C10
    # This is a sum of 4 rank-7 tensors (since each 2x2x2 is rank 7).
    # So total rank for 4x4x2 is 28.
    
    # Total rank = 28 (for 4x4x4) + 28 (for 4x4x2) = 56?
    # Wait, the schoolbook for 4x4x2 is 4*4*2 = 32.
    # Strassen for 4x4x2:
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # We can compute M1 = A00*B00, M2 = A01*B10, M3 = A10*B00, M4 = A11*B10.
    # Each Mi is a 2x2 matrix product, rank 7.
    # Total rank = 4 * 7 = 28.
    # This is better than schoolbook 32.
    
    # So total rank = 28 + 28 = 56.
    
    # But wait, can we do better for the 4x4x4 part?
    # The 4x4x4 part is 4 blocks of 2x2x2.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # This is 8 block multiplications of 2x2x2.
    # Each is rank 7.
    # Total rank = 8 * 7 = 56.
    
    # So total rank = 56 (for 4x4x4) + 28 (for 4x4x2) = 84.
    
    # This matches the "inspiration 1" finding: 84 rank.
    
    # Let's try to implement this.
    
    # We need to construct U, V, W matrices.
    # U has shape (n*m, rank) = (16, 84)
    # V has shape (m*p, rank) = (24, 84)
    # W has shape (p*n, rank) = (24, 84)
    
    # We will define a helper function to add a 2x2x2 Strassen block.
    
    def add_strassen_block(U, V, W, base_col, 
                           a_r1, a_c1, a_r2, a_c2,  # A block rows/cols
                           b_r1, b_c1, b_r2, b_c2,  # B block rows/cols
                           c_r1, c_c1, c_r2, c_c2): # C block rows/cols
        # A block is 2x2: rows a_r1..a_r1+1, cols a_c1..a_c1+1
        # B block is 2x2: rows b_r1..b_r1+1, cols b_c1..b_c1+1
        # C block is 2x2: rows c_r1..c_r1+1, cols c_c1..c_c1+1
        
        # Strassen vectors for 2x2x2:
        # u_vecs: 7 vectors of length 4 (for A's 4 elements)
        # v_vecs: 7 vectors of length 4 (for B's 4 elements)
        # w_vecs: 7 vectors of length 4 (for C's 4 elements)
        
        u_vecs = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ], dtype=float)
        
        v_vecs = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=float)
        
        w_vecs = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Map local indices (0,1,2,3) to global indices
        # A local index l corresponds to:
        # l=0: (a_r1, a_c1)
        # l=1: (a_r1, a_c1+1)
        # l=2: (a_r1+1, a_c1)
        # l=3: (a_r1+1, a_c1+1)
        # Global index in U: row = r*m + c
        
        a_global_indices = [
            (a_r1, a_c1),
            (a_r1, a_c1+1),
            (a_r1+1, a_c1),
            (a_r1+1, a_c1+1)
        ]
        
        b_global_indices = [
            (b_r1, b_c1),
            (b_r1, b_c1+1),
            (b_r1+1, b_c1),
            (b_r1+1, b_c1+1)
        ]
        
        c_global_indices = [
            (c_r1, c_c1),
            (c_r1, c_c1+1),
            (c_r1+1, c_c1),
            (c_r1+1, c_c1+1)
        ]
        
        for k in range(7):
            col = base_col + k
            
            for l in range(4):
                if u_vecs[k, l] != 0:
                    r, c = a_global_indices[l]
                    U[r*m + c, col] += u_vecs[k, l]
                
                if v_vecs[k, l] != 0:
                    r, c = b_global_indices[l]
                    V[r*p + c, col] += v_vecs[k, l]
                
                if w_vecs[k, l] != 0:
                    r, c = c_global_indices[l]
                    W[r*n + c, col] += w_vecs[k, l]
    
    rank = 84
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 4x4 * 4x4 (first 4 cols of B)
    # 8 block multiplications:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    col_offset = 0
    # Block 1: A00*B00 -> C00
    add_strassen_block(U, V, W, col_offset, 0, 0, 0, 0, 0, 0, 0, 0)
    col_offset += 7
    
    # Block 2: A01*B10 -> C00
    add_strassen_block(U, V, W, col_offset, 0, 2, 0, 0, 2, 0, 0, 0)
    col_offset += 7
    
    # Block 3: A00*B01 -> C01
    add_strassen_block(U, V, W, col_offset, 0, 0, 0, 0, 0, 2, 0, 0)
    col_offset += 7
    
    # Block 4: A01*B11 -> C01
    add_strassen_block(U, V, W, col_offset, 0, 2, 0, 0, 2, 2, 0, 0)
    col_offset += 7
    
    # Block 5: A10*B00 -> C10
    add_strassen_block(U, V, W, col_offset, 2, 0, 0, 0, 0, 0, 2, 0)
    col_offset += 7
    
    # Block 6: A11*B10 -> C10
    add_strassen_block(U, V, W, col_offset, 2, 2, 0, 0, 2, 0, 2, 0)
    col_offset += 7
    
    # Block 7: A10*B01 -> C11
    add_strassen_block(U, V, W, col_offset, 2, 0, 0, 0, 0, 2, 2, 0)
    col_offset += 7
    
    # Block 8: A11*B11 -> C11
    add_strassen_block(U, V, W, col_offset, 2, 2, 0, 0, 2, 2, 2, 0)
    col_offset += 7
    
    # Part 2: 4x4 * 4x2 (last 2 cols of B)
    # 4 block multiplications:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Block 9: A00*B02 -> C02
    add_strassen_block(U, V, W, col_offset, 0, 0, 0, 0, 0, 4, 0, 0)
    col_offset += 7
    
    # Block 10: A01*B12 -> C02
    add_strassen_block(U, V, W, col_offset, 0, 2, 0, 0, 2, 4, 0, 0)
    col_offset += 7
    
    # Block 11: A10*B02 -> C12
    add_strassen_block(U, V, W, col_offset, 2, 0, 0, 0, 0, 4, 2, 0)
    col_offset += 7
    
    # Block 12: A11*B12 -> C12
    add_strassen_block(U, V, W, col_offset, 2, 2, 0, 0, 2, 4, 2, 0)
    col_offset += 7
    
    if time.time() > deadline:
        # Fallback to schoolbook if we ran out of time (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
