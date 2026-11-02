# EVOLVE-BLOCK-START
import time

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Total rank: 7 outer Strassen steps * 11 inner terms (7 Strassen + 4 Schoolbook)
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x6
    # A is 4x4, B is 4x6, C is 4x6
    # Partition A into 2x2 blocks of size 2x2: A00, A01, A10, A11
    # Partition B into 2x2 blocks of size 2x3: B00, B01, B10, B11
    #   B00: rows 0,1; cols 0,1,2
    #   B01: rows 0,1; cols 3,4,5
    #   B10: rows 2,3; cols 0,1,2
    #   B11: rows 2,3; cols 3,4,5
    # Partition C into 2x2 blocks of size 2x3: C00, C01, C10, C11
    #   C00: rows 0,1; cols 0,1,2
    #   C01: rows 0,1; cols 3,4,5
    #   C10: rows 2,3; cols 0,1,2
    #   C11: rows 2,3; cols 3,4,5
    
    # Strassen steps for outer blocks
    # Each step defines linear combinations of A-blocks, B-blocks, and C-blocks.
    # Format: (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are lists of (block_id, coeff)
    # Block IDs: 0=A00, 1=A01, 2=A10, 3=A11 (and similarly for B, C)
    
    outer_steps = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00 + C11
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        # M2 = (A10 + A11) * B00 -> C10 - C11
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        # M3 = A00 * (B01 - B11) -> C01 + C11
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        # M4 = A11 * (B10 - B00) -> C00 + C10
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        # M5 = (A00 + A01) * B11 -> C01 - C00
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        # M6 = (A10 - A00) * (B00 + B01) -> C11
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        # M7 = (A01 - A11) * (B10 + B11) -> C00
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    # Inner Decomposition for a 2x2 * 2x3 block
    # This decomposes the multiplication of a 2x2 matrix A_loc and 2x3 matrix B_loc into C_loc (2x3).
    # We use Strassen for the first 2 columns (2x2 * 2x2) and Schoolbook for the 3rd column (2x2 * 2x1).
    # Total inner rank = 7 + 4 = 11.
    
    # Helper to map local block coordinates to global indices
    # A block (2x2): local (r, s) -> global (I, J)
    # B block (2x3): local (s, k) -> global (J, K)
    # C block (2x3): local (r, k) -> global (I, K)
    
    def get_global_A(block_id, r, s):
        # block_id 0: 00, 1: 01, 2: 10, 3: 11
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        # block_id 0: 00, 1: 01, 2: 10, 3: 11
        # B is 4x6. Blocks are 2x3.
        # Block 00: Rows 0,1; Cols 0,1,2
        # Block 01: Rows 0,1; Cols 3,4,5
        # Block 10: Rows 2,3; Cols 0,1,2
        # Block 11: Rows 2,3; Cols 3,4,5
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        # C is 4x6. Blocks are 2x3.
        # Block 00: Rows 0,1; Cols 0,1,2
        # Block 01: Rows 0,1; Cols 3,4,5
        # Block 10: Rows 2,3; Cols 0,1,2
        # Block 11: Rows 2,3; Cols 3,4,5
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Inner Strassen steps for 2x2 * 2x2 (first 2 cols)
    # Local A (2x2), Local B (2x2), Local C (2x2)
    # Indices: A(r,s), B(s,k), C(r,k) with r,s,k in {0,1}
    inner_strassen = [
        # S1: (A00+A11)(B00+B11) -> C00+C11
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        # S2: (A10+A11)B00 -> C10-C11
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        # S3: A00(B01-B11) -> C01+C11
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        # S4: A11(B10-B00) -> C00+C10
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        # S5: (A00+A01)B11 -> C01-C00
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        # S6: (A10-A00)(B00+B01) -> C11
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        # S7: (A01-A11)(B10+B11) -> C00
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    # Inner Schoolbook terms for 2x2 * 2x1 (3rd col)
    # Local A (2x2), Local B (2x1), Local C (2x1)
    # Indices: A(r,s), B(s,0), C(r,0) with r,s in {0,1}
    # Note: In the global context, this corresponds to k=2 in the B and C blocks.
    inner_schoolbook = [
        # T1: A00 * B02 -> C02
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        # T2: A01 * B12 -> C02
        (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
        # T3: A10 * B02 -> C12
        (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
        # T4: A11 * B12 -> C12
        (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
    ]
    
    # Construct the decomposition
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 11
        
        # Process Inner Strassen (7 terms)
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            c_global_rank = base_rank + j
            
            # Update U
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            # Update V
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            # Update W
            # W index is k*n + i. Here r is the row of C (which is I), k is the col of C (which is K).
            # So W index is K * n + I.
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

        # Process Inner Schoolbook (4 terms)
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            c_global_rank = base_rank + 7 + j
            
            # Update U
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            # Update V
            # Note: In schoolbook for the 3rd column, the local B index k is 0 (representing the single column).
            # We must map this to global K=2 offset within the block.
            # My get_global_B maps local k to global K. 
            # For schoolbook, the local B is 2x1. The entries are B(s, 0).
            # In the global block, this corresponds to the 3rd column (index 2 within the block of 3).
            # So we need to adjust the mapping or the input to get_global_B.
            # get_global_B(block_id, s, k) maps k directly.
            # For the 3rd column of the 2x3 block, the local index in the block is 2.
            # So we should pass k=2 to get_global_B for the schoolbook terms?
            # Let's check inner_strassen. It used k in {0,1}.
            # For schoolbook, we are multiplying by the 3rd column.
            # So local k should be 2?
            # Let's re-verify the structure.
            # Inner Strassen handles cols 0,1.
            # Inner Schoolbook handles col 2.
            # So for Schoolbook, the local B index k should be 2.
            # My inner_schoolbook definition above used k=0 in the tuples.
            # I need to change the k in the tuples to 2, OR modify the loop to add an offset.
            # It's cleaner to define the tuples with the correct local block index k.
            
            # Let's redefine inner_schoolbook with k=2 for B and C.
            # Wait, I defined inner_schoolbook above with k=0.
            # I will fix the loop to use k=2 for B and C in the schoolbook part.
            
            for (s, k_local, coeff) in B_inner:
                # k_local is 0 in the tuple, but represents the only column of the 2x1 block.
                # In the global 2x3 block, this is column index 2.
                k_global_in_block = 2 
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k_global_in_block)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            # Update W
            for (r, k_local, coeff) in C_inner:
                k_global_in_block = 2
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k_global_in_block)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
