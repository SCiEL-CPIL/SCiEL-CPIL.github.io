import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Two-level hierarchical Strassen decomposition targeting rank 42.
    #
    # Level 1: Split m=4 into two 2-dim blocks.
    #   A = [A1 | A2] where A1, A2 are 2x2
    #   B = [B1; B2] where B1, B2 are 2x8
    #   C = A1*B1 + A2*B2
    #
    # Karatsuba on the intermediate dimension:
    #   P1 = A1 * B1
    #   P2 = A2 * B2  
    #   P3 = (A1+A2) * (B1+B2)
    #   C = P1 + P3 - P2
    #
    # Each Pi is a 2x8 matrix.
    #
    # Level 2: Each 2x2 * 2x8 multiplication can be decomposed as 4 independent
    #   2x2 * 2x2 multiplications (one for each pair of columns in the 2x8).
    #   Each 2x2 * 2x2 uses Strassen with rank 7.
    #
    # Total rank = 3 (Karatsuba) * 4 (column blocks) * 7 (Strassen) = 84
    # But we can do better: the three Karatsuba products share structure.
    #
    # Actually, let's think more carefully. We have:
    #   P1 = A1 * B1: 4 blocks of 2x2*2x2, rank 28
    #   P2 = A2 * B2: 4 blocks of 2x2*2x2, rank 28  
    #   P3 = (A1+A2) * (B1+B2): 4 blocks of 2x2*2x2, rank 28
    #   C = P1 + P3 - P2
    #
    # Total rank = 28 + 28 + 28 = 84. That's worse than 56.
    #
    # Let me reconsider. The original approach gives rank 56:
    #   C = A1*B1 + A2*B2
    #   Each term is 2x2 * 2x8 = 4 * (2x2*2x2) with rank 7 each = 28 per term
    #   Total = 28 + 28 = 56
    #
    # To get rank 42, we need a different decomposition.
    #
    # Alternative: Use Strassen directly on the full problem structure.
    # View the multiplication as computing C[i,k] = sum_j A[i,j] B[j,k]
    # where i in {0,1}, j in {0,1,2,3}, k in {0,...,7}.
    #
    # Let's try a different block structure. Split the output C (2x8) into
    # four 2x2 blocks: C = [C0 | C1 | C2 | C3] where each Ci is 2x2.
    #
    # C0 = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]  -- no, that's not right.
    #
    # Actually C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # So C0 (columns 0,1) = sum_{j=0}^{3} A[i,j] B[j,0:2]
    #   = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    #
    # This doesn't simplify things.
    #
    # Let me try yet another approach. The tensor for 2x4 * 4x8 has rank at most 64.
    # We know rank 56 works. Can we do better?
    #
    # Consider using the fact that m=4 is even. We can write:
    #   A = [[a00, a01, a02, a03],
    #        [a10, a11, a12, a13]]
    #   B = [[b00, b01, ..., b07],
    #        [b10, b11, ..., b17],
    #        [b20, b21, ..., b27],
    #        [b30, b31, ..., b37]]
    #
    # C[i,k] = a_i0*b_0k + a_i1*b_1k + a_i2*b_2k + a_i3*b_3k
    #
    # One approach: group terms.
    # C[i,k] = (a_i0*b_0k + a_i1*b_1k) + (a_i2*b_2k + a_i3*b_3k)
    #        = (A_left[:,i] dot B_top[:,k]) + (A_right[:,i] dot B_bottom[:,k])
    #
    # This is what the original code does, giving rank 56.
    #
    # For rank 42, we'd need to find a decomposition with 42 terms.
    # 42 = 6 * 7, so maybe 6 Strassen-like blocks?
    # Or 42 = 3 * 14, etc.
    #
    # Let me try a different splitting. What if we split differently?
    #
    # Actually, let me just implement the rank-56 solution cleanly since that's
    # what we know works, and the directive says to keep the design family.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    u_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(8):
        if t < 4:
            a_col_offset = 0
            b_row_offset = 0
        else:
            a_col_offset = 2
            b_row_offset = 2
            
        b_col_offset = (t % 4) * 2
        
        a_indices = [
            0 * m + a_col_offset,
            0 * m + a_col_offset + 1,
            1 * m + a_col_offset,
            1 * m + a_col_offset + 1
        ]
        
        b_indices = [
            b_row_offset * p + b_col_offset,
            b_row_offset * p + b_col_offset + 1,
            (b_row_offset + 1) * p + b_col_offset,
            (b_row_offset + 1) * p + b_col_offset + 1
        ]
        
        c_indices = [
            b_col_offset * n + 0,
            (b_col_offset + 1) * n + 0,
            b_col_offset * n + 1,
            (b_col_offset + 1) * n + 1
        ]
        
        for l in range(7):
            r = t * 7 + l
            
            for local_idx, coeff in enumerate(u_coeffs[l]):
                if coeff != 0:
                    U[a_indices[local_idx], r] += coeff
            
            for local_idx, coeff in enumerate(v_coeffs[l]):
                if coeff != 0:
                    V[b_indices[local_idx], r] += coeff
            
            for local_idx, coeff in enumerate(w_coeffs[l]):
                if coeff != 0:
                    W[c_indices[local_idx], r] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
