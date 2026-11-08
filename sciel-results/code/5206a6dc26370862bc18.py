import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Rank 4 decomposition for M_2,2,1
    # A: 2x2, B: 2x1, C: 2x1
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    
    # Term 0: A00 B00 -> C0
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # Term 1: A01 B10 -> C0
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    # Term 2: A10 B00 -> C1
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    # Term 3: A11 B10 -> C1
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1

    # Rank 7 decomposition for M_2,2,2 (Strassen)
    # A: 2x2, B: 2x2, C: 2x2
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # p1 = (a00+a11)(b00+b11) -> c00, c11
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # p2 = (a10+a11)b00 -> c10, -c11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = -1
    
    # p3 = a00(b01-b11) -> c01, c11
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # p4 = a11(b10-b00) -> c00, c10
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1
    
    # p5 = (a00+a01)b11 -> -c00, c01
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[1, 4] = 1
    
    # p6 = (a10-a00)(b00+b01) -> c11
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # p7 = (a01-a11)(b10+b11) -> c00, c01
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[1, 6] = 1

    # Construct M_4,4,2 using Kronecker product of M_2,2,1 and M_2,2,2
    # Rank = 4 * 7 = 28
    U_442 = np.kron(U_221, U_222)
    V_442 = np.kron(V_221, V_222)
    W_442 = np.kron(W_221, W_222)
    
    # Construct M_4,4,1 using Kronecker product of M_2,2,1 and M_2,2,1
    # Rank = 4 * 4 = 16
    U_441 = np.kron(U_221, U_221)
    V_441 = np.kron(V_221, V_221)
    W_441 = np.kron(W_221, W_221)
    
    # Total rank = 4 * 28 + 16 = 128
    total_rank = 4 * 28 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Helper to map Kronecker indices to global indices
    # Global index i (0..15) for A (4x4)
    # Kronecker index k (0..15) from U_442 (16x28)
    # Mapping: i = 4*r + c. r = 2*br + lr, c = 2*bc + lc.
    # k = 4*b_i + l_i. b_i = 2*br + bc, l_i = 2*lr + lc.
    # We need to permute the rows of U_442, V_442, W_442 to match global order.
    
    def permute_kron_indices(mat, dim1, dim2):
        # dim1, dim2 are the dimensions of the factors (e.g., 4 and 4 for A)
        # mat is (dim1*dim2, rank)
        # We want to reorder rows such that row k in mat goes to row i in global
        # where i = 4*r + c and k = 4*b_i + l_i
        # r = 2*br + lr, c = 2*bc + lc
        # b_i = 2*br + bc, l_i = 2*lr + lc
        # k = 4*(2*br + bc) + (2*lr + lc) = 8*br + 4*bc + 2*lr + lc
        # i = 4*(2*br + lr) + (2*bc + lc) = 8*br + 4*lr + 2*bc + lc
        
        new_mat = np.zeros_like(mat)
        for i in range(dim1 * dim2):
            r = i // dim2
            c = i % dim2
            br = r // 2
            lr = r % 2
            bc = c // 2
            lc = c % 2
            b_i = 2 * br + bc
            l_i = 2 * lr + lc
            k = 4 * b_i + l_i
            new_mat[i] = mat[k]
        return new_mat

    # Permute U, V, W for the 4x4 parts
    U_442_perm = permute_kron_indices(U_442, 4, 4)
    V_442_perm = permute_kron_indices(V_442, 4, 2) # V is 4x2 for M_4,4,2? No, V_221 is 2x4, V_222 is 4x7. Kron is 8x28.
    # Wait, V_221 is (2, 4), V_222 is (4, 7). Kron is (8, 28).
    # Global V for B (4x2) has 8 entries.
    # Mapping for V: B is 4x2.
    # B global index j (0..7). j = 4*row + col.
    # Kronecker index k_v (0..7).
    # B is split into 2x2 blocks?
    # B_221 is 2x1. B_222 is 2x2.
    # B_442 is 4x2.
    # B_221 corresponds to the "block" dimension?
    # In M_2,2,1, B is 2x1.
    # In M_2,2,2, B is 2x2.
    # Kron B is 4x2.
    # Rows of B_442:
    # B_221 rows: 0,1. B_222 rows: 0,1,2,3.
    # Kron index k_v = 2 * r_221 + r_222.
    # Global B index j = 4 * r_global + c_global.
    # r_global = 2 * br + lr. c_global = 2 * bc + lc.
    # This mapping is complex.
    
    # Let's just use the Kronecker product directly and assume the evaluator handles the indices correctly?
    # No, the evaluator checks the tensor identity.
    
    # I will rely on the standard Kronecker product index ordering matching the row-major flattening of the block structure.
    # If I define the blocks in the order (block_row, block_col) and local (local_row, local_col),
    # and I flatten B in row-major order, the indices might not align with np.kron.
    
    # To be safe, I will construct the terms explicitly for the 4x4x2 case.
    
    pass
