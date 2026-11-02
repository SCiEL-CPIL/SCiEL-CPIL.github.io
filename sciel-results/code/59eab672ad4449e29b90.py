import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a decomposition based on the Strassen algorithm for 4x4 matrices,
    # extended to 4x5 and 5x5 by padding the 4x4 block and handling the extra row/column separately.
    # 
    # The idea:
    # Let A be 4x5, B be 5x5.
    # Split A into A_1 (4x4) and A_2 (4x1).
    # Split B into B_1 (4x5) and B_2 (1x5).
    # C = A_1 B_1 + A_2 B_2.
    #
    # A_1 B_1 is a 4x4 times 4x5 product.
    # We can compute A_1 B_1 using a rank-7 decomposition for the 4x4 part (Strassen) applied to the 4x4 block of B_1,
    # and handle the 5th column of B_1 separately?
    # Actually, it's easier to think of the whole 4x5 x 5x5 product.
    #
    # Let's use a known result: The rank of 4x5 x 5x5 multiplication.
    # Schoolbook rank: 4*5*5 = 100.
    #
    # We can use Strassen's algorithm on the 4x4 submatrix.
    # Let A = [A_4x4, A_4x1], B = [B_4x4, B_4x1; B_1x4, B_1x1].
    # C = A_4x4 B_4x4 + A_4x4 B_4x1 + A_4x1 B_1x4 + A_4x1 B_1x1.
    #
    # We can compute:
    # M1 = A_4x4 * B_4x4 using Strassen (rank 7).
    # M2 = A_4x4 * B_4x1 (4x4 * 4x1). This is just 4*4=16 scalar products? No, we can use a rank-4 decomposition?
    #    Actually, 4x4 * 4x1 is just multiplying a 4x4 matrix by a 4x1 vector.
    #    This can be done with 4*4 = 16 bilinear forms? No, it's linear in B_4x1 if A is fixed, but bilinear overall.
    #    Wait, the tensor is A_i,j * B_j,k.
    #    For the part A_4x4 * B_4x1, the indices are i in 0..3, j in 0..3, k in 4.
    #    This is a 4x4x1 tensor. The rank is 4 (since B_4x1 has only one column, we can just sum over j for each i).
    #    Wait, C_i,4 = sum_j A_i,j B_j,4. This is 4 terms.
    #
    # Let's be more systematic.
    # We want to decompose the tensor T_{i,j,k} = 1 for all i,j,k (with appropriate indexing).
    #
    # Strategy:
    # 1. Use Strassen's algorithm for the 4x4x4 core.
    # 2. Handle the extra dimensions (row 4 of A, col 4 of B, etc.) with standard multiplication.
    #
    # Let's define the indices:
    # i in 0..3 (rows of A), j in 0..4 (cols of A / rows of B), k in 0..4 (cols of B).
    #
    # Split:
    # j = j' + 4*delta, where j' in 0..3, delta in {0, 1}.
    # k = k' + 4*delta', where k' in 0..3, delta' in {0, 1}.
    #
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    #          = sum_{j'} A_{i,j'} B_{j',k}  (j' in 0..3)
    #          + A_{i,4} B_{4,k}
    #
    # Let's split B_{j',k} into B_{j',k'} (k' in 0..3) and B_{j',4}.
    # C_{i,k'} = sum_{j'} A_{i,j'} B_{j',k'} + sum_{j'} A_{i,j'} B_{j',4} * 0 (if k!=4) ... wait.
    #
    # C_{i,k} = sum_{j'=0}^3 A_{i,j'} B_{j',k} + A_{i,4} B_{4,k}.
    #
    # For k in 0..3:
    # C_{i,k} = sum_{j'=0}^3 A_{i,j'} B_{j',k} + A_{i,4} B_{4,k}.
    #
    # For k=4:
    # C_{i,4} = sum_{j'=0}^3 A_{i,j'} B_{j',4} + A_{i,4} B_{4,4}.
    #
    # Let's group the terms:
    # Term 1: sum_{j'=0}^3 A_{i,j'} B_{j',k} for k in 0..3. This is the 4x4x4 core.
    # Term 2: sum_{j'=0}^3 A_{i,j'} B_{j',4}. This is 4x4x1.
    # Term 3: A_{i,4} B_{4,k} for k in 0..4. This is 4x1x5.
    #
    # We can compute Term 1 using Strassen (rank 7).
    # Term 2: C_{i,4}^{(2)} = sum_{j'} A_{i,j'} B_{j',4}. This is just a matrix-vector product.
    #    For each i, we sum over j'. This requires 4*4 = 16 bilinear terms?
    #    No, we can just use the standard multiplication for this part.
    #    Actually, we can combine Term 2 and Term 3?
    #
    # Let's use a different approach: Pad A to 5x5 and B to 5x5? No, A is 4x5.
    #
    # Let's use the explicit Strassen construction for 4x4x4.
    # The Strassen algorithm for 4x4x4 uses 7 blocks, each of size 2x2x2, which uses 7 scalar products.
    # So total rank for 4x4x4 is 7 * 7 = 49? No.
    # Strassen's algorithm for n x n uses n^log2(7) operations.
    # For 4x4, it's 4^log2(7) ~ 21.8, so 22?
    # Actually, the rank of 4x4x4 multiplication is known to be 49? No, 4*4*4=64 is schoolbook.
    # Strassen reduces it to 7 * (2x2x2 rank). 2x2x2 rank is 7. So 7*7=49.
    #
    # So for the 4x4x4 core, we can use rank 49.
    #
    # Now, let's handle the rest.
    # Term 2: sum_{j'=0}^3 A_{i,j'} B_{j',4}.
    # This is a 4x4x1 tensor. We can compute this with 4*4 = 16 terms?
    # Actually, for each i, we have a sum of 4 terms. So 4*4=16 terms.
    #
    # Term 3: A_{i,4} B_{4,k}.
    # This is a 4x1x5 tensor. For each i and k, it's a product. So 4*5=20 terms.
    #
    # Total rank = 49 + 16 + 20 = 85.
    #
    # This is much better than 100.
    #
    # Let's implement this.
    #
    # We need to construct U, V, W matrices.
    # U has shape (4*5, rank) = (20, rank).
    # V has shape (5*5, rank) = (25, rank).
    # W has shape (5*4, rank) = (20, rank).
    #
    # We will build the decomposition term by term.
    #
    # First, the 4x4x4 core (Term 1).
    # We need a rank-49 decomposition for 4x4x4.
    # We can use the Strassen algorithm recursively.
    #
    # Let's define a function to get the Strassen decomposition for 2x2x2.
    # The 7 terms are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4x4, we split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    #
    # We compute M1..M7, each of which is a 2x2x2 product.
    # Each 2x2x2 product has rank 7.
    # So total rank is 7 * 7 = 49.
    #
    # We will construct the U, V, W matrices for this part.
    #
    # Let's define the Strassen coefficients.
    # For each of the 7 terms (l in 0..6), we have:
    # A_l = combination of A blocks.
    # B_l = combination of B blocks.
    # C_l = combination of C blocks.
    #
    # We will use a helper function to generate the Strassen decomposition for 2x2x2.
    
    def strassen_2x2():
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00 is index 0, A11 is index 3.
        # B00 is index 0, B11 is index 3.
        # C00 += M1, C11 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11) * B00
        # A10 is index 2, A11 is index 3.
        # B00 is index 0.
        # C10 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00 * (B01 - B11)
        # A00 is index 0.
        # B01 is index 1, B11 is index 3.
        # C01 += M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11 * (B10 - B00)
        # A11 is index 3.
        # B10 is index 2, B00 is index 0.
        # C00 += M4, C10 += M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01) * B11
        # A00 is index 0, A01 is index 1.
        # B11 is index 3.
        # C00 -= M5, C01 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # A10 is index 2, A00 is index 0.
        # B00 is index 0, B01 is index 1.
        # C11 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # A01 is index 1, A11 is index 3.
        # B10 is index 2, B11 is index 3.
        # C00 += M7, C11 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    U_core, V_core, W_core = strassen_2x2()
    
    # Now, build the 4x4x4 decomposition.
    # U_4x4 shape (16, 49), V_4x4 shape (16, 49), W_4x4 shape (16, 49)
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    # The 7 terms of Strassen for 4x4:
    # Each term l (0..6) corresponds to a 2x2x2 product.
    # The indices for U, V, W in the 4x4 case are:
    # U index: (block_row, block_col, inner_row, inner_col)
    # V index: (block_row, block_col, inner_row, inner_col)
    # W index: (block_row, block_col, inner_row, inner_col)
    
    # We need to map the Strassen coefficients to the 4x4 indices.
    # For each term l, we have:
    # A_l is a 2x2 matrix of blocks.
    # B_l is a 2x2 matrix of blocks.
    # C_l is a 2x2 matrix of blocks.
    
    # Let's define the Strassen coefficients for the 4x4 case.
    # This is a bit complex. Let's use a simpler approach:
    # We will use the known rank-49 decomposition for 4x4x4.
    # Actually, we can just use the schoolbook for the 4x4x4 part to be safe, but that would be rank 64.
    # 64 + 16 + 20 = 100. No improvement.
    #
    # Let's try to implement the Strassen construction properly.
    #
    # The 7 terms for 4x4 are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # Where A00, A01, A10, A11 are 2x2 blocks.
    #
    # For each term, we use the 2x2x2 Strassen decomposition.
    #
    # Let's define the block indices.
    # A block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1.
    #
    # We will iterate over the 7 terms and the 7 inner terms.
    
    for l in range(7):
        # Determine which blocks are involved in A_l and B_l.
        # This is a bit tedious. Let's use a lookup table.
        
        # A_l blocks:
        if l == 0: # M1
            A_blocks = [(0,0,1), (1,1,1)]
            B_blocks = [(0,0,1), (1,1,1)]
            C_blocks = [(0,0,1), (1,1,1)]
        elif l == 1: # M2
            A_blocks = [(1,0,1), (1,1,1)]
            B_blocks = [(0,0,1)]
            C_blocks = [(1,0,1)]
        elif l == 2: # M3
            A_blocks = [(0,0,1)]
            B_blocks = [(0,1,1), (1,1,-1)]
            C_blocks = [(0,1,1)]
        elif l == 3: # M4
            A_blocks = [(1,1,1)]
            B_blocks = [(1,0,1), (0,0,-1)]
            C_blocks = [(0,0,1), (1,0,1)]
        elif l == 4: # M5
            A_blocks = [(0,0,1), (0,1,1)]
            B_blocks = [(1,1,1)]
            C_blocks = [(0,0,-1), (0,1,1)]
        elif l == 5: # M6
            A_blocks = [(1,0,1), (0,0,-1)]
            B_blocks = [(0,0,1), (0,1,1)]
            C_blocks = [(1,1,1)]
        elif l == 6: # M7
            A_blocks = [(0,1,1), (1,1,-1)]
            B_blocks = [(1,0,1), (1,1,1)]
            C_blocks = [(0,0,1), (1,1,1)]
        else:
            continue
        
        for s in range(7):
            # s is the inner Strassen term index.
            # We need to add the contribution of U_core[inner_i, s], etc.
            
            # U_core has shape (4, 7). The indices are (br, bc) for the 2x2 block.
            # For each block in A_blocks, we add U_core[inner_i, s] * coeff.
            
            for (br, bc, coeff) in A_blocks:
                for inner_i in range(4):
                    if U_core[inner_i, s] != 0:
                        # inner_i is (br_inner, bc_inner) in the 2x2 block.
                        br_inner = inner_i // 2
                        bc_inner = inner_i % 2
                        # Global index: (2*br + br_inner, 2*bc + bc_inner)
                        gi = 2 * (2 * br + br_inner) + (2 * bc + bc_inner)
                        U_4x4[gi, l * 7 + s] += coeff * U_core[inner_i, s]
                        
            for (br, bc, coeff) in B_blocks:
                for inner_i in range(4):
                    if V_core[inner_i, s] != 0:
                        br_inner = inner_i // 2
                        bc_inner = inner_i % 2
                        gi = 2 * (2 * br + br_inner) + (2 * bc + bc_inner)
                        V_4x4[gi, l * 7 + s] += coeff * V_core[inner_i, s]
                        
            for (br, bc, coeff) in C_blocks:
                for inner_i in range(4):
                    if W_core[inner_i, s] != 0:
                        br_inner = inner_i // 2
                        bc_inner = inner_i % 2
                        gi = 2 * (2 * br + br_inner) + (2 * bc + bc_inner)
                        W_4x4[gi, l * 7 + s] += coeff * W_core[inner_i, s]
    
    # Now, we have the 4x4x4 decomposition.
    # U_4x4 has shape (16, 49), V_4x4 has shape (16, 49), W_4x4 has shape (16, 49).
    
    # Now, let's build the full 4x5 x 5x5 decomposition.
    # U_full shape (20, 85), V_full shape (25, 85), W_full shape (20, 85).
    
    rank = 85
    U_full = np.zeros((20, rank))
    V_full = np.zeros((25, rank))
    W_full = np.zeros((20, rank))
    
    # Copy the 4x4x4 core into the full matrices.
    # The core corresponds to i in 0..3, j in 0..3, k in 0..3.
    # U_full indices: i*m + j = i*5 + j.
    # V_full indices: j*p + k = j*5 + k.
    # W_full indices: k*n + i = k*4 + i.
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # The core tensor entry is U_4x4[i*4+j, r] * V_4x4[j*4+k, r] * W_4x4[k*4+i, r].
                # We need to map this to the full indices.
                pass
    
    # Actually, it's easier to just copy the matrices.
    # U_full[i*5+j, r] = U_4x4[i*4+j, r] for i in 0..3, j in 0..3.
    for i in range(4):
        for j in range(4):
            U_full[i * 5 + j, :49] = U_4x4[i * 4 + j, :]
            
    for j in range(4):
        for k in range(4):
            V_full[j * 5 + k, :49] = V_4x4[j * 4 + k, :]
            
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :49] = W_4x4[k * 4 + i, :]
    
    # Now, handle Term 2: sum_{j'=0}^3 A_{i,j'} B_{j',4}.
    # This contributes to C_{i,4}.
    # We can use 16 terms for this.
    # For each i in 0..3, j' in 0..3, we have a term A_{i,j'} * B_{j',4}.
    # This is a rank-1 term.
    
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U_full[i * 5 + j, r] = 1
            V_full[j * 5 + 4, r] = 1
            W_full[4 * 4 + i, r] = 1  # k=4, so index is 4*4 + i = 16 + i.
            
    # Now, handle Term 3: A_{i,4} B_{4,k}.
    # This contributes to C_{i,k} for all k.
    # We can use 20 terms for this.
    # For each i in 0..3, k in 0..4, we have a term A_{i,4} * B_{4,k}.
    
    for i in range(4):
        for k in range(5):
            r = 49 + 16 + i * 5 + k
            U_full[i * 5 + 4, r] = 1
            V_full[4 * 5 + k, r] = 1
            W_full[k * 4 + i, r] = 1
    
    # Now, we have the full decomposition.
    # Let's verify it.
    
    # Verify the tensor.
    is_exact = True
    for i in range(4):
        for j in range(5):
            for k in range(5):
                val = np.dot(U_full[i * 5 + j, :], V_full[j * 5 + k, :] * W_full[k * 4 + i, :])
                if abs(val - 1.0) > 1e-9:
                    is_exact = False
                    break
            if not is_exact:
                break
        if not is_exact:
            break
            
    if is_exact:
        # Convert to lists of lists
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if verification fails
        rank = 100
        U = [[0] * rank for _ in range(20)]
        V = [[0] * rank for _ in range(25)]
        W = [[0] * rank for _ in range(20)]
        column = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j][column] = 1
                    V[j * 5 + k][column] = 1
                    W[k * 4 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
