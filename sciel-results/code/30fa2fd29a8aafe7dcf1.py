import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p=7 into p1=4 and p2=3
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    # Total rank: 97
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x total_rank
    
    # Part 1: 4x4x4 Strassen decomposition
    # A is 4x4, B is 4x4 (first 4 columns), C is 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Each Mi is a 2x2x2 multiplication, which has rank 7.
    # So we need to construct the 2x2x2 Strassen decomposition.
    
    # 2x2x2 Strassen decomposition (rank 7):
    # For 2x2 matrices A, B, C = A*B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # The 2x2x2 tensor decomposition has 7 terms.
    # Each term has a coefficient for each of the 8 input pairs (A_ij, B_jk) and each of the 8 output (C_ik).
    # For 2x2x2, indices: i,j,k in {0,1}
    # U has 4 entries (i*m+j = i*2+j), V has 4 entries (j*p+k = j*2+k), W has 4 entries (k*n+i = k*2+i)
    
    # Strassen 2x2x2 decomposition (7 terms):
    # Term 1: M1 = (A11+A22)(B11+B22)
    #   U: A11=1, A22=1 -> [1, 0, 0, 1]
    #   V: B11=1, B22=1 -> [1, 0, 0, 1]
    #   W: C11 += 1, C22 += 1 -> [1, 0, 0, 1]
    # Term 2: M2 = (A21+A22)B11
    #   U: A21=1, A22=1 -> [0, 0, 1, 1]
    #   V: B11=1 -> [1, 0, 0, 0]
    #   W: C21 += 1 -> [0, 0, 1, 0]
    # Term 3: M3 = A11(B12-B22)
    #   U: A11=1 -> [1, 0, 0, 0]
    #   V: B12=1, B22=-1 -> [0, 1, -1, 0]
    #   W: C12 += 1 -> [0, 1, 0, 0]
    # Term 4: M4 = A22(B21-B11)
    #   U: A22=1 -> [0, 0, 0, 1]
    #   V: B21=1, B11=-1 -> [-1, 0, 1, 0]
    #   W: C11 += 1, C21 += 1 -> [1, 0, 1, 0]
    # Term 5: M5 = (A11+A12)B22
    #   U: A11=1, A12=1 -> [1, 1, 0, 0]
    #   V: B22=1 -> [0, 0, 0, 1]
    #   W: C11 -= 1, C12 += 1 -> [-1, 1, 0, 0]
    # Term 6: M6 = (A21-A11)(B11+B12)
    #   U: A21=1, A11=-1 -> [-1, 0, 1, 0]
    #   V: B11=1, B12=1 -> [1, 1, 0, 0]
    #   W: C22 += 1 -> [0, 0, 0, 1]
    # Term 7: M7 = (A12-A22)(B21+B22)
    #   U: A12=1, A22=-1 -> [0, 1, 0, -1]
    #   V: B21=1, B22=1 -> [0, 0, 1, 1]
    #   W: C11 += 1, C22 += 1 -> [1, 0, 0, 1]
    
    # Let me verify this decomposition by checking C = A*B:
    # C11 = A11*B11 + A12*B21
    # From terms: M1 contributes 1, M4 contributes 1, M5 contributes -1, M7 contributes 1
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M4 = A22(B21-B11) = A22B21 - A22B11
    # M5 = (A11+A12)B22 = A11B22 + A12B22
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    # Sum: A11B11 + A11B22 + A22B11 + A22B22 + A22B21 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
    # = A11B11 + A12B21 ✓
    
    # C12 = A11*B12 + A12*B22
    # From terms: M3 contributes 1, M5 contributes 1
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M5 = (A11+A12)B22 = A11B22 + A12B22
    # Sum: A11B12 - A11B22 + A11B22 + A12B22 = A11B12 + A12B22 ✓
    
    # C21 = A21*B11 + A22*B21
    # From terms: M2 contributes 1, M4 contributes 1
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M4 = A22(B21-B11) = A22B21 - A22B11
    # Sum: A21B11 + A22B11 + A22B21 - A22B11 = A21B11 + A22B21 ✓
    
    # C22 = A21*B12 + A22*B22
    # From terms: M1 contributes 1, M2 contributes -1, M3 contributes 1, M6 contributes 1
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    # Sum: A11B11 + A11B22 + A22B11 + A22B22 - A21B11 - A22B11 + A11B12 - A11B22 + A21B11 + A21B12 - A11B11 - A11B12
    # = A22B22 + A21B12 ✓
    
    # Great! The 2x2x2 Strassen decomposition is correct.
    
    # Now for 4x4x4, we apply Strassen at the 2x2 block level.
    # Each of the 7 M terms involves a 2x2x2 multiplication.
    # So we need to expand each 2x2x2 multiplication into 7 terms of the 4x4x4 tensor.
    
    # For the 4x4x4 case:
    # A is 4x4, split into 2x2 blocks: A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right)
    # B is 4x4, split into 2x2 blocks: B11, B12, B21, B22
    # C is 4x4, split into 2x2 blocks: C11, C12, C21, C22
    
    # M1 = (A11+A22)(B11+B22): This is a 2x2x2 multiplication where:
    #   The "A" matrix is A11+A22 (2x2), the "B" matrix is B11+B22 (2x2)
    #   In terms of 4x4 indices:
    #     A11 has indices (0,0), (0,1), (1,0), (1,1) in 4x4
    #     A22 has indices (2,2), (2,3), (3,2), (3,3) in 4x4
    #     B11 has indices (0,0), (0,1), (1,0), (1,1) in 4x4 (as B)
    #     B22 has indices (2,2), (2,3), (3,2), (3,3) in 4x4 (as B)
    
    # Let me define the index mappings:
    # For 4x4, A[i,j] where i,j in 0..3
    # A11: i in 0..1, j in 0..1
    # A12: i in 0..1, j in 2..3
    # A21: i in 2..3, j in 0..1
    # A22: i in 2..3, j in 2..3
    # Similarly for B and C.
    
    # U index: i*m + j = i*4 + j (0..15)
    # V index: j*p + k = j*4 + k (0..15) for the 4x4x4 part
    # W index: k*n + i = k*4 + i (0..15)
    
    # For each of the 7 Strassen terms at the 4x4 level, we need to expand into 7 terms at the 2x2 level.
    # Total: 7 * 7 = 49 terms.
    
    # Let me define the 2x2x2 Strassen terms as a list of (U_coeffs, V_coeffs, W_coeffs)
    # where each is a dict mapping 2x2 index to coefficient.
    
    # 2x2 indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    strassen_2x2 = [
        # Term 1: M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 2: M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # Term 3: M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # Term 4: M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # Term 5: M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # Term 6: M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # Term 7: M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    # For the 4x4x4 Strassen, we have 7 "outer" terms, each involving a 2x2x2 multiplication.
    # For each outer term, we need to determine which 2x2 blocks of A, B, C are involved
    # and with what signs, then expand using the 2x2x2 Strassen decomposition.
    
    # Outer Strassen terms for 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each outer term, the "A" part is a linear combination of 2x2 blocks,
    # the "B" part is a linear combination of 2x2 blocks,
    # and the result contributes to specific 2x2 blocks of C with specific signs.
    
    # Let me define the outer terms as (A_blocks, B_blocks, C_blocks)
    # where A_blocks is a list of (block, coeff), etc.
    
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22 (same for B and C)
    
    outer_strassen = [
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ([((0, 1), (1, 1))], [((0, 1), (1, 1))], [((0, 1), (3, 1))]),
        # M2: (A21+A22)B11 -> C21 += 1
        ([((2, 1), (3, 1))], [((0, 1))], [((2, 1))]),
        # M3: A11(B12-B22) -> C12 += 1
        ([((0, 1))], [((1, 1), (3, -1))], [((1, 1))]),
        # M4: A22(B21-B11) -> C11 += 1, C21 += 1
        ([((3, 1))], [((2, 1), (0, -1))], [((0, 1), (2, 1))]),
        # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        ([((0, 1), (1, 1))], [((3, 1))], [((0, -1), (1, 1))]),
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        ([((2, 1), (0, -1))], [((0, 1), (1, 1))], [((3, 1))]),
        # M7: (A12-A22)(B21+B22) -> C11 += 1, C22 += 1
        ([((1, 1), (3, -1))], [((2, 1), (3, 1))], [((0, 1), (3, 1))]),
    ]
    
    # For each outer term and each inner (2x2x2) term, we combine the coefficients.
    # The 4x4 index for a block (b) and local index (i,j) is:
    #   If b=0 (top-left): row = i, col = j
    #   If b=1 (top-right): row = i, col = j+2
    #   If b=2 (bottom-left): row = i+2, col = j
    #   If b=3 (bottom-right): row = i+2, col = j+2
    
    def block_to_4x4(block, i, j):
        if block == 0:
            return i, j
        elif block == 1:
            return i, j + 2
        elif block == 2:
            return i + 2, j
        elif block == 3:
            return i + 2, j + 2
    
    col = 0
    for outer_idx, (A_blocks, B_blocks, C_blocks) in enumerate(outer_strassen):
        for inner_idx, (U_inner, V_inner, W_inner) in enumerate(strassen_2x2):
            # For this combination, we need to sum over all combinations of
            # A blocks, B blocks, C blocks with their coefficients,
            # multiplied by the inner 2x2x2 coefficients.
            
            # U part: for each (block_a, coeff_a) in A_blocks and each (i,j) in U_inner
            # V part: for each (block_b, coeff_b) in B_blocks and each (j',k) in V_inner
            # W part: for each (block_c, coeff_c) in C_blocks and each (k',i') in W_inner
            
            # Actually, the 2x2x2 decomposition works on the 2x2 matrices.
            # The U_inner gives coefficients for the 2x2 "A" matrix indices.
            # The V_inner gives coefficients for the 2x2 "B" matrix indices.
            # The W_inner gives coefficients for the 2x2 "C" matrix indices.
            
            # For the 4x4 case, each 2x2 index (i,j) in U_inner corresponds to
            # multiple 4x4 indices depending on which block is used.
            
            # Let me build the U, V, W vectors for this term.
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            # U: sum over A blocks and 2x2 indices
            for (block_a, coeff_a) in A_blocks:
                for (i, j), coeff_u in U_inner.items():
                    r, c = block_to_4x4(block_a, i, j)
                    U_vec[r * 4 + c] += coeff_a * coeff_u
            
            # V: sum over B blocks and 2x2 indices
            for (block_b, coeff_b) in B_blocks:
                for (j, k), coeff_v in V_inner.items():
                    r, c = block_to_4x4(block_b, j, k)
                    V_vec[r * 4 + c] += coeff_b * coeff_v
            
            # W: sum over C blocks and 2x2 indices
            for (block_c, coeff_c) in C_blocks:
                for (k, i), coeff_w in W_inner.items():
                    r, c = block_to_4x4(block_c, k, i)
                    W_vec[r * 4 + c] += coeff_c * coeff_w
            
            # Add to global U, V, W
            for idx in range(16):
                U[idx][col] = U_vec[idx]
                V[idx][col] = V_vec[idx]
                W[idx][col] = W_vec[idx]
            col += 1
    
    # Part 2: 4x4x3 schoolbook
    # A is 4x4, B is 4x3, C is 4x3
    # For each (i,j,k) where i in 0..3, j in 0..3, k in 0..2:
    #   U[i*4+j] = 1
    #   V[j*4+k] = 1  (but V has m*p = 4*7 = 28 entries, and we're using columns 4..6 of B)
    #   W[k*4+i] = 1  (but W has p*n = 7*4 = 28 entries, and we're using rows 0..2 of C)
    
    # Wait, the V matrix has shape (m*p, rank) = (28, total_rank)
    # For the 4x4x4 part, we used V indices 0..15 (j*4+k where k in 0..3)
    # For the 4x4x3 part, we use V indices 16..27 (j*4+k where k in 4..6)
    # Similarly for W: 4x4x4 part used W indices 0..15 (k*4+i where k in 0..3)
    # 4x4x3 part uses W indices 16..27 (k*4+i where k in 0..2)
    
    # Actually, let me reconsider the indexing.
    # V has shape (m*p, rank) = (4*7, rank) = (28, rank)
    # V index = j*p + k = j*7 + k where j in 0..3, k in 0..6
    # For the 4x4x4 part (k in 0..3): V index = j*7 + k
    # For the 4x4x3 part (k in 4..6): V index = j*7 + k
    
    # W has shape (p*n, rank) = (7*4, rank) = (28, rank)
    # W index = k*n + i = k*4 + i where k in 0..6, i in 0..3
    # For the 4x4x4 part (k in 0..3): W index = k*4 + i
    # For the 4x4x3 part (k in 0..2): W index = k*4 + i
    
    # I made an error above. Let me redo with correct indexing.
    
    # Reset U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for outer_idx, (A_blocks, B_blocks, C_blocks) in enumerate(outer_strassen):
        for inner_idx, (U_inner, V_inner, W_inner) in enumerate(strassen_2x2):
            U_vec = [0] * 16
            V_vec = [0] * 28
            W_vec = [0] * 28
            
            # U: A is 4x4, index = i*4 + j
            for (block_a, coeff_a) in A_blocks:
                for (i, j), coeff_u in U_inner.items():
                    r, c = block_to_4x4(block_a, i, j)
                    U_vec[r * 4 + c] += coeff_a * coeff_u
            
            # V: B is 4x7, but we only use first 4 columns for this part
            # B index = j*7 + k where j in 0..3, k in 0..3
            for (block_b, coeff_b) in B_blocks:
                for (j, k), coeff_v in V_inner.items():
                    r, c = block_to_4x4(block_b, j, k)
                    # r is the row of B (which is j in 4x4), c is the column (k in 0..3)
                    V_vec[r * 7 + c] += coeff_b * coeff_v
            
            # W: C is 7x4, but we only use first 4 rows for this part
            # C index = k*4 + i where k in 0..3, i in 0..3
            for (block_c, coeff_c) in C_blocks:
                for (k, i), coeff_w in W_inner.items():
                    r, c = block_to_4x4(block_c, k, i)
                    # r is the row of C (which is k in 0..3), c is the column (i in 4x4)
                    W_vec[r * 4 + c] += coeff_c * coeff_w
            
            for idx in range(16):
                U[idx][col] = U_vec[idx]
            for idx in range(28):
                V[idx][col] = V_vec[idx]
                W[idx][col] = W_vec[idx]
            col += 1
    
    # Part 2: 4x4x3 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 7 + (k + 4)][col] = 1  # k in 0..2 maps to column k+4 in B
                W[(k) * 4 + i][col] = 1       # k in 0..2 maps to row k in C
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

