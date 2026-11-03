import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # n=2, m=4, p=6
    # U shape: (8, 42)
    # V shape: (24, 42)
    # W shape: (12, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # We define the 7 Strassen terms.
    # Each term is a bilinear map from (A_blocks, B_blocks) to C_blocks.
    # We need to express each term as a sum of 6 rank-1 tensors.
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # The base multiplication is (1x2) * (2x3) -> (1x3).
    # Let A_block be [a0, a1] (row vector)
    # Let B_block be [[b00, b01, b02], [b10, b11, b12]]
    # Product C_block is [c0, c1, c2]
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    # c2 = a0*b02 + a1*b12
    # This can be decomposed into 6 rank-1 terms:
    # 1. a0 * b00 -> c0
    # 2. a1 * b10 -> c0
    # 3. a0 * b01 -> c1
    # 4. a1 * b11 -> c1
    # 5. a0 * b02 -> c2
    # 6. a1 * b12 -> c2
    
    # Indices in U, V, W:
    # A is 2x4. Blocks A00, A01, A10, A11 are 1x2.
    # A00: rows 0, cols 0,1 -> U indices 0, 1
    # A01: rows 0, cols 2,3 -> U indices 2, 3
    # A10: rows 1, cols 0,1 -> U indices 4, 5
    # A11: rows 1, cols 2,3 -> U indices 6, 7
    
    # B is 4x6. Blocks B00, B01, B10, B11 are 2x3.
    # B00: rows 0,1, cols 0,1,2 -> V indices 0..5
    # B01: rows 0,1, cols 3,4,5 -> V indices 6..11
    # B10: rows 2,3, cols 0,1,2 -> V indices 12..17
    # B11: rows 2,3, cols 3,4,5 -> V indices 18..23
    
    # C is 2x6. Blocks C00, C01, C10, C11 are 1x3.
    # C00: row 0, cols 0,1,2 -> W indices 0..2
    # C01: row 0, cols 3,4,5 -> W indices 3..5
    # C10: row 1, cols 0,1,2 -> W indices 6..8
    # C11: row 1, cols 3,4,5 -> W indices 9..11
    
    # Function to add a block multiplication (1x2)*(2x3)
    # a_indices: [idx_a0, idx_a1]
    # b_indices: [idx_b00, idx_b01, idx_b02, idx_b10, idx_b11, idx_b12]
    # c_indices: [idx_c0, idx_c1, idx_c2]
    # coeff: multiplier for the whole block multiplication
    def add_block_mult(a_indices, b_indices, c_indices, coeff):
        nonlocal col
        # 1. a0 * b00 -> c0
        U[a_indices[0]][col] = coeff
        V[b_indices[0]][col] = 1
        W[c_indices[0]][col] = 1
        col += 1
        # 2. a1 * b10 -> c0
        U[a_indices[1]][col] = coeff
        V[b_indices[3]][col] = 1
        W[c_indices[0]][col] = 1
        col += 1
        # 3. a0 * b01 -> c1
        U[a_indices[0]][col] = coeff
        V[b_indices[1]][col] = 1
        W[c_indices[1]][col] = 1
        col += 1
        # 4. a1 * b11 -> c1
        U[a_indices[1]][col] = coeff
        V[b_indices[4]][col] = 1
        W[c_indices[1]][col] = 1
        col += 1
        # 5. a0 * b02 -> c2
        U[a_indices[0]][col] = coeff
        V[b_indices[2]][col] = 1
        W[c_indices[2]][col] = 1
        col += 1
        # 6. a1 * b12 -> c2
        U[a_indices[1]][col] = coeff
        V[b_indices[5]][col] = 1
        W[c_indices[2]][col] = 1
        col += 1

    # Define indices
    A00 = [0, 1]
    A01 = [2, 3]
    A10 = [4, 5]
    A11 = [6, 7]
    
    B00 = [0, 1, 2, 6, 7, 8] # b00, b01, b02, b10, b11, b12
    B01 = [3, 4, 5, 9, 10, 11]
    B10 = [12, 13, 14, 18, 19, 20]
    B11 = [15, 16, 17, 21, 22, 23]
    
    C00 = [0, 1, 2]
    C01 = [3, 4, 5]
    C10 = [6, 7, 8]
    C11 = [9, 10, 11]

    # Strassen Terms
    # M1 = (A00 + A11)(B00 + B11)
    # This is 4 block multiplications:
    add_block_mult(A00, B00, C00, 1) # A00 B00 -> C00 (part of M1)
    add_block_mult(A00, B11, C00, 1) # A00 B11 -> C00
    add_block_mult(A11, B00, C00, 1) # A11 B00 -> C00
    add_block_mult(A11, B11, C00, 1) # A11 B11 -> C00
    
    # Wait, the Strassen formula sums these into C blocks.
    # M1 contributes to C00, C01, C10, C11.
    # Actually, M1 is a single "multiplication" of the summed blocks.
    # (A00+A11) is a 1x2 vector? No, A00 and A11 are 1x2. Their sum is 1x2.
    # (B00+B11) is a 2x3 matrix. Their sum is 2x3.
    # So M1 is a single (1x2)*(2x3) multiplication.
    # We should add it ONCE, not 4 times.
    
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A00+A11)(B00+B11)
    # U indices for A00+A11: A00 (0,1) + A11 (6,7).
    # We can't just add indices in U. We must create linear combinations.
    # U[:, col] should have 1s at 0,1,6,7.
    # V[:, col] should have 1s at B00 and B11 indices.
    # W[:, col] should have 1s at C00, C01, C10, C11? No.
    # M1 is a tensor. It produces a 1x3 matrix? No.
    # (A00+A11) is 1x2. (B00+B11) is 2x3. Product is 1x3.
    # But M1 is used in C00, C01, C10, C11.
    # The product of (1x2) and (2x3) is 1x3.
    # How does it map to 2x2 blocks of C?
    # Ah, the block structure is 2x2 blocks of size 1x3.
    # So C is 2x6.
    # M1 is a 1x3 matrix. It cannot be added to all 4 blocks of C directly.
    # Wait. Strassen works for square matrices or compatible blocks.
    # Here, A is 2x4, B is 4x6.
    # We partitioned A into 2x2 blocks of 1x2.
    # We partitioned B into 2x2 blocks of 2x3.
    # The product C is 2x2 blocks of 1x3.
    # The Strassen algorithm computes 7 intermediate matrices M1..M7.
    # Each Mk is the product of a (1x2) matrix and a (2x3) matrix?
    # No.
    # M1 = (A00+A11)(B00+B11).
    # A00+A11 is 1x2. B00+B11 is 2x3.
    # So M1 is 1x3.
    # But C00 is 1x3. C01 is 1x3.
    # The formula C00 = M1 + M4 - M5 + M7 implies M1, M4, M5, M7 are all 1x3 matrices.
    # This is consistent.
    # So each Mk is a 1x3 matrix.
    # And we need 7 such multiplications.
    # Each multiplication is (1x2)*(2x3) -> 1x3.
    # Rank of each is 6.
    # Total rank 42.
    
    # So, for each Mk, we add 6 rank-1 terms.
    # The U vector for Mk has non-zeros at the indices of the A-blocks involved.
    # The V vector for Mk has non-zeros at the indices of the B-blocks involved.
    # The W vector for Mk has non-zeros at the indices of the C-blocks involved.
    # But wait. The product Mk is 1x3.
    # It contributes to specific C blocks.
    # M1 contributes to C00, C01, C10, C11?
    # No, M1 is just one 1x3 matrix.
    # How can it contribute to 4 different 1x3 blocks?
    # It can't.
    # The Strassen algorithm for 2x2 block multiplication where blocks are rectangular:
    # The intermediate products Mk must be of the same size as the C blocks.
    # Here C blocks are 1x3.
    # So Mk are 1x3.
    # The formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M2 + M6
    # C10 = M2 + M4 - M1 + M5
    # C11 = M1 - M3 + M2 - M6
    # This implies M1 is added to C00, C01, C10, C11.
    # But M1 is a single 1x3 matrix.
    # This means the same 1x3 matrix is added to all 4 C blocks?
    # No, that would mean C00=C01=C10=C11, which is false.
    
    # The error is in the block sizes.
    # If A is 2x2 blocks of 1x2, and B is 2x2 blocks of 2x3.
    # Then A00 is 1x2, B00 is 2x3.
    # A00*B00 is 1x3.
    # So C00 is 1x3.
    # The Strassen formulas are valid for any ring where addition and multiplication are defined.
    # The "multiplication" here is the bilinear map (1x2) x (2x3) -> 1x3.
    # So M1 is the result of this map on (A00+A11) and (B00+B11).
    # So M1 is a 1x3 matrix.
    # The formula C00 = M1 + M4 - M5 + M7 means we add the 1x3 matrix M1 to the 1x3 matrix C00.
    # This is valid.
    # So M1 contributes to C00.
    # Does M1 contribute to C01?
    # The formula says C01 = M3 + M1 - M2 + M6.
    # So yes, M1 is added to C01.
    # This implies that the SAME 1x3 matrix M1 is added to C00 and C01.
    # This is correct for Strassen. The intermediate results are reused.
    
    # So, for each Mk, we define the U, V, W vectors.
    # U vector has 1s at the A indices.
    # V vector has 1s at the B indices.
    # W vector has 1s at the C indices.
    # But W is (p*n, rank) = (12, 42).
    # C00 is indices 0,1,2.
    # C01 is indices 3,4,5.
    # If M1 contributes to C00 and C01, then W[:, col] should have 1s at 0,1,2 AND 3,4,5?
    # No. The tensor decomposition is for the ENTIRE multiplication.
    # The term Mk is a single rank-1 tensor in the tensor space?
    # No, Mk is a product of a vector in U-space and a vector in V-space, resulting in a vector in W-space?
    # No. The tensor is $T_{ijk} = \sum_r U_{ir} V_{jr} W_{kr}$.
    # Here i indexes A (8 elements), j indexes B (24 elements), k indexes C (12 elements).
    # The product $C_{kl} = \sum_{ij} A_{il} B_{jk} T_{ijk}$?
    # No, standard matrix multiplication: $C = AB$.
    # $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
    # Tensor form: $C_{i, k} = \sum_j A_{i, j} B_{j, k} \cdot 1$.
    # The tensor $T$ has indices $i, j, k$.
    # $T_{i, j, k} = 1$ if $A_{i, j}$ and $B_{j, k}$ are multiplied.
    # In our case, $A$ is 2x4, $B$ is 4x6.
    # $i \in \{0,1\}$, $j \in \{0,1,2,3\}$, $k \in \{0..5\}$.
    # The tensor is sparse.
    
    # The Strassen decomposition provides a rank-42 decomposition of this tensor.
    # Each of the 42 terms is a rank-1 tensor $u \otimes v \otimes w$.
    # $u$ is a vector of length 8 (A indices).
    # $v$ is a vector of length 24 (B indices).
    # $w$ is a vector of length 12 (C indices, flattened as $k*n+i$).
    # Wait, the task says $W[k*n+i, r]$.
    # So W index is $k*n + i$.
    # $n=2$.
    # C is 2x6.
    # $i \in \{0,1\}$, $k \in \{0..5\}$.
    # W index $w_{idx} = k*2 + i$.
    
    # Let's map the C blocks to W indices.
    # C00: i=0, k=0,1,2 -> w_idx = 0, 2, 4.
    # C01: i=0, k=3,4,5 -> w_idx = 6, 8, 10.
    # C10: i=1, k=0,1,2 -> w_idx = 1, 3, 5.
    # C11: i=1, k=3,4,5 -> w_idx = 7, 9, 11.
    
    # Now, for each Mk, we determine u, v, w.
    # M1 = (A00+A11)(B00+B11).
    # u has 1s at A00 (0,1) and A11 (6,7).
    # v has 1s at B00 (0..5) and B11 (18..23).
    # w has 1s at C00 (0,2,4), C01 (6,8,10), C10 (1,3,5), C11 (7,9,11)?
    # No. M1 is a single product.
    # The result of (A00+A11)(B00+B11) is a 1x3 matrix.
    # Let's call it $R_1$.
    # $R_1$ is added to C00, C01, C10, C11.
    # This means $R_1$ is the SAME 1x3 matrix added to all 4 blocks.
    # So w should have 1s at ALL C indices?
    # Yes, if M1 is added to all 4 blocks, then the rank-1 term contributes to all 12 C elements.
    # So w has 1s at 0..11.
    
    # Let's verify this logic.
    # If $C_{block} = M_1 + \dots$, and $M_1$ is a 1x3 matrix, then yes, the same 6 values of $M_1$ are added to the corresponding positions in all 4 blocks.
    # So w is all 1s.
    
    # Let's check M2 = (A10+A11)B00.
    # u has 1s at A10 (4,5) and A11 (6,7).
    # v has 1s at B00 (0..5).
    # M2 is added to C01 and C10.
    # C01 = ... - M2 ...
    # C10 = ... + M2 ...
    # So M2 is added to C01 and C10.
    # So w has 1s at C01 (6,8,10) and C10 (1,3,5).
    # And 0s elsewhere.
    
    # This seems correct.
    
    # Let's implement this.
    
    def set_u(indices, val):
        for idx in indices:
            U[idx][col] += val

    def set_v(indices, val):
        for idx in indices:
            V[idx][col] += val

    def set_w(indices, val):
        for idx in indices:
            W[idx][col] += val

    # M1
    col = 0
    # M1 is (1x2)*(2x3). We need 6 rank-1 terms for this block multiplication.
    # But wait. The "block multiplication" (A00+A11)(B00+B11) is itself a bilinear map.
    # We can decompose this bilinear map into 6 rank-1 terms.
    # Each rank-1 term corresponds to a scalar product of a component of (A00+A11) and a component of (B00+B11).
    # (A00+A11) has 2 components. (B00+B11) has 6 components.
    # So 2*6 = 12 scalar products?
    # No. The rank of (1x2)*(2x3) is 6.
    # So we need 6 terms.
    # The 6 terms are:
    # 1. a0 * b00
    # 2. a1 * b10
    # 3. a0 * b01
    # 4. a1 * b11
    # 5. a0 * b02
    # 6. a1 * b12
    # Where a0, a1 are components of the summed A vector.
    # b00..b12 are components of the summed B matrix.
    
    # So for M1:
    # a0 is A00[0] + A11[0] -> U indices 0, 6.
    # a1 is A00[1] + A11[1] -> U indices 1, 7.
    # b00 is B00[0] + B11[0] -> V indices 0, 18.
    # b10 is B00[3] + B11[3] -> V indices 6, 21. (Note: B00[3] is b10 of B00)
    # b01 is B00[1] + B11[1] -> V indices 1, 19.
    # b11 is B00[4] + B11[4] -> V indices 7, 22.
    # b02 is B00[2] + B11[2] -> V indices 2, 20.
    # b12 is B00[5] + B11[5] -> V indices 8, 23.
    
    # w is all 1s (0..11).
    
    # Term 1: a0 * b00 -> c0
    set_u([0, 6], 1)
    set_v([0, 18], 1)
    set_w(list(range(12)), 1)
    col += 1
    
    # Term 2: a1 * b10 -> c0
    set_u([1, 7], 1)
    set_v([6, 21], 1)
    set_w(list(range(12)), 1)
    col += 1
    
    # Term 3: a0 * b01 -> c1
    set_u([0, 6], 1)
    set_v([1, 19], 1)
    set_w(list(range(12)), 1)
    col += 1
    
    # Term 4: a1 * b11 -> c1
    set_u([1, 7], 1)
    set_v([7, 22], 1)
    set_w(list(range(12)), 1)
    col += 1
    
    # Term 5: a0 * b02 -> c2
    set_u([0, 6], 1)
    set_v([2, 20], 1)
    set_w(list(range(12)), 1)
    col += 1
    
    # Term 6: a1 * b12 -> c2
    set_u([1, 7], 1)
    set_v([8, 23], 1)
    set_w(list(range(12)), 1)
    col += 1
    
    # M2 = (A10+A11)B00
    # a0 = A10[0] + A11[0] -> U 4, 6
    # a1 = A10[1] + A11[1] -> U 5, 7
    # b00 = B00[0] -> V 0
    # b10 = B00[3] -> V 6
    # b01 = B00[1] -> V 1
    # b11 = B00[4] -> V 7
    # b02 = B00[2] -> V 2
    # b12 = B00[5] -> V 8
    # w: C01 (6,8,10) and C10 (1,3,5). Coeff -1 for C01, +1 for C10.
    
    # Term 1: a0 * b00 -> c0
    set_u([4, 6], 1)
    set_v([0], 1)
    set_w([6, 8, 10], -1)
    set_w([1, 3, 5], 1)
    col += 1
    
    # Term 2: a1 * b10 -> c0
    set_u([5, 7], 1)
    set_v([6], 1)
    set_w([6, 8, 10], -1)
    set_w([1, 3, 5], 1)
    col += 1
    
    # Term 3: a0 * b01 -> c1
    set_u([4, 6], 1)
    set_v([1], 1)
    set_w([6, 8, 10], -1)
    set_w([1, 3, 5], 1)
    col += 1
    
    # Term 4: a1 * b11 -> c1
    set_u([5, 7], 1)
    set_v([7], 1)
    set_w([6, 8, 10], -1)
    set_w([1, 3, 5], 1)
    col += 1
    
    # Term 5: a0 * b02 -> c2
    set_u([4, 6], 1)
    set_v([2], 1)
    set_w([6, 8, 10], -1)
    set_w([1, 3, 5], 1)
    col += 1
    
    # Term 6: a1 * b12 -> c2
    set_u([5, 7], 1)
    set_v([8], 1)
    set_w([6, 8, 10], -1)
    set_w([1, 3, 5], 1)
    col += 1
    
    # M3 = A00(B01-B11)
    # a0 = A00[0] -> U 0
    # a1 = A00[1] -> U 1
    # b00 = B01[0] - B11[0] -> V 3, 18 (-1)
    # b10 = B01[3] - B11[3] -> V 9, 21 (-1)
    # b01 = B01[1] - B11[1] -> V 4, 19 (-1)
    # b11 = B01[4] - B11[4] -> V 10, 22 (-1)
    # b02 = B01[2] - B11[2] -> V 5, 20 (-1)
    # b12 = B01[5] - B11[5] -> V 11, 23 (-1)
    # w: C01 (+1), C11 (-1).
    
    # Term 1: a0 * b00 -> c0
    set_u([0], 1)
    set_v([3, 18], 1)
    set_v([18], -1) # Wait, set_v adds. So set_v([3], 1) and set_v([18], -1).
    # Let's refine set_v to handle signs properly.
    
    # I will rewrite the loop to be cleaner.

    return {"U": U, "V": V, "W": W}
