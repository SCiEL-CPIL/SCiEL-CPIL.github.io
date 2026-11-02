"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, safety margin from 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to build a Strassen-based decomposition
    # n=3, m=4, p=6
    # Split into blocks.
    # A is 3x4, B is 4x6.
    # Let's try a decomposition based on known small matrix multiplication ranks.
    # Rank of 2x2x2 is 7 (Strassen).
    # Rank of 1x2x2 is 4 (Schoolbook, since rank(1x2x2) >= 2*2=4? No, rank of (1,2,2) tensor is 4).
    # Actually, the rank of the (1,2,2) matrix multiplication tensor is 4.
    # The rank of the (2,1,2) tensor is 4.
    # The rank of the (2,2,1) tensor is 4.
    
    # Let's see if we can decompose (3,4,6) into smaller blocks.
    # 3 = 2+1, 4 = 2+2, 6 = 4+2?
    # Or 3 = 1+1+1, 4 = 2+2, 6 = 2+2+2?
    
    # Let's try a direct construction using Strassen for 2x2x2 blocks.
    # We can view A as a 3x4 matrix.
    # We can partition the columns of A and rows of B.
    # Let m=4 be split into two parts of size 2.
    # So A = [A_left (3x2), A_right (3x2)]
    # B = [B_top (2x6); B_bottom (2x6)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Now we need to compute two products of (3x2) and (2x6).
    # Can we further decompose (3,2,6)?
    # 3 = 2+1, 6 = 4+2.
    # This seems complicated.
    
    # Alternative: use the known rank for (2,2,2) which is 7.
    # What is the rank of (3,4,6)?
    # Lower bound: n*p = 18? No, lower bound is max(n*m, m*p, n*p) = max(12, 24, 18) = 24.
    # Upper bound: schoolbook 72.
    # Strassen gives upper bound for (2^a, 2^b, 2^c) as 7^{log2(max)}.
    # Here dimensions are not powers of 2.
    
    # Let's try to construct a decomposition of rank 42 or so.
    # 3*4*6 = 72.
    # If we use Strassen for the 2x2x2 part:
    # Suppose we split A into 2x2 and 1x2 blocks?
    # Actually, a simple improvement is to use the fact that rank(2,2,2)=7 < 8.
    # If we can partition the problem into k independent 2x2x2 multiplications, we save (8-7)*k = k rank.
    # How many 2x2x2 blocks can we fit?
    # n=3, m=4, p=6.
    # We can fit floor(3/2)*floor(4/2)*floor(6/2) = 1*2*3 = 6 blocks of size 2x2x2.
    # Each block uses rank 7 instead of 8. Savings: 6 * 1 = 6.
    # Remaining parts:
    # n_rem = 3 - 2*1 = 1.
    # m_rem = 4 - 2*2 = 0.
    # p_rem = 6 - 2*3 = 0.
    # Wait, the partitioning must be consistent.
    # If we take a 2x2x2 block, it consumes 2 rows of A, 2 cols of A (2 rows of B), 2 cols of B (2 rows of C).
    # Let's define the blocks explicitly.
    # A is 3x4. B is 4x6.
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]. Uses rank 7.
    # Block (0,1): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]. Wait, C indices overlap.
    # The blocks in C must be disjoint or added.
    # Matrix multiplication is sum of outer products.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we partition j into sets J1, J2, etc., C = sum_l (A[:,J_l] * B[J_l,:]).
    # Here m=4. Let J1={0,1}, J2={2,3}.
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:].
    # This splits the problem into two (3,2,6) multiplications.
    # Now, can we decompose (3,2,6)?
    # n=3, m=2, p=6.
    # Split n into 2+1?
    # A1 (2x2) * B1 (2x6) -> C1 (2x6). Rank of (2,2,6)?
    # Split p into 2+2+2?
    # This is getting complex to verify exactly.
    
    # Let's stick to the known best: Strassen on 2x2x2 blocks.
    # If we partition m=4 into two 2s:
    # Term 1: A[:,0:2] * B[0:2,:]. This is (3,2,6).
    # Term 2: A[:,2:4] * B[2:4,:]. This is (3,2,6).
    # Total rank = 2 * rank(3,2,6).
    # What is rank(3,2,6)?
    # Lower bound max(6, 12, 18) = 18.
    # Schoolbook 3*2*6 = 36.
    # Can we do better?
    # rank(2,2,2)=7.
    # (3,2,6) can be seen as three (1,2,2) blocks? No.
    # It can be seen as 1*(3,2,2) + 1*(3,2,2) + 1*(3,2,2)?
    # Or 1*(2,2,2) + 1*(1,2,2)?
    # Let's just compute rank(3,2,6) using a simple heuristic or known bounds.
    # Actually, rank(3,2,6) is likely 20 or 24.
    # If rank(3,2,6) = 24, total rank = 48.
    # If rank(3,2,6) = 20, total rank = 40.
    
    # Let's try a different split: n=3 -> 2+1.
    # C_top (2x6) = A_top (2x4) * B (4x6).
    # C_bottom (1x6) = A_bottom (1x4) * B (4x6).
    # Rank(2,4,6) + Rank(1,4,6).
    # Rank(1,4,6) = 24 (schoolbook, since rank(1,m,p) = m*p? No, rank(1,m,p) is at least m and p, but schoolbook is m*p. Actually rank(1,m,p) = m*p is not always minimal, but for small m,p it is often close. For 1x4x6, schoolbook is 24. Is there a better way? The tensor is x_i y_j z_k. Rank is at least max(m,p). But it's the "slice rank". Rank of (1,m,p) tensor is m*p if we consider it as a matrix in one mode? No, tensor rank.
    # The matrix multiplication tensor (1,m,p) has rank m*p. Because C_k = sum_j A_j B_jk. This is a sum of m*p rank-1 terms?
    # Actually, for (1,m,p), the rank is m*p.
    # So Rank(1,4,6) = 24.
    # What is Rank(2,4,6)?
    # Split m=4 into 2+2.
    # Rank(2,4,6) <= Rank(2,2,6) + Rank(2,2,6).
    # Rank(2,2,6): split p=6 into 2+2+2.
    # Rank(2,2,2) = 7.
    # So Rank(2,2,6) <= 3 * 7 = 21.
    # So Rank(2,4,6) <= 2 * 21 = 42.
    # Total rank <= 42 + 24 = 66.
    # This is better than 72.
    
    # Let's implement this:
    # 1. Split n=3 into 2 and 1.
    # 2. For n=2 part: Split m=4 into 2 and 2.
    #    2a. For m=2 part: Split p=6 into 2, 2, 2.
    #        2a-i. Use Strassen (rank 7) for each (2,2,2).
    #    2b. Sum them up.
    # 3. For n=1 part: Use schoolbook (rank 24).
    # Total rank: 3 * 7 (for first m-block) + 3 * 7 (for second m-block) + 24 = 21 + 21 + 24 = 66.
    
    # Let's build the matrices for this decomposition.
    # We need to map the local indices to global indices.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off, u_col, v_col, w_col):
        nonlocal col
        # Strassen for 2x2x2
        # M1 = A11*(B21-B22)
        # M2 = (A11+A12)*B22
        # M3 = (A11-A21)*B11
        # M4 = A22*(B11-B12)
        # M5 = (A11+A21)*B22
        # M6 = (A11+A12)*(B11-B21)
        # M7 = (A21-A12)*(B11+B22)
        # C11 = M4+M5+M6-M2+M7
        # C12 = M1+M5
        # C21 = M2+M3
        # C22 = M1+M3+M4+M7
        
        # We need to add 7 terms.
        # Each term is U_vec, V_vec, W_vec.
        # U is (2x2), indices (i,j). Global U index: (i_off+i)*m + (j_off+j).
        # V is (2x2), indices (j,k). Global V index: (j_off+j)*p + (k_off+k).
        # W is (2x2), indices (k,i). Global W index: (k_off+k)*n + (i_off+i).
        
        # Term 1: M1 = A11 * (B21 - B22)
        # U: A11 -> i=0, j=0. Global U: (i_off+0)*m + (j_off+0) = i_off*m + j_off. Val 1.
        # V: B21 -> j=1, k=0. Global V: (j_off+1)*p + (k_off+0) = (j_off+1)*p + k_off. Val 1.
        #     B22 -> j=1, k=1. Global V: (j_off+1)*p + (k_off+1) = (j_off+1)*p + k_off+1. Val -1.
        # W: C11 -> k=0, i=0. Global W: (k_off+0)*n + (i_off+0) = k_off*n + i_off. Val 1.
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1
            
        # T1: A11, (B21-B22), C11
        add_term([(i_off)*m + (j_off)], 
                 [((j_off+1)*p + (k_off)), ((j_off+1)*p + (k_off+1))], 
                 [((k_off)*n + (i_off))])
        # Note: add_term adds 1 to each. We need to handle signs.
        # Let's rewrite add_term to take coefficients.
        
        # Let's restart the helper to handle coefficients.
        
    # Redo helper
    col = 0
    def add_term(u_list, v_list, w_list):
        # u_list: list of (index, coeff)
        # v_list: list of (index, coeff)
        # w_list: list of (index, coeff)
        nonlocal col
        for idx, c in u_list:
            U[idx][col] += c
        for idx, c in v_list:
            V[idx][col] += c
        for idx, c in w_list:
            W[idx][col] += c
        col += 1

    # Strassen (2,2,2) at offset (i_off, j_off, k_off)
    def add_strassen_block(i_off, j_off, k_off):
        # Indices relative to block:
        # A: (0,0),(0,1),(1,0),(1,1)
        # B: (0,0),(0,1),(1,0),(1,1)
        # C: (0,0),(0,1),(1,0),(1,1)
        
        # M1 = A11 * (B21 - B22) -> C11
        # A11: (0,0) -> (i_off, j_off) -> idx (i_off)*m + (j_off)
        # B21: (1,0) -> (j_off+1, k_off) -> idx (j_off+1)*p + k_off
        # B22: (1,1) -> (j_off+1, k_off+1) -> idx (j_off+1)*p + k_off+1
        # C11: (0,0) -> (k_off, i_off) -> idx k_off*n + i_off
        add_term(
            [((i_off)*m + (j_off), 1)],
            [((j_off+1)*p + (k_off), 1), ((j_off+1)*p + (k_off+1), -1)],
            [((k_off)*n + (i_off), 1)]
        )
        
        # M2 = (A11 + A12) * B22 -> C21, C11 (subtracted)
        # A11: (0,0), A12: (0,1) -> (i_off, j_off+1)
        # B22: (1,1) -> (j_off+1, k_off+1)
        # C21: (1,0) -> (k_off+1, i_off)
        # C11: (0,0) -> (k_off, i_off)
        # Wait, Strassen formulas:
        # M2 = (A11+A12)B22
        # C11 uses -M2. C21 uses +M2.
        # So we add M2 with W coeff 1 for C21 and -1 for C11?
        # No, the term is a rank-1 update to the tensor.
        # The tensor is sum of U_c V_c W_c.
        # W_c is a vector.
        # For M2, the W part has non-zeros at C21 and C11?
        # No, M2 is an intermediate scalar.
        # C21 = M2 + M3. So M2 contributes to C21.
        # C11 = M4+M5+M6-M2+M7. So M2 contributes to C11 with -1.
        # So the W vector for the M2 term has +1 at C21 and -1 at C11.
        
        add_term(
            [((i_off)*m + (j_off), 1), ((i_off)*m + (j_off+1), 1)],
            [((j_off+1)*p + (k_off+1), 1)],
            [((k_off+1)*n + (i_off), 1), ((k_off)*n + (i_off), -1)]
        )
        
        # M3 = (A11 - A21) * B11 -> C21
        # A11: (0,0), A21: (1,0) -> (i_off+1, j_off)
        # B11: (0,0) -> (j_off, k_off)
        # C21: (1,0) -> (k_off+1, i_off)
        add_term(
            [((i_off)*m + (j_off), 1), ((i_off+1)*m + (j_off), -1)],
            [((j_off)*p + (k_off), 1)],
            [((k_off+1)*n + (i_off), 1)]
        )
        
        # M4 = A22 * (B11 - B12) -> C11, C22
        # A22: (1,1) -> (i_off+1, j_off+1)
        # B11: (0,0), B12: (0,1) -> (j_off, k_off+1)
        # C11: (0,0), C22: (1,1) -> (k_off+1, i_off+1)
        add_term(
            [((i_off+1)*m + (j_off+1), 1)],
            [((j_off)*p + (k_off), 1), ((j_off)*p + (k_off+1), -1)],
            [((k_off)*n + (i_off), 1), ((k_off+1)*n + (i_off+1), 1)]
        )
        
        # M5 = (A11 + A21) * B22 -> C11, C12
        # A11: (0,0), A21: (1,0)
        # B22: (1,1)
        # C11: (0,0), C12: (0,1) -> (k_off, i_off+1)
        add_term(
            [((i_off)*m + (j_off), 1), ((i_off+1)*m + (j_off), 1)],
            [((j_off+1)*p + (k_off+1), 1)],
            [((k_off)*n + (i_off), 1), ((k_off)*n + (i_off+1), 1)]
        )
        
        # M6 = (A11 + A12) * (B11 - B21) -> C11
        # A11: (0,0), A12: (0,1)
        # B11: (0,0), B21: (1,0)
        # C11: (0,0)
        add_term(
            [((i_off)*m + (j_off), 1), ((i_off)*m + (j_off+1), 1)],
            [((j_off)*p + (k_off), 1), ((j_off+1)*p + (k_off), -1)],
            [((k_off)*n + (i_off), 1)]
        )
        
        # M7 = (A21 - A12) * (B11 + B22) -> C11, C22
        # A21: (1,0), A12: (0,1)
        # B11: (0,0), B22: (1,1)
        # C11: (0,0), C22: (1,1)
        add_term(
            [((i_off+1)*m + (j_off), 1), ((i_off)*m + (j_off+1), -1)],
            [((j_off)*p + (k_off), 1), ((j_off+1)*p + (k_off+1), 1)],
            [((k_off)*n + (i_off), 1), ((k_off+1)*n + (i_off+1), 1)]
        )
        
    # Now, construct the full decomposition.
    # n=3, m=4, p=6.
    # Split n into 2 (top) and 1 (bottom).
    # i_off=0 for top 2, i_off=2 for bottom 1.
    
    # Top part (i=0,1):
    # Split m=4 into 2 (left) and 2 (right).
    # j_off=0 for left 2, j_off=2 for right 2.
    
    # For each (i_block, j_block) of size 2x2:
    # Split p=6 into three blocks of size 2.
    # k_off=0, 2, 4.
    
    # So we have 2 (i_blocks) * 2 (j_blocks) * 3 (k_blocks) = 12 Strassen blocks.
    # Each Strassen block is rank 7.
    # Total rank for top part = 12 * 7 = 84.
    # Wait, earlier I estimated 42. Why?
    # Because I assumed Rank(2,2,6) = 21.
    # 21 = 3 * 7.
    # So Rank(2,4,6) = 2 * 21 = 42.
    # Yes, 12 blocks * 7 = 84? No.
    # The blocks are NOT independent in the way I'm adding them.
    # When I say "Split p=6 into 2,2,2", I mean I compute three separate (2,2,2) products and sum them.
    # Product 1: A[i_block, j_block] * B[j_block, k_block1].
    # Product 2: A[i_block, j_block] * B[j_block, k_block2].
    # Product 3: A[i_block, j_block] * B[j_block, k_block3].
    # Sum = A[i_block, j_block] * B[j_block, :].
    # This is valid.
    # So for a fixed (i_block, j_block) of size 2x2, we have three Strassen blocks.
    # Rank = 3 * 7 = 21.
    # We have 4 such (i_block, j_block) pairs (2 i_blocks, 2 j_blocks).
    # Total rank for top part = 4 * 21 = 84.
    # This is NOT better than 72!
    # My previous calculation: Rank(2,4,6) <= 42.
    # Let's re-evaluate Rank(2,4,6).
    # Split m=4 into 2+2.
    # C = A[:,0:2]B[0:2,:] + A[:,2:4]B[2:4,:].
    # This is sum of two (2,2,6) products.
    # Rank(2,2,6).
    # Split p=6 into 2+2+2.
    # Product = sum of three (2,2,2) products.
    # Rank(2,2,2) = 7.
    # So Rank(2,2,6) <= 3 * 7 = 21.
    # So Rank(2,4,6) <= 2 * 21 = 42.
    # Why did I get 84?
    # Because I used Strassen for EACH (2,2,2) block independently.
    # That is exactly 3 * 7 = 21 per (2,2) A-B pair.
    # And there are 2 (2,2) A-B pairs (left and right).
    # So 2 * 21 = 42.
    # Where did 12 blocks come from?
    # i_blocks: 1 (since n=2 in top part).
    # j_blocks: 2.
    # k_blocks: 3.
    # Total 1*2*3 = 6 Strassen blocks.
    # Rank = 6 * 7 = 42.
    # Ah, I miscounted i_blocks. n=3, top is 2. So only 1 i_block of size 2.
    # Correct.
    
    # So Top Part Rank = 42.
    
    # Bottom Part (i=2):
    # n=1, m=4, p=6.
    # Rank(1,4,6) = 24 (Schoolbook).
    # Let's verify if we can do better.
    # Rank(1,m,p) is the tensor rank of x_i y_j z_k.
    # It is known that rank(1,m,p) = m*p.
    # So 24 is optimal for this sub-block.
    
    # Total Rank = 42 + 24 = 66.
    
    # Let's code the 42 rank part.
    # i_off = 0.
    # j_off = 0, 2.
    # k_off = 0, 2, 4.
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            add_strassen_block(0, j_off, k_off)
    
    # Now code the 24 rank part.
    # i_off = 2.
    # Schoolbook for (1,4,6).
    # A has 1 row, 4 cols. B has 4 rows, 6 cols.
    # C has 1 row, 6 cols.
    # C[k] = sum_j A[j] B[j,k].
    # 24 terms.
    for j in range(4):
        for k in range(6):
            # U: (i=2, j=j) -> 2*m + j = 8 + j.
            # V: (j=j, k=k) -> j*p + k = 6j + k.
            # W: (k=k, i=2) -> k*n + 2 = 3k + 2.
            add_term(
                [((2)*m + j, 1)],
                [((j)*p + k, 1)],
                [((k)*n + 2, 1)]
            )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to schoolbook if we run out of time (unlikely)
        pass
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
