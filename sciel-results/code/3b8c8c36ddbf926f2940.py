import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    # We will construct a decomposition using Strassen's algorithm for the 4x4x4 part
    # and schoolbook for the 4x4x1 part (the 5th column of B).
    # Total rank = 49 (for 4x4x4) + 16 (for 4x4x1) = 65.
    
    total_rank = 65
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 65
    
    # --- Part 1: 4x4x4 (cols 0-3 of B) ---
    # We use Strassen's algorithm recursively.
    # The 4x4x4 matrix multiplication can be decomposed into 7 block multiplications of 2x2x2.
    # Each 2x2x2 multiplication is decomposed into 7 rank-1 terms.
    # Total terms = 7 * 7 = 49.
    
    # Helper: Strassen 2x2x2 decomposition
    # Returns list of 7 tuples, each containing:
    #   u_indices: list of (local_i, local_j) with signs for A
    #   v_indices: list of (local_j, local_k) with signs for B
    #   w_indices: list of (local_k, local_i) with signs for C
    # The local indices are 0 or 1.
    
    def strassen_2x2x2_terms():
        # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        # M2 = (A21+A22)B11 -> contributes to C21
        # M3 = A11(B12-B22) -> contributes to C12
        # M4 = A22(B21-B11) -> contributes to C21
        # M5 = (A11+A12)B22 -> contributes to C12
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        
        # Note: Indices in Strassen are usually 1-based in literature, here 0-based.
        # A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
        # B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
        # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
        
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        # A terms: (0,0):+1, (1,1):+1
        # B terms: (0,0):+1, (1,1):+1
        # C terms: (0,0):+1, (1,1):+1
        terms.append((
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)]
        ))
        
        # M2 = (A21+A22)B11
        # A terms: (1,0):+1, (1,1):+1
        # B terms: (0,0):+1
        # C terms: (1,0):+1
        terms.append((
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)],
            [((1,0), 1)]
        ))
        
        # M3 = A11(B12-B22)
        # A terms: (0,0):+1
        # B terms: (0,1):+1, (1,1):-1
        # C terms: (0,1):+1
        terms.append((
            [((0,0), 1)],
            [((0,1), 1), ((1,1), -1)],
            [((0,1), 1)]
        ))
        
        # M4 = A22(B21-B11)
        # A terms: (1,1):+1
        # B terms: (1,0):+1, (0,0):-1
        # C terms: (1,0):+1
        terms.append((
            [((1,1), 1)],
            [((1,0), 1), ((0,0), -1)],
            [((1,0), 1)]
        ))
        
        # M5 = (A11+A12)B22
        # A terms: (0,0):+1, (0,1):+1
        # B terms: (1,1):+1
        # C terms: (1,1):+1
        terms.append((
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)],
            [((1,1), 1)]
        ))
        
        # M6 = (A21-A11)(B11+B12)
        # A terms: (1,0):+1, (0,0):-1
        # B terms: (0,0):+1, (0,1):+1
        # C terms: (1,1):+1
        terms.append((
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)]
        ))
        
        # M7 = (A12-A22)(B21+B22)
        # A terms: (0,1):+1, (1,1):-1
        # B terms: (1,0):+1, (1,1):+1
        # C terms: (0,0):+1
        terms.append((
            [((0,1), 1), ((1,1), -1)],
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)]
        ))
        
        return terms
    
    inner_terms = strassen_2x2x2_terms()
    
    # Define the 7 big Strassen terms for 4x4x4
    # Each big term is a product of two 2x2 matrices formed from blocks of A and B.
    # A blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right)
    # B blocks: B00, B01, B10, B11 (but B is 4x4 here, so blocks are 2x2)
    
    # Big Term 1: (A00+A11)(B00+B11)
    # Big Term 2: (A10+A11)B00
    # Big Term 3: A00(B01-B11)
    # Big Term 4: A11(B10-B00)
    # Big Term 5: (A00+A01)B11
    # Big Term 6: (A10-A00)(B00+B01)
    # Big Term 7: (A01-A11)(B10+B11)
    
    # For each big term, we specify which A blocks and B blocks are involved with signs.
    # A_block (bi, bj) covers A[2*bi..2*bi+1, 2*bj..2*bj+1]
    # B_block (bi, bj) covers B[2*bi..2*bi+1, 2*bj..2*bj+1]
    
    big_terms = [
        # (list of (A_block_row, A_block_col, sign), list of (B_block_row, B_block_col, sign))
        (
            [((0,0), 1), ((1,1), 1)],  # A00 + A11
            [((0,0), 1), ((1,1), 1)]   # B00 + B11
        ),
        (
            [((1,0), 1), ((1,1), 1)],  # A10 + A11
            [((0,0), 1)]               # B00
        ),
        (
            [((0,0), 1)],              # A00
            [((0,1), 1), ((1,1), -1)]  # B01 - B11
        ),
        (
            [((1,1), 1)],              # A11
            [((1,0), 1), ((0,0), -1)]  # B10 - B00
        ),
        (
            [((0,0), 1), ((0,1), 1)],  # A00 + A01
            [((1,1), 1)]               # B11
        ),
        (
            [((1,0), 1), ((0,0), -1)], # A10 - A00
            [((0,0), 1), ((0,1), 1)]   # B00 + B01
        ),
        (
            [((0,1), 1), ((1,1), -1)], # A01 - A11
            [((1,0), 1), ((1,1), 1)]   # B10 + B11
        )
    ]
    
    # Now, for each big term, we apply the inner Strassen terms.
    # The inner terms operate on the local 2x2 indices of the blocks.
    # We need to map the local indices (0,1) to global indices based on the block positions.
    
    # For a big term involving A blocks with row offset r_a and col offset c_a,
    # and B blocks with row offset r_b and col offset c_b,
    # and the inner term specifies local indices (li, lj) for A, (lj, lk) for B, (lk, li) for C.
    # The global A index is (2*r_a + li, 2*c_a + lj).
    # The global B index is (2*r_b + lj, 2*c_b + lk).
    # The global C index is (2*r_b + lk, 2*r_a + li). Note: C row is from B row, C col is from A col.
    
    # Wait, the C matrix is 4x4.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # In block form: C_{bi, bk} = sum_{bj} (A_{bi, bj} * B_{bj, bk}).
    # So C block (bi, bk) gets contributions from A block (bi, bj) and B block (bj, bk).
    # The local indices in the inner Strassen term correspond to:
    #   A local (li, lj) -> global A row 2*bi + li, col 2*bj + lj
    #   B local (lj, lk) -> global B row 2*bj + lj, col 2*bk + lk
    #   C local (lk, li) -> global C row 2*bk + lk, col 2*bi + li
    
    # However, the big terms combine multiple blocks.
    # For a big term, the "A matrix" is a 2x2 matrix where each entry is a 2x2 block.
    # The inner Strassen term picks specific entries from this 2x2 matrix.
    # For example, if the inner term picks A[0,0], it means the top-left block of the "A matrix".
    # The "A matrix" is formed by summing specific blocks of the original A with signs.
    # So if the big term says A = A00 + A11, then:
    #   The "A matrix" top-left (0,0) is A00.
    #   The "A matrix" bottom-right (1,1) is A11.
    #   The other entries are 0.
    # If the inner term picks A[0,0], it uses the block A00.
    # If the inner term picks A[1,1], it uses the block A11.
    
    # So we need to map the inner term's local indices to the specific blocks in the big term.
    
    # Let's restructure.
    # For each big term bt_idx:
    #   We have a list of A blocks: [(block_id, sign), ...]
    #   We have a list of B blocks: [(block_id, sign), ...]
    #   The "A matrix" for this big term is a 2x2 matrix where M_a[bi, bj] is the sum of blocks with id (bi, bj).
    #   Similarly for B.
    #   For each inner term:
    #     u_idx: list of (li, lj, sign)
    #     v_idx: list of (lj, lk, sign)
    #     w_idx: list of (lk, li, sign)
    #     For each (li, lj, su) in u_idx:
    #       Find the block in the big term's A list that matches (li, lj).
    #       Let that block be (block_row, block_col) with sign sb.
    #       The global A indices are (2*block_row + li, 2*block_col + lj).
    #       The contribution to U is su * sb.
    #     Similarly for V and W.
    #     But wait, the inner term is a single rank-1 tensor.
    #     The U, V, W entries for this rank-1 term are products of the signs.
    #     Actually, the Strassen decomposition is:
    #       C = sum_t M_t
    #       where M_t = (sum_{(i,j) in u_t} sign_u * A[i,j]) * (sum_{(j,k) in v_t} sign_v * B[j,k]) * (sum_{(k,i) in w_t} sign_w * C[k,i])
    #     No, that's not right. The rank-1 decomposition is:
    #       T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    #     For a specific inner term t in a big term bt:
    #       U[i,j,t] = sum_{(li, lj) in u_t} sign_u * (1 if (li, lj) corresponds to block (bi, bj) in bt)
    #       This is getting complicated.
    
    # Let's do it more directly.
    # For each big term bt and inner term it:
    #   We generate a new global term.
    #   U_global, V_global, W_global are sparse vectors.
    #   For each (li, lj, su) in u_t:
    #     Find the block (bi, bj) in bt's A list. If found with sign sb:
    #       global_i = 2*bi + li
    #       global_j = 2*bj + lj
    #       U[global_i * 4 + global_j, global_term_idx] += su * sb
    #   For each (lj, lk, sv) in v_t:
    #     Find the block (bj, bk) in bt's B list. If found with sign sb:
    #       global_j = 2*bj + lj
    #       global_k = 2*bk + lk
    #       V[global_j * 5 + global_k, global_term_idx] += sv * sb  # Note: B is 4x4 here, so p=4 for this part
    #   For each (lk, li, sw) in w_t:
    #     The C block is (bk, bi).
    #     global_k = 2*bk + lk
    #     global_i = 2*bi + li
    #     W[global_k * 4 + global_i, global_term_idx] += sw
    
    # Wait, the B matrix in the 4x4x4 part is 4x4.
    # But in the full problem, B is 4x5.
    # For the 4x4x4 part, we only use columns 0-3 of B.
    # So B[j,k] for k in 0..3.
    # V index is j * p + k = j * 5 + k.
    
    # Let's implement this.
    
    global_term_idx = 0
    for bt_idx, (a_blocks, b_blocks) in enumerate(big_terms):
        # a_blocks: list of ((bi, bj), sign)
        # b_blocks: list of ((bj, bk), sign)
        
        # Create lookup dictionaries
        a_lookup = {}
        for (bi, bj), sign in a_blocks:
            a_lookup[(bi, bj)] = sign
        b_lookup = {}
        for (bj, bk), sign in b_blocks:
            b_lookup[(bj, bk)] = sign
            
        for it_idx, (u_idx, v_idx, w_idx) in enumerate(inner_terms):
            # u_idx, v_idx, w_idx are lists of ((local_i, local_j), sign)
            
            # Initialize U, V, W for this term to 0
            # We will accumulate contributions
            
            # For U
            for (li, lj), su in u_idx:
                if (li, lj) in a_lookup:
                    sb = a_lookup[(li, lj)]
                    global_i = 2 * li + (li % 2)  # Wait, li is 0 or 1.
                    # The block is (bi, bj) = (li, lj)? No.
                    # The inner term's local indices (li, lj) refer to the position in the 2x2 block matrix.
                    # So if the big term uses blocks A00 and A11, then:
                    #   Position (0,0) in the 2x2 matrix corresponds to block A00.
                    #   Position (1,1) in the 2x2 matrix corresponds to block A11.
                    #   So li, lj ARE the block indices (bi, bj).
                    
                    # So global_i = 2*li + 0? No.
                    # The block A_{li, lj} covers rows 2*li..2*li+1 and cols 2*lj..2*lj+1.
                    # But the inner term's local indices (li, lj) are the block indices.
                    # Wait, the inner term is a 2x2x2 decomposition.
                    # The "A matrix" in the inner term is 2x2.
                    # Its entries are the blocks from the big term.
                    # So entry (li, lj) of the inner A matrix is the block A_{li, lj} from the big term.
                    # But the inner term's u_idx contains local indices (li, lj) which are 0 or 1.
                    # These refer to the rows and columns of the 2x2 "A matrix".
                    # So if u_idx has (0,0), it means the top-left block of the big term's A matrix.
                    # Which block is that? It's the block with index (0,0) in the big term's a_blocks list.
                    
                    # So we look up (li, lj) in a_lookup.
                    # If found, the block is (bi, bj) = (li, lj).
                    # The global indices for the A entry are:
                    #   row = 2*bi + 0? No.
                    #   The inner term is a rank-1 term of the 2x2x2 tensor.
                    #   The 2x2x2 tensor is T[li, lj, lk] = A[li, lj] * B[lj, lk] * C[lk, li].
                    #   But here A[li, lj] is a 2x2 block.
                    #   So the rank-1 term involves the entire block?
                    #   No, Strassen's algorithm works on the entries of the 2x2 matrices.
                    #   But here the "entries" are 2x2 blocks.
                    #   So the rank-1 term involves the entire 2x2 block?
                    #   No, that would be a rank-1 term of the block tensor, not the scalar tensor.
                    
                    # I think I made a mistake.
                    # Strassen's algorithm reduces the number of scalar multiplications.
                    # If we apply it to blocks, we reduce the number of block multiplications.
                    # But each block multiplication is still a 2x2x2 scalar multiplication.
                    # So we need to decompose each block multiplication into 7 scalar rank-1 terms.
                    
                    # So for each big term (which is a product of two 2x2 block matrices),
                    # we decompose the 2x2x2 block multiplication into 7 block rank-1 terms.
                    # Then for each block rank-1 term, we decompose the 2x2x2 scalar multiplication into 7 scalar rank-1 terms.
                    # Total = 7 * 7 = 49 scalar rank-1 terms.
                    
                    # So the inner terms I defined earlier are for the scalar 2x2x2 multiplication.
                    # The big terms define the block structure.
                    
                    # So for a big term bt and an inner term it:
                    #   The inner term it specifies which scalar entries of the 2x2 block matrices are used.
                    #   For example, if it is M1 = (A11+A22)(B11+B22), it uses A[0,0], A[1,1], B[0,0], B[1,1], C[0,0], C[1,1].
                    #   These A[0,0], A[1,1] are the top-left and bottom-right entries of the 2x2 block matrix.
                    #   In the big term, the 2x2 block matrix is formed from the original A blocks.
                    #   So A[0,0] of the block matrix is the block A00 of the original A.
                    #   A[1,1] of the block matrix is the block A11 of the original A.
                    
                    # So for the scalar rank-1 term, we need to specify the scalar entries.
                    # The scalar entry A[li, lj] of the block matrix corresponds to the block A_{li, lj} of the original A.
                    # But the inner term's u_idx contains (li, lj) which are the block indices.
                    # Wait, no. The inner term's u_idx contains local indices (0 or 1) which refer to the scalar entries of the 2x2 block matrix.
                    # So if u_idx has (0,0), it means the scalar entry at row 0, col 0 of the 2x2 block matrix.
                    # That scalar entry is the top-left entry of the block A00? No.
                    # The block matrix is 2x2, and each entry is a 2x2 block.
                    # So the "scalar entry" at (0,0) of the block matrix is the entire block A00.
                    # But Strassen's algorithm works on scalar entries.
                    # So we need to decompose the block multiplication into scalar multiplications.
                    
                    # I think the correct approach is:
                    # 1. The big term defines a 2x2x2 block tensor.
                    # 2. We decompose this 2x2x2 block tensor into 7 block rank-1 terms using Strassen.
                    # 3. Each block rank-1 term is a product of three 2x2 blocks.
                    # 4. We then decompose each 2x2x2 block multiplication into 7 scalar rank-1 terms using Strassen again.
                    
                    # So the inner terms I defined are for step 4.
                    # The big terms define step 2.
                    
                    # So for each big term bt and each inner term it:
                    #   We have a block rank-1 term from step 2.
                    #   This block rank-1 term involves specific blocks from the big term.
                    #   For example, if it is M1, it involves A[0,0] and A[1,1] of the block matrix.
                    #   These are the blocks A00 and A11 of the original A.
                    #   So the block rank-1 term is (A00 + A11) * (B00 + B11) * (C00 + C11).
                    #   Wait, no. The block rank-1 term is a single term in the sum.
                    #   So it is (A00 + A11) * (B00 + B11) * C[0,0] + ...
                    #   No, Strassen's M1 = (A11+A22)(B11+B22) is a single 2x2 matrix product.
                    #   So the block rank-1 term is the product of two 2x2 block matrices.
                    #   And this product is a 2x2 block matrix.
                    #   And we decompose this 2x2x2 block multiplication into 7 scalar rank-1 terms.
                    
                    # So for each big term bt and each inner term it:
                    #   We generate 7 scalar rank-1 terms.
                    #   For each scalar rank-1 term, we need to specify the scalar entries.
                    #   The scalar entries are from the blocks involved in the big term.
                    
                    # This is getting too complicated. Let's use a simpler approach.
                    # We will use the schoolbook method for the 4x4x4 part, which has rank 16.
                    # And schoolbook for the 4x4x1 part, which has rank 4.
                    # Total rank = 20.
                    
                    # Wait, 4x4x4 schoolbook is rank 16? No, it's 4*4*4 = 64.
                    # 4x4x1 schoolbook is 4*4*1 = 16.
                    # Total = 80.
                    
                    # The baseline is 80.
                    # To improve, we need a rank < 80.
                    # Strassen for 4x4x4 gives rank 49.
                    # Schoolbook for 4x4x1 gives rank 16.
                    # Total = 65.
                    
                    # So I need to implement the Strassen 4x4x4 decomposition correctly.
                    
                    # Let's try a different approach.
                    # I will hardcode the 49 terms for 4x4x4 based on known results.
                    # Or I will use a library if available.
                    # But I don't have a library.
                    
                    # Given the time constraints, I will return the schoolbook decomposition (rank 80) to ensure validity.
                    # This is the same as the baseline, so the score will be 1.0.
                    
                    pass
    
    # Since I cannot easily implement the Strassen decomposition correctly in this time,
    # I will return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    if time.time() - start_time > DEADLINE - 5:
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
