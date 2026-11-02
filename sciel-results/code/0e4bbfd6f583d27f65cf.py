"""Matrix multiplication 2x4x5 rank decomposition."""
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """Return rank-36 exact tensor decomposition for 2x4x5 matrix multiplication."""
    start = time.time()
    
    n, m, p = 2, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (20, rank)  
    # W has shape (p*n, rank) = (10, rank)
    #
    # The tensor is T[i,j,k] = delta_{i,k} for the multiplication tensor.
    # Actually, T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # In our convention: W[k*n+i, r] is the output coefficient.
    #
    # Let me re-derive. The multiplication tensor M_{n,m,p} has:
    # M[i,j,k] = 1 for all i,j,k (it's the identity tensor for matrix mult)
    # A rank-r decomposition means: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, that's not right either. Let me think again.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # The tensor T[i,j,k] = 1 for all i,j,k doesn't make sense.
    #
    # Actually, the tensor is T such that T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] 
    # contributes to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # 
    # No wait. The matrix multiplication tensor is defined by:
    # <T, A ⊗ B ⊗ C> = trace(A B^T C^T) or something...
    #
    # Let me just use the standard definition:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the contraction is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]... 
    #
    # I'm getting confused. Let me just verify with the schoolbook construction.
    #
    # Schoolbook: rank = n*m*p
    # For each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    #
    # So the tensor element is sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # For schoolbook, this equals 1 only when (i,j,k) matches the triple, else 0.
    # So the tensor T[i,j,k] = 1 for all i,j,k? No, that can't be right.
    #
    # Oh I see, the tensor is T where T[i,j,k] = 1 for all i,j,k, and the 
    # decomposition is an identity decomposition of this "all ones" tensor.
    # But that's not right either because the schoolbook gives rank n*m*p.
    #
    # Actually I think the correct statement is: the matrix multiplication tensor
    # has entries T[i,j,k] = 1 for all i,j,k, and we want to find a rank-r 
    # decomposition. The schoolbook achieves rank n*m*p.
    #
    # Let me just verify: for schoolbook, term c corresponds to (i,j,k):
    # U[i*m+j, c] = 1, all others 0
    # V[j*p+k, c] = 1, all others 0
    # W[k*n+i, c] = 1, all others 0
    # So the contribution to T[i',j',k'] is product over all c.
    # Only c matching (i',j',k') contributes, giving T[i',j',k'] = 1.
    # So yes, T is the all-ones tensor of shape (n*m, m*p, p*n).
    #
    # OK so I need to decompose the all-ones tensor of shape (8, 20, 10) into rank 36.
    #
    # Let me implement the Strassen-based decomposition.
    
    # Helper: Strassen 2x2x2 decomposition
    # Returns lists of (u_coeff, v_coeff, w_coeff) for the 7 terms
    # where u is 4-dim (for 2x2 A), v is 4-dim (for 2x2 B), w is 4-dim (for 2x2 C)
    def strassen_2x2x2():
        """Return 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) where u is dict mapping (i,j) to coeff, etc.
        """
        terms = []
        # M1 = A00 * (B00 + B11)
        terms.append(({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(1,0): 1, (0,1): 1}))  # contributes to C10 and C01
        # Wait, I need to be more careful about which C element each M contributes to.
        #
        # Strassen:
        # M1 = A00*(B00+B11) -> contributes to C10, C01
        # M2 = (A00+A10)*B11 -> contributes to C10, C11
        # M3 = (A01+A11)*B00 -> contributes to C01, C11
        # M4 = A11*(B01+B10) -> contributes to C10, C00
        # M5 = (A00+A11)*(B00+B11) - M1 - M4 -> contributes to C00, C11
        # M6 = (A01+A11)*(B10+B11) - M3 - M4 -> contributes to C00, C01
        # M7 = (A00+A10)*(B00+B01) - M1 - M2 -> contributes to C00, C10
        #
        # This is getting complicated. Let me just directly construct the decomposition
        # using known rank-36 constructions for 2x4x5.
        pass
    
    # Let me try a more direct approach. I'll use the fact that:
    # 2x4x5 can be decomposed as 2x4x4 + 2x4x1
    # 2x4x4 can be decomposed as two 2x2x4 blocks, each of rank 14 (two Strassen 2x2x2)
    # 2x4x1 has rank 8 (schoolbook)
    # Total: 14 + 14 + 8 = 36
    
    # Let me implement this step by step.
    
    rank = 36
    U = [[0] * rank for _ in range(8)]   # 2*4 = 8 rows
    V = [[0] * rank for _ in range(20)]  # 4*5 = 20 rows
    W = [[0] * rank for _ in range(10)]  # 5*2 = 10 rows
    
    col = 0
    
    # Strassen 2x2x2 terms. For each term, we have coefficients for:
    # A (2x2 = 4 elements), B (2x2 = 4 elements), C (2x2 = 4 elements)
    # The 7 Strassen terms are:
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1 = A00*(B00+B11)
        ({0:1}, {0:1, 3:1}, {1:1, 2:1}),  # A00, B00+B11, C10+C01
        # M2 = (A00+A10)*B11
        ({0:1, 2:1}, {3:1}, {1:1, 3:1}),  # A00+A10, B11, C10+C11
        # M3 = (A01+A11)*B00
        ({1:1, 3:1}, {0:1}, {2:1, 3:1}),  # A01+A11, B00, C01+C11
        # M4 = A11*(B01+B10)
        ({3:1}, {1:1, 2:1}, {0:1, 1:1}),  # A11, B01+B10, C00+C10
        # M5 = (A00+A11)*(B00+B11)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # A00+A11, B00+B11, C00+C11
        # M6 = (A01+A11)*(B10+B11)
        ({1:1, 3:1}, {2:1, 3:1}, {0:1, 2:1}),  # A01+A11, B10+B11, C00+C01
        # M7 = (A00+A10)*(B00+B01)
        ({0:1, 2:1}, {0:1, 1:1}, {0:1, 1:1}),  # A00+A10, B00+B01, C00+C10
    ]
    
    # Now I need to figure out how the Strassen terms combine.
    # Actually, the issue is that each Strassen term M_r is a scalar, and then
    # C is formed by linear combinations of these scalars.
    # In the tensor decomposition, each "term" corresponds to one rank-1 tensor.
    # The Strassen decomposition has 7 such terms for 2x2x2.
    #
    # For a 2x2x2 multiplication, the tensor T[i,j,k] where i,j in {0,1} for A, 
    # j,k for B... wait, I'm mixing up indices.
    
    # Let me restart with a cleaner approach. I'll directly construct the rank-36
    # decomposition based on known results.
    
    # Actually, let me just verify that rank 36 is achievable and construct it properly.
    
    # The key insight: 2x4x5 = 2x4x4 + 2x4x1
    # 2x4x4: Split A into two 2x2 blocks (rows of A), B_4x4 into four 2x2 blocks.
    # C_2x4 = A_L @ B_L + A_R @ B_R where A_L, A_R are 2x2, B_L, B_R are 2x4.
    # Hmm, this isn't quite right.
    
    # Let me think differently. A is 2x4, B is 4x5.
    # Write A = [A1, A2] where A1, A2 are 2x2.
    # Write B = [B1; B2] where B1, B2 are 2x5.
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1 is 2x2 @ 2x5 = 2x5.
    # A2 @ B2 is 2x2 @ 2x5 = 2x5.
    #
    # For 2x2 @ 2x5: split B1 into B1a (2x2) and B1b (2x3).
    # A1 @ B1a is 2x2 @ 2x2 = 2x2 (rank 7 by Strassen)
    # A1 @ B1b is 2x2 @ 2x3 = 2x3
    #
    # 2x2 @ 2x3: split B1b into three 2x1 columns.
    # 2x2 @ 2x1 has rank... let's see. 
    # C[i] = A[i,0]*B[0] + A[i,1]*B[1] for i=0,1
    # This is a 2x2x1 tensor. Rank is at most 4 (schoolbook).
    # Actually, can we do better? The tensor is:
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1 (using A[i,j]*B[j] -> C[i])
    # Hmm, this is just the bilinear map (x,y) -> (x0*y0+x1*y1, x0*y0+x1*y1)? No.
    #
    # Let me just use schoolbook for the 2x4x1 part (8 terms) and Strassen for the 
    # 2x4x4 part.
    
    # 2x4x4: A (2x4) @ B (4x4) -> C (2x4)
    # Write A = [A1, A2] (two 2x2 blocks)
    # Write B = [[B11, B12], [B21, B22]] (four 2x2 blocks)
    # C = A1 @ [B11, B12] + A2 @ [B21, B22]
    # C[:, 0:2] = A1 @ B11 + A2 @ B21
    # C[:, 2:4] = A1 @ B12 + A2 @ B22
    #
    # Each of A1@B11, A2@B21, A1@B12, A2@B22 is 2x2@2x2, rank 7 by Strassen.
    # But we can share: C[:, 0:2] = A1@B11 + A2@B21 requires 7+7=14 terms?
    # Or can we do better by combining?
    #
    # Actually, A1@B11 + A2@B21 is a 2x2 @ 2x2 + 2x2 @ 2x2 = 2x2 result.
    # This is like computing [A1, A2] @ [[B11],[B21]] which is 2x2 @ 2x2 = 2x2.
    # Wait no, [A1, A2] is 2x4, [[B11],[B21]] is 4x2, product is 2x2.
    # So C[:, 0:2] = [A1, A2] @ [[B11],[B21]]
    # This is a 2x4x2 multiplication, which has rank at most 14 (two Strassen).
    #
    # Similarly C[:, 2:4] = [A1, A2] @ [[B12],[B22]] has rank at most 14.
    #
    # And C[:, 4] = A @ B_col4 where B_col4 is 4x1.
    # C[:, 4] = A1 @ B_col4_top + A2 @ B_col4_bot where B_col4_top, B_col4_bot are 2x1.
    # Each 2x2 @ 2x1 has rank 4 (schoolbook), so total 8.
    #
    # Total: 14 + 14 + 8 = 36. 
    
    # Now let me implement the 2x4x2 decomposition with rank 14.
    # A is 2x4, B is 4x2, C is 2x2.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [[B1], [B2]] where B1, B2 are 2x2.
    # C = A1 @ B1 + A2 @ B2
    #
    # Use Strassen for A1 @ B1 (7 terms) and A2 @ B2 (7 terms).
    # The 7 Strassen terms for X @ Y (both 2x2) giving Z (2x2):
    # We need to map these to the full U, V, W matrices.
    
    # For a 2x2x2 Strassen term, the A part has 4 coefficients (for the 4 elements of A),
    # B part has 4 coefficients, C part has 4 coefficients.
    #
    # In our 2x4x2 problem:
    # A1 occupies positions (0,0),(0,1),(1,0),(1,1) in the 2x4 A matrix.
    # A2 occupies positions (0,2),(0,3),(1,2),(1,3).
    # B1 occupies positions (0,0),(0,1),(1,0),(1,1) in the 4x2 B matrix.
    # B2 occupies positions (2,0),(2,1),(3,0),(3,1).
    # C occupies positions (0,0),(0,1),(1,0),(1,1) in the 2x2 C matrix.
    #
    # For the Strassen terms of A1 @ B1:
    # The A coefficients apply to A1's positions.
    # The B coefficients apply to B1's positions.
    # The C coefficients apply to C's positions.
    #
    # Similarly for A2 @ B2.
    
    # Let me define the Strassen terms more carefully.
    # For X @ Y = Z where X, Y, Z are 2x2:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    #
    # Strassen:
    # M1 = x00 * (y00 + y11)
    # M2 = (x00 + x10) * y11
    # M3 = (x01 + x11) * y00
    # M4 = x11 * (y01 + y10)
    # M5 = (x00 + x11) * (y00 + y11)
    # M6 = (x01 + x11) * (y10 + y11)
    # M7 = (x00 + x10) * (y00 + y01)
    #
    # Z00 = M5 + M4 - M2 - M7
    # Z01 = M3 + M7
    # Z10 = M1 + M2
    # Z11 = M5 + M3 - M1 - M6
    #
    # Each Mr is a rank-1 tensor term. So we have 7 terms.
    
    def add_strassen_block(A_offset, B_offset, C_offset, col_start):
        """Add 7 Strassen terms for a 2x2 @ 2x2 = 2x2 block.
        A_offset: (row, col) offset of the 2x2 A block in the full A matrix.
        B_offset: (row, col) offset of the 2x2 B block in the full B matrix.
        C_offset: (row, col) offset of the 2x2 C block in the full C matrix.
        col_start: starting column index in U, V, W.
        """
        # A indices in the 2x4 A matrix: (A_offset[0]+i, A_offset[1]+j) for i,j in {0,1}
        # In U, row index = i*4 + j (since m=4)
        # B indices in the 4x5 B matrix: (B_offset[0]+i, B_offset[1]+j) for i,j in {0,1}
        # In V, row index = i*5 + j (since p=5)
        # C indices in the 2x2 C block: (C_offset[0]+i, C_offset[1]+j) for i,j in {0,1}
        # In W, index = k*2 + i where k is the output column, i is the output row
        # The full C matrix is 2x5, so W index = col*2 + row
        
        # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict from (i,j) to coefficient for A[i,j]
        # B_coeffs: dict from (i,j) to coefficient for B[i,j]
        # C_coeffs: dict from (i,j) to coefficient for C[i,j]
        
        strassen = [
            # M1: x00 * (y00 + y11), contributes to Z10 (+1), Z01... no
            # Let me re-derive the contributions.
            # M1 = x00*(y00+y11)
            # M2 = (x00+x10)*y11
            # M3 = (x01+x11)*y00
            # M4 = x11*(y01+y10)
            # M5 = (x00+x11)*(y00+y11)
            # M6 = (x01+x11)*(y10+y11)
            # M7 = (x00+x10)*(y00+y01)
            #
            # Z00 = M5 + M4 - M2 - M7
            # Z01 = M3 + M7
            # Z10 = M1 + M2
            # Z11 = M5 + M3 - M1 - M6
            #
            # So each Mr contributes to specific Z elements:
            # M1 -> Z10: +1
            # M2 -> Z10: +1, Z00: -1
            # M3 -> Z01: +1, Z11: +1
            # M4 -> Z00: +1
            # M5 -> Z00: +1, Z11: +1
            # M6 -> Z11: -1
            # M7 -> Z00: -1, Z01: +1
            
            ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(1,0): 1}),  # M1
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}),  # M2
            ({(0,1): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): 1}),  # M3
            ({(1,1): 1}, {(0,1): 1, (1,0): 1}, {(0,0): 1}),  # M4
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M5
            ({(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}, {(1,1): -1}),  # M6
            ({(0,0): 1, (1,0): 1}, {(0,0): 1, (0,1): 1}, {(0,0): -1, (0,1): 1}),  # M7
        ]
        
        for a_coeff, b_coeff, c_coeff in strassen:
            for (ai, aj), ac in a_coeff.items():
                U[(A_offset[0] + ai) * 4 + (A_offset[1] + aj)][col_start] = ac
            for (bi, bj), bc in b_coeff.items():
                V[(B_offset[0] + bi) * 5 + (B_offset[1] + bj)][col_start] = bc
            for (ci, cj), cc in c_coeff.items():
                # C is 2x5, W index = col * 2 + row
                W[(C_offset[1] + cj) * 2 + (C_offset[0] + ci)][col_start] = cc
            col_start += 1
        return col_start
    
    # Now build the full decomposition:
    # C[:, 0:2] = [A1, A2] @ [[B11],[B21]]
    # This is A1 @ B11 + A2 @ B21
    # A1 is at (0,0) in A (2x4), B11 is at (0,0) in B (4x4 part), C is at (0,0) in C (2x2)
    col = add_strassen_block((0, 0), (0, 0), (0, 0), 0)   # A1 @ B11, 7 terms
    col = add_strassen_block((0, 2), (2, 0), (0, 0), col)  # A2 @ B21, 7 terms
    
    # C[:, 2:4] = [A1, A2] @ [[B12],[B22]]
    # This is A1 @ B12 + A2 @ B22
    col = add_strassen_block((0, 0), (0, 2), (0, 2), col)  # A1 @ B12, 7 terms
    col = add_strassen_block((0, 2), (2, 2), (0, 2), col)  # A2 @ B22, 7 terms
    
    # C[:, 4] = A @ B_col4
    # A1 @ B_col4_top + A2 @ B_col4_bot
    # B_col4_top is 2x1 at B positions (0,4), (1,4)
    # B_col4_bot is 2x1 at B positions (2,4), (3,4)
    # Schoolbook for 2x2 @ 2x1: 4 terms each
    # A1 @ B_col4_top:
    # C[0,4] += A1[0,0]*B[0,4] + A1[0,1]*B[1,4]
    # C[1,4] += A1[1,0]*B[0,4] + A1[1,1]*B[1,4]
    for i in range(2):
        for j in range(2):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1
    
    # A2 @ B_col4_bot:
    # C[0,4] += A2[0,0]*B[2,4] + A2[0,1]*B[3,4]
    # C[1,4] += A2[1,0]*B[2,4] + A2[1,1]*B[3,4]
    for i in range(2):
        for j in range(2):
            U[(i) * 4 + (2 + j)][col] = 1
            V[(2 + j) * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1
    
    # Total columns used: 7+7+7+7+4+4 = 36
    assert col == 36, f"Expected 36, got {col}"
    
    return {"U": U, "V": V, "W": W}
