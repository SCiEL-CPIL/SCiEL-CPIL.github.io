<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the 4x4x4 Strassen decomposition
    # Strassen's 7-term algorithm for 2x2x2, recursively applied
    # For 4x4x4, we can use the 49-term decomposition from Strassen's method
    # Actually, let me use a known 4x4x4 decomposition with rank 49
    
    # The standard Strassen-based 4x4x4 decomposition uses rank 49
    # Let me construct it using the tensor decomposition approach
    
    # For now, use the schoolbook 5x5x5 = 125 as fallback
    # But let's try to construct a 78-rank decomposition
    
    # Approach: Use 4x4 Strassen (rank 49) + boundary terms
    # But the boundary terms for 5x5 are complex
    
    # Let me just return the schoolbook for now and try to improve
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists for return
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 as (U_entries, V_entries, W_entries, C_coeffs).
    Each term is a rank-1 decomposition of part of the tensor.
    U_entries: list of (row_idx, coeff) for A
    V_entries: list of (row_idx, coeff) for B  
    W_entries: list of (row_idx, coeff) for C
    C_coeffs: dict mapping (i,k) -> list of (term_idx, coeff) for C[i,k] = sum coeff * term
    """
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A10)*B11
    # P6 = (A01-A00)*(B00+B10)
    # P7 = (A10-A11)*(B01+B11)
    
    # For each product, we need to express it as sum of rank-1 terms
    # where each rank-1 term has U[row], V[col], W[diag] entries
    
    # P1 = (A00+A11)*(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is not a single rank-1 term in the tensor sense.
    # In the tensor decomposition, each "term" corresponds to one product in the algorithm.
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For Strassen, each product P_r involves linear combinations of A and B entries.
    # P1 involves A00, A11 (rows 0,1 of A) and B00, B11 (cols 0,1 of B in the 2x2 block)
    
    # Let me think of it differently. For 2x2x2:
    # A has indices (i,j) with i,j in {0,1}
    # B has indices (j,k) with j,k in {0,1}
    # C has indices (i,k) with i,k in {0,1}
    
    # The tensor T[i,j,k] = 1 if C[i,k] uses A[i,j] and B[j,k]
    # Actually T[i,j,k] = 1 for all valid i,j,k (it's the full multiplication tensor)
    
    # For Strassen decomposition of the 2x2x2 tensor:
    # We express T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    
    # Where u_r, v_r, w_r are vectors in R^4 (for 2x2 = 4 entries each)
    
    # P1 = (A00+A11)*(B00+B11)
    # u_1: A00=1, A11=1, A01=0, A10=0 -> [1,0,0,1] (indexing: 00,01,10,11)
    # v_1: B00=1, B11=1, B01=0, B10=0 -> [1,0,0,1] (indexing: 00,01,10,11)
    # w_1: C00 += P1, C11 += P1 -> C00=1, C11=1 -> [1,0,0,1] (indexing: 00,01,10,11)
    
    # P2 = (A10+A11)*B00
    # u_2: A10=1, A11=1 -> [0,0,1,1]
    # v_2: B00=1 -> [1,0,0,0]
    # w_2: C10 += P2, C11 -= P2 -> C10=1, C11=-1 -> [0,0,1,-1]
    
    # P3 = A00*(B10-B11)
    # u_3: A00=1 -> [1,0,0,0]
    # v_3: B10=1, B11=-1 -> [0,0,1,-1]
    # w_3: C01 += P3, C11 += P3 -> C01=1, C11=1 -> [0,1,0,1]
    
    # P4 = A11*(B01-B00)
    # u_4: A11=1 -> [0,0,0,1]
    # v_4: B01=1, B00=-1 -> [-1,1,0,0]
    # w_4: C00 += P4, C10 += P4 -> C00=1, C10=1 -> [1,0,1,0]
    
    # P5 = (A00+A10)*B11
    # u_5: A00=1, A10=1 -> [1,0,1,0]
    # v_5: B11=1 -> [0,0,0,1]
    # w_5: C00 -= P5, C01 += P5 -> C00=-1, C01=1 -> [-1,1,0,0]
    
    # P6 = (A01-A00)*(B00+B10)
    # u_6: A01=1, A00=-1 -> [-1,1,0,0]
    # v_6: B00=1, B10=1 -> [1,0,1,0]
    # w_6: C11 += P6 -> C11=1 -> [0,0,0,1]
    
    # P7 = (A10-A11)*(B01+B11)
    # u_7: A10=1, A11=-1 -> [0,0,1,-1]
    # v_7: B01=1, B11=1 -> [0,1,0,1]
    # w_7: C00 += P7 -> C00=1 -> [1,0,0,0]
    
    U_terms = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 0, 1, 0],  # P5
        [-1, 1, 0, 0], # P6
        [0, 0, 1, -1], # P7
    ]
    V_terms = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 0, 1, -1], # P3
        [-1, 1, 0, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 0, 1, 0],  # P6
        [0, 1, 0, 1],  # P7
    ]
    W_terms = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1],  # P3
        [1, 0, 1, 0],  # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 0, 0, 0],  # P7
    ]
    
    return U_terms, V_terms, W_terms


def _build_strassen_4x4():
    """Build 4x4x4 Strassen decomposition using two levels of Strassen.
    Returns U, V, W matrices of shape (16, 49).
    Indexing: A[i,j] -> row i*2+j, B[j,k] -> row j*2+k, C[k,i] -> row k*2+i
    """
    U_2, V_2, W_2 = _strassen_2x2_terms()
    
    # For 4x4, we apply Strassen to 2x2 blocks where each "entry" is a 2x2 matrix.
    # A_4x4 = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B and C.
    
    # The 7 products at the top level:
    # P1 = (A00+A11)*(B00+B11) - both are 2x2 matrix multiplications
    # Each of these 2x2 multiplications uses 7 Strassen terms
    # So P1 contributes 7*7 = 49 terms? No, that's not right.
    
    # Actually, for 4x4x4 using Strassen:
    # We partition into 2x2 blocks of size 2x2.
    # The top-level Strassen gives 7 products, each of which is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication uses 7 terms.
    # Total: 7*7 = 49 terms.
    
    # But we can do better by sharing terms. Let me think...
    
    # Actually, the standard way: 
    # Level 1: 7 products, each is a 2x2 matrix multiply
    # Level 2: Each 2x2 multiply uses 7 scalar products
    # Total: 49 scalar products
    
    # For the tensor decomposition, each scalar product corresponds to a rank-1 term.
    
    U_4 = []
    V_4 = []
    W_4 = []
    
    # For each top-level product r (0..6), and each bottom-level product s (0..6):
    for r in range(7):
        for s in range(7):
            # Top-level U contribution: U_2[r] applied to the 2x2 block structure
            # U_2[r] = [u00, u01, u10, u11] for the 4 entries of the 2x2 "super-entry"
            # Each super-entry is a 2x2 matrix, so we need to expand.
            
            # The 4x4 A matrix has entries A[i,j] for i,j in {0,1,2,3}
            # Block A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], etc.
            
            # U_2[r] gives coefficients for blocks: [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
            # Each block is 2x2, so we need to apply the bottom-level Strassen.
            
            # Wait, I think I'm overcomplicating this. Let me use a different approach.
            
            # For the 4x4x4 tensor, I'll use the fact that it's a Kronecker product structure.
            # Actually, let me just build it directly using the two-level Strassen.
            
            pass
    
    # Let me use a cleaner approach: build the 4x4x4 decomposition directly.
    # I'll use the known 49-term decomposition.
    
    # Actually, let me construct it step by step.
    # The 4x4x4 multiplication tensor can be decomposed using Strassen at two levels.
    
    # Top level: 7 products P1..P7, each involving 2x2 block matrices
    # Bottom level: Each 2x2 block multiplication uses 7 terms
    
    # For the tensor decomposition, each term is a rank-1 component.
    
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    col = 0
    for r in range(7):
        for s in range(7):
            # Top-level term r: U_2[r] = [a00, a01, a10, a11]
            # Bottom-level term s: U_2[s] = [b00, b01, b10, b11]
            
            # The combined U vector for this term:
            # For each block position (bi, bj) in {0,1}^2:
            #   coefficient from top level: U_2[r][bi*2+bj]
            #   For each entry (ei, ej) in the 2x2 block:
            #     coefficient from bottom level: U_2[s][ei*2+ej]
            #     A index: (bi*2+ei, bj*2+ej)
            
            # Similarly for V and W.
            
            a_top = U_2[r]  # [a00, a01, a10, a11]
            b_top = V_2[r]
            c_top = W_2[r]
            
            a_bot = U_2[s]
            b_bot = V_2[s]
            c_bot = W_2[s]
            
            for bi in range(2):
                for bj in range(2):
                    a_coeff_top = a_top[bi * 2 + bj]
                    b_coeff_top = b_top[bi * 2 + bj]
                    c_coeff_top = c_top[bi * 2 + bj]
                    
                    for ei in range(2):
                        for ej in range(2):
                            a_coeff_bot = a_bot[ei * 2 + ej]
                            b_coeff_bot = b_bot[ei * 2 + ej]
                            c_coeff_bot = c_bot[ei * 2 + ej]
                            
                            a_val = a_coeff_top * a_coeff_bot
                            b_val = b_coeff_top * b_coeff_bot
                            c_val = c_coeff_top * c_coeff_bot
                            
                            # A index: row = bi*2+ei, col = bj*2+ej
                            a_row = (bi * 2 + ei) * 2 + (bj * 2 + ej)
                            # B index: row = bj*2+ej, col = bi*2+ei (since B[j,k] and j is from A's col, k is to C's col)
                            # Wait, I need to be careful with indices.
                            
                            # A[i,j], B[j,k], C[i,k]
                            # For the block structure:
                            # A block (bi,bj) has entries A[bi*2+ei, bj*2+ej]
                            # B block (bj,bi) has entries B[bj*2+ej, bi*2+ei]  -- NO
                            
                            # Let me reconsider. In matrix multiplication C = A*B:
                            # C[i,k] = sum_j A[i,j] * B[j,k]
                            # 
                            # For 4x4 with 2x2 blocks:
                            # C block (ci, ck) = sum_{bj} A block (ci, bj) * B block (bj, ck)
                            #
                            # For Strassen at the top level, we have 7 products.
                            # Each product involves specific blocks of A and B.
                            #
                            # The tensor T[i,j,k] = 1 means C[i,k] gets contribution from A[i,j]*B[j,k].
                            #
                            # For the decomposition, each rank-1 term has:
                            # U[i,j], V[j,k], W[i,k] (using W[k*n+i] convention, so W[k,i])
                            #
                            # Let me redefine: W is indexed by (k,i) where k is column of C, i is row of C.
                            # So W[k*n + i, r] corresponds to C[i,k].
                            
                            # For the 4x4 case:
                            # A[i,j] -> row i*4+j in U
                            # B[j,k] -> row j*4+k in V
                            # C[i,k] -> W[k*4+i, r] (since W[k*n+i] with n=4)
                            
                            # For the term (r,s):
                            # Top level: r determines which blocks are involved
                            # Bottom level: s determines which entries within blocks
                            
                            # A[i,j] where i = bi*2+ei, j = bj*2+ej
                            # But the top-level coefficient depends on which block pair is used.
                            
                            # I think the issue is that the top-level Strassen doesn't just pick blocks;
                            # it forms linear combinations. So for product P1 = (A00+A11)*(B00+B11):
                            # A contribution: A00 + A11, meaning all entries of blocks 00 and 11
                            # B contribution: B00 + B11, meaning all entries of blocks 00 and 11
                            
                            # So for the U vector of this term:
                            # For entry A[i,j]:
                            #   if (i,j) is in block 00 or block 11: coefficient = 1
                            #   otherwise: 0
                            # Then multiplied by the bottom-level coefficient.
                            
                            # This is getting complex. Let me just compute it directly.
                            
                            pass
            col += 1
    
    # This approach is getting too complex. Let me use a different strategy.
    # I'll construct the 4x4x4 Strassen decomposition by directly computing the rank-1 terms.
    
    # Actually, let me just use the schoolbook for 4x4 (rank 64) and add boundary terms.
    # That gives 64 + 45 = 109, which is worse than 125. Not helpful.
    
    # Let me try a different approach: use the known 49-term 4x4x4 decomposition.
    
    # I'll look up or derive the 49 terms.
    
    # For now, let me just return the schoolbook 5x5x5 = 125.
    # The improvement will come from a better decomposition.
    
    return None


def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try to build a Strassen-based decomposition
    # For now, use schoolbook as fallback
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists for return
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I realize my approach is getting too complex. Let me simplify and actually implement the Strassen-based 4×4×4 decomposition properly, then add the boundary terms for 5×5×5.

