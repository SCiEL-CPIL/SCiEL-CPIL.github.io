"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # We'll build the decomposition using a block structure:
    # i: 4, j: 4+1, k: 4+1
    # Blocks: (4,4,4), (4,4,1), (4,1,4), (4,1,1)
    
    U_rows = n * m  # 20
    V_rows = m * p  # 25
    W_rows = p * n  # 20
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(U_coeff, V_coeff, W_coeff):
        """Add a rank-1 term. U_coeff: dict (i*m+j -> val), V_coeff: dict (j*p+k -> val), W_coeff: dict (k*n+i -> val)"""
        U_terms.append(U_coeff)
        V_terms.append(V_coeff)
        W_terms.append(W_coeff)
    
    def strassen_2x2x2(oi, oj, ok):
        """Add Strassen decomposition for 2x2x2 block at offset (oi, oj, ok)."""
        # M1 = (a00+a11)(b00+b11)
        U1 = {oi*m + oj: 1, (oi+1)*m + (oj+1): 1}
        V1 = {oj*p + ok: 1, (oj+1)*p + (ok+1): 1}
        W1 = {ok*n + oi: 1, (ok+1)*n + (oi+1): 1}
        add_term(U1, V1, W1)
        
        # M2 = (a10+a11)b00
        U2 = {(oi+1)*m + oj: 1, (oi+1)*m + (oj+1): 1}
        V2 = {oj*p + ok: 1}
        W2 = {(ok+1)*n + (oi+1): 1}
        add_term(U2, V2, W2)
        
        # M3 = a00(b01-b11)
        U3 = {oi*m + oj: 1}
        V3 = {oj*p + (ok+1): 1, (oj+1)*p + (ok+1): -1}
        W3 = {(ok+1)*n + oi: 1}
        add_term(U3, V3, W3)
        
        # M4 = a11(b10-b00)
        U4 = {(oi+1)*m + (oj+1): 1}
        V4 = {(oj+1)*p + ok: 1, oj*p + ok: -1}
        W4 = {ok*n + (oi+1): 1}
        add_term(U4, V4, W4)
        
        # M5 = (a00+a01)b11
        U5 = {oi*m + oj: 1, oi*m + (oj+1): 1}
        V5 = {(oj+1)*p + (ok+1): 1}
        W5 = {ok*n + oi: -1, (ok+1)*n + oi: 1}
        add_term(U5, V5, W5)
        
        # M6 = (a10-a00)(b00+b01)
        U6 = {(oi+1)*m + oj: 1, oi*m + oj: -1}
        V6 = {oj*p + ok: 1, oj*p + (ok+1): 1}
        W6 = {(ok+1)*n + (oi+1): 1}
        add_term(U6, V6, W6)
        
        # M7 = (a01-a11)(b10+b11)
        U7 = {oi*m + (oj+1): 1, (oi+1)*m + (oj+1): -1}
        V7 = {(oj+1)*p + ok: 1, (oj+1)*p + (ok+1): 1}
        W7 = {ok*n + oi: 1}
        add_term(U7, V7, W7)
    
    def strassen_4x4x4(oi, oj, ok):
        """Add Strassen decomposition for 4x4x4 block at offset (oi, oj, ok).
        Uses 7 Strassen terms at the 2x2 block level, each decomposed with 2x2x2 Strassen.
        Total rank: 7 * 7 = 49.
        """
        # Split into 2x2 blocks of 2x2 matrices
        # A00 at (oi, oj), A01 at (oi, oj+2), A10 at (oi+2, oj), A11 at (oi+2, oj+2)
        # B00 at (oj, ok), B01 at (oj, ok+2), B10 at (oj+2, ok), B11 at (oj+2, ok+2)
        # C00 at (oi, ok), C01 at (oi, ok+2), C10 at (oi+2, ok), C11 at (oi+2, ok+2)
        
        # M1 = (A00+A11)(B00+B11)
        # This is a 2x2x2 Strassen where each "entry" is a 2x2 matrix
        # The 2x2x2 Strassen has 7 terms, each of which is a product of 2x2 matrices
        # Each 2x2 matrix product can be done with schoolbook (rank 4) or Strassen (rank 7)
        # For simplicity and correctness, let's use the direct approach:
        # M1 = (A00+A11)(B00+B11) means we compute (A00+A11) * (B00+B11)
        # This is a 4x4 * 4x4 product where the matrices are sparse (block diagonal)
        # Actually, let me think about this differently.
        
        # The 4x4x4 Strassen algorithm:
        # We have 4x4 matrices A, B and want C = AB
        # Split into 2x2 blocks of 2x2 matrices
        # M1 = (A00+A11)(B00+B11) - this is a 4x4 matrix product where A00+A11 has A00 in top-left, A11 in bottom-right
        # Similarly B00+B11 has B00 in top-left, B11 in bottom-right
        # So M1 is the product of two block-diagonal 4x4 matrices
        # M1 = [[A00, 0], [0, A11]] * [[B00, 0], [0, B11]] = [[A00*B00, 0], [0, A11*B11]]
        # So M1 contributes A00*B00 to C00 and A11*B11 to C11
        
        # Let me just implement the 7 Strassen terms directly, where each term is a 4x4x4 sub-problem
        # that can be decomposed using the known structure.
        
        # Actually, the cleanest way: for each of the 7 Strassen terms, the "matrix product" 
        # involves multiplying 2x2 matrices (at the block level). Each such 2x2 matrix product
        # is itself a 2x2x2 tensor with rank 7 (Strassen) or 4 (schoolbook).
        
        # Let me use schoolbook for the inner 2x2 products to keep it simple and correct.
        # Each Strassen term at the 4x4 level involves a 2x2x2 block multiplication.
        # The 2x2x2 block multiplication has rank 4 with schoolbook.
        # Total: 7 * 4 = 28 rank-1 terms for the 4x4x4 block.
        
        # Wait, that's not right either. Let me think more carefully.
        
        # The 4x4x4 matrix multiplication tensor T[i,j,k] for i,j,k in {0,1,2,3}.
        # Strassen splits this into 7 sub-tensors based on the 2x2 block structure.
        # Each sub-tensor corresponds to multiplying two 2x2 matrices (at the block level).
        # A 2x2 matrix multiplication has rank 4 (schoolbook) or 7 (Strassen).
        # But here, each "entry" of the 2x2 matrices is itself a 2x2 matrix.
        # So the sub-tensor is actually a 4x4x4 tensor with a specific sparse structure.
        
        # I think the cleanest implementation is to just enumerate the rank-1 terms directly.
        # For the 4x4x4 Strassen, the 7 terms are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a 4x4x4 tensor. Let me decompose each one.
        
        # M1 = (A00+A11)(B00+B11)
        # A00+A11 is a 4x4 matrix with A00 in top-left 2x2, A11 in bottom-right 2x2, zeros elsewhere.
        # B00+B11 is a 4x4 matrix with B00 in top-left 2x2, B11 in bottom-right 2x2, zeros elsewhere.
        # M1 = (A00+A11)(B00+B11) is a 4x4 matrix with A00*B00 in top-left, A11*B11 in bottom-right.
        # So M1 as a tensor: M1[i,j,k] is non-zero only when (i,j) and (j,k) are in the same 2x2 block.
        # Specifically:
        # - If i,j in {0,1} and j,k in {0,1}: M1[i,j,k] = A00[i,j]*B00[j,k] -> contributes to C00
        # - If i,j in {2,3} and j,k in {2,3}: M1[i,j,k] = A11[i-2,j-2]*B11[j-2,k-2] -> contributes to C11
        
        # So M1 can be decomposed into 4 rank-1 terms for A00*B00 and 4 for A11*B11 = 8 terms total.
        # But wait, that's schoolbook for each 2x2x2, giving 4+4=8 terms.
        
        # Let me just use schoolbook for all 4x4x4 to be safe. Rank = 64.
        # Actually, let me try a hybrid: use Strassen for the 2x2x2 sub-blocks within each Strassen term.
        
        # I'll implement it by calling strassen_2x2x2 for each of the 7 block-level terms,
        # but with appropriate offsets and coefficient combinations.
        
        # For M1 = (A00+A11)(B00+B11):
        # This equals A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each of these is a 2x2x2 multiplication (at the entry level).
        # So M1 = sum of 4 2x2x2 multiplications.
        # Each 2x2x2 has rank 7 (Strassen), so M1 has rank 28.
        # Total for all 7 terms: 7 * 28 = 196. That's worse than schoolbook!
        
        # I think I'm overcomplicating this. Let me just use schoolbook for the 4x4x4 block.
        # Rank = 64.
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_coeff = {(oi+i)*m + (oj+j): 1}
                    V_coeff = {(oj+j)*p + (ok+k): 1}
                    W_coeff = {(ok+k)*n + (oi+i): 1}
                    add_term(U_coeff, V_coeff, W_coeff)
    
    def schoolbook_block(oi, oj, ok, ni, nj, nk):
        """Schoolbook decomposition for an ni x nj x nk block at offset (oi, oj, ok)."""
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    U_coeff = {(oi+i)*m + (oj+j): 1}
                    V_coeff = {(oj+j)*p + (ok+k): 1}
                    W_coeff = {(ok+k)*n + (oi+i): 1}
                    add_term(U_coeff, V_coeff, W_coeff)
    
    # Block 1: 4x4x4 at offset (0, 0, 0)
    strassen_4x4x4(0, 0, 0)
    
    # Block 2: 4x4x1 at offset (0, 0, 4) - A[0:4, 0:4] * B[0:4, 4]
    schoolbook_block(0, 0, 4, 4, 4, 1)
    
    # Block 3: 4x1x4 at offset (0, 4, 0) - A[0:4, 4] * B[4, 0:4]
    schoolbook_block(0, 4, 0, 4, 1, 4)
    
    # Block 4: 4x1x1 at offset (0, 4, 4) - A[0:4, 4] * B[4, 4]
    schoolbook_block(0, 4, 4, 4, 1, 1)
    
    # Convert to matrices
    rank = len(U_terms)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for col, (U_coeff, V_coeff, W_coeff) in enumerate(zip(U_terms, V_terms, W_terms)):
        for idx, val in U_coeff.items():
            U[idx][col] += val
        for idx, val in V_coeff.items():
            V[idx][col] += val
        for idx, val in W_coeff.items():
            W[idx][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
