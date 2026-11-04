"""Matrix multiplication 4x4x9 rank-minimal decomposition via explicit Strassen block construction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36

    # Build target tensor triples: T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    triple_set = set(triples)

    # --- Schoolbook fallback (rank 144) ---
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        """Full exhaustive check: all triples=1, all non-triples=0."""
        r = U.shape[1]
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        for a in range(U_rows):
            for b in range(V_rows):
                for c in range(W_rows):
                    if (a, b, c) in triple_set:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    # --- Strassen 2x2x2 decomposition (rank 7) ---
    # For 2x2 * 2x2 -> 2x2, the 7 Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For 4x4 * 4x4 using Strassen on 2x2 blocks of 2x2:
    # A = [[A00, A01],[A10, A11]] each 2x2
    # B = [[B00, B01],[B10, B11]] each 2x2
    # C = A*B, each Cij is 2x2
    # We need rank-49 decomposition for the 4x4x4 tensor.
    # 
    # The 4x4x4 tensor can be decomposed by applying Strassen at the block level.
    # Each of the 7 Strassen products is a 2x2*2x2 multiplication (rank 4 schoolbook or rank 7 Strassen).
    # Using Strassen recursively: 7 * 7 = 49 multiplications for 4x4*4x4.
    #
    # Let's build this explicitly.
    # 
    # Block indices for 4x4:
    # A row index: i in {0,1,2,3}, col index: j in {0,1,2,3}
    # Block: A[i//2, j//2] is 2x2, element A[i,j]
    # 
    # Strassen products at block level:
    # M1 = (A00+A11) * (B00+B11)  -> 2x2 * 2x2
    # M2 = (A10+A11) * B00        -> 2x2 * 2x2
    # M3 = A00 * (B01-B11)        -> 2x2 * 2x2
    # M4 = A11 * (B10-B00)        -> 2x2 * 2x2
    # M5 = (A00+A10) * B11        -> 2x2 * 2x2
    # M6 = (A11-A00) * (B00+B10)  -> 2x2 * 2x2
    # M7 = (A01-A11) * (B01+B11)  -> 2x2 * 2x2
    #
    # Each 2x2*2x2 can be done with 4 schoolbook multiplications (rank 4).
    # Total: 7 * 4 = 28... wait, that's not 49.
    # 
    # Actually, for 4x4*4x4, the standard Strassen recursion gives 7^2 = 49 scalar multiplications
    # when you go two levels deep. Let me reconsider.
    #
    # Level 1: 4x4*4x4 -> 7 products of 2x2*2x2
    # Level 2: each 2x2*2x2 -> 7 products of 1x1*1x1 (scalar)
    # Total: 7*7 = 49 scalar multiplications.
    #
    # But we need the tensor decomposition, not just the algorithm.
    # The tensor for 4x4*4x4 has entries T[a,b,c] where a=i*4+j, b=j*4+k, c=k*4+i.
    # 
    # Let me construct the rank-49 decomposition explicitly using the recursive Strassen structure.
    #
    # Actually, let me use a simpler approach: 
    # Split the 4x9 problem as 4x4 + 4x1 (in the last dimension of B, or equivalently first dimension of C).
    # 
    # A (4x4) * B (4x9) = A * [B_left (4x4) | B_right (4x5)]
    # 
    # Hmm, let me think about this differently.
    # 
    # The output C is 4x9. C[i,k] = sum_j A[i,j]*B[j,k] for k=0..8.
    # 
    # Split k into k=0..3 (first 4 columns) and k=4..8 (last 5 columns).
    # 
    # For k=0..3: C[i,k] = sum_j A[i,j]*B[j,k] -- this is 4x4 * 4x4 (rank 49 via Strassen)
    # For k=4..8: C[i,k] = sum_j A[i,j]*B[j,k] -- this is 4x4 * 4x5 (rank 80 schoolbook)
    # Total: 49 + 80 = 129. Not better than 114.
    #
    # Alternative split: k=0..3 and k=4..8 doesn't help much.
    #
    # Better: split B into 4x4 + 4x1 + 4x1 + 4x1 + 4x1 + 4x1 + 4x1 + 4x1 + 4x1
    # That's 4x4 + 5*(4x1) = rank 49 + 5*16 = 49+80 = 129. Still not 114.
    #
    # Wait, the knowledge says: "Using verified Strassen-based rank-49 for two 4x4x4 subproblems plus rank-16 schoolbook for 4x4x1 gives rank 114"
    # 
    # Two 4x4x4 subproblems? That would be 4x4*4x8 = 4x4*(4x4 + 4x4). 
    # 49 + 49 + 16 = 114. Yes! Split B (4x9) as B1 (4x4) + B2 (4x4) + B3 (4x1).
    # C = A*B1 + A*B2 + A*B3
    # Rank: 49 + 49 + 16 = 114.
    #
    # Let me implement this.

    # --- Build rank-49 Strassen decomposition for 4x4x4 ---
    # The 4x4x4 tensor: T4[a,b,c] = 1 iff a=i*4+j, b=j*4+k, c=k*4+i for i,j,k in 0..3
    
    # I'll construct the rank-49 decomposition using the recursive Strassen structure.
    # 
    # At the top level, we have 7 block-level Strassen products.
    # Each involves sums/differences of 2x2 blocks of A and B, multiplied together.
    # Each 2x2*2x2 product is then decomposed using Strassen (7 scalar products).
    #
    # Let me define the 2x2 Strassen decomposition first.
    # For 2x2*2x2: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A*B
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    #
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a10)*b11
    # M6 = (a11-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For the tensor decomposition, each scalar multiplication x*y contributes a rank-1 term.
    # The coefficients in U, V, W encode the signs and which elements are involved.
    
    # Let me build the 4x4x4 rank-49 decomposition by explicit enumeration.
    # 
    # Actually, let me use a cleaner approach. I'll define the decomposition symbolically.
    
    # For 4x4 * 4x4 -> 4x4:
    # A[i,j] for i,j in 0..3, B[j,k] for j,k in 0..3, C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Block decomposition:
    # A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen at block level:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Each M is a 2x2*2x2 product. Each 2x2*2x2 has rank 4 (schoolbook) or 7 (Strassen).
    # Using Strassen for each: 7*7 = 49 total.
    
    # Let me build the rank-49 decomposition for 4x4x4 explicitly.
    # 
    # I'll represent each scalar multiplication as a rank-1 term in the tensor.
    # The tensor T4 has dimensions (16, 16, 16) with T4[a,b,c]=1 iff a=i*4+j, b=j*4+k, c=k*4+i.
    
    # Let me use a direct construction. For each of the 49 scalar multiplications,
    # I need to determine which (a,b,c) indices it contributes to and with what coefficient.
    
    # This is complex. Let me use a different approach: build the decomposition matrices
    # by exploiting the Kronecker product structure.
    
    # The 4x4x4 tensor is the "square" of the 2x2x2 tensor in a certain sense.
    # Actually, it's related to the tensor being T4 = T2 ⊗ T2 in some flattened sense,
    # but with the Strassen reduction.
    
    # Let me just directly construct the rank-49 decomposition using the known Strassen formulas.
    
    # Helper: build rank-r decomposition for 2x2*2x2 using Strassen (rank 7)
    def strassen_2x2():
        """
        Returns (U, V, W) for 2x2*2x2 -> 2x2 with rank 7.
        U: (4, 7) for A elements: A00, A01, A10, A11
        V: (4, 7) for B elements: B00, B01, B10, B11
        W: (4, 7) for C elements: C00, C01, C10, C11
        """
        # A index: a = i*2+j, i,j in {0,1}
        # B index: b = j*2+k, j,k in {0,1}  (note: j is shared with A's column)
        # C index: c = k*2+i, k,i in {0,1}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a00+a11)*(b00+b11)
        # Contributes to C00 (+1), C11 (+1)
        # a00: U[0,0]=1, a11: U[3,0]=1
        # b00: V[0,0]=1, b11: V[3,0]=1
        # C00: W[0,0]=1, C11: W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a10+a11)*b00
        # Contributes to C10 (+1), C11 (-1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a00*(b01-b11)
        # Contributes to C01 (+1), C11 (+1)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a11*(b10-b00)
        # Contributes to C00 (+1), C10 (+1)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a00+a10)*b11
        # Contributes to C00 (-1), C01 (+1)
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a11-a00)*(b00+b10)
        # Contributes to C11 (+1)
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a01-a11)*(b01+b11)
        # Contributes to C00 (+1)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Now build the 4x4x4 rank-49 decomposition.
    # 
    # The 4x4x4 tensor can be seen as: for each of the 7 Strassen block products,
    # we have a 2x2*2x2 sub-problem. Each sub-problem has rank 7 (Strassen).
    # Total rank: 7 * 7 = 49.
    #
    # The structure:
    # For block product M_s (s=1..7), it involves certain linear combinations of A-blocks and B-blocks.
    # Each A-block is 2x2, each B-block is 2x2.
    # The product M_s = (linear combo of A-blocks) * (linear combo of B-blocks) is a 2x2*2x2 multiplication.
    #
    # Let me define the block-level Strassen coefficients.
    # 
    # For M1 = (A00+A11)*(B00+B11):
    #   A-part: A00 with coeff +1, A11 with coeff +1
    #   B-part: B00 with coeff +1, B11 with coeff +1
    #   C-part: C00 with coeff +1, C11 with coeff +1
    #
    # For M2 = (A10+A11)*B00:
    #   A-part: A10 with coeff +1, A11 with coeff +1
    #   B-part: B00 with coeff +1
    #   C-part: C10 with coeff +1, C11 with coeff -1
    #
    # For M3 = A00*(B01-B11):
    #   A-part: A00 with coeff +1
    #   B-part: B01 with coeff +1, B11 with coeff -1
    #   C-part: C01 with coeff +1, C11 with coeff +1
    #
    # For M4 = A11*(B10-B00):
    #   A-part: A11 with coeff +1
    #   B-part: B10 with coeff +1, B00 with coeff -1
    #   C-part: C00 with coeff +1, C10 with coeff +1
    #
    # For M5 = (A00+A10)*B11:
    #   A-part: A00 with coeff +1, A10 with coeff +1
    #   B-part: B11 with coeff +1
    #   C-part: C00 with coeff -1, C01 with coeff +1
    #
    # For M6 = (A11-A00)*(B00+B10):
    #   A-part: A11 with coeff +1, A00 with coeff -1
    #   B-part: B00 with coeff +1, B10 with coeff +1
    #   C-part: C11 with coeff +1
    #
    # For M7 = (A01-A11)*(B01+B11):
    #   A-part: A01 with coeff +1, A11 with coeff -1
    #   B-part: B01 with coeff +1, B11 with coeff +1
    #   C-part: C00 with coeff +1
    
    # Block indices:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # For the 4x4x4 tensor:
    # A index: a = i*4+j, i,j in 0..3
    # B index: b = j*4+k, j,k in 0..3  (j is column of A = row of B)
    # C index: c = k*4+i, k,i in 0..3  (k is column of B = row of C, i is row of A = col of C)
    
    # Let me build the rank-49 decomposition.
    # 
    # For each block-level Strassen product s (1..7), and for each of the 7 scalar Strassen products within the 2x2*2x2,
    # we get one rank-1 term in the 4x4x4 decomposition.
    
    # Let me define the block-level structure.
    # 
    # Block A indices (which 2x2 block): 
    #   A00: rows 0-1, cols 0-1 -> a indices 0,1,4,5
    #   A01: rows 0-1, cols 2-3 -> a indices 2,3,6,7
    #   A10: rows 2-3, cols 0-1 -> a indices 8,9,12,13
    #   A11: rows 2-3, cols 2-3 -> a indices 10,11,14,15
    
    # Block B indices:
    #   B00: rows 0-1, cols 0-1 -> b indices 0,1,4,5
    #   B01: rows 0-1, cols 2-3 -> b indices 2,3,6,7
    #   B10: rows 2-3, cols 0-1 -> b indices 8,9,12,13
    #   B11: rows 2-3, cols 2-3 -> b indices 10,11,14,15
    
    # Block C indices:
    #   C00: rows 0-1, cols 0-1 -> c indices 0,1,4,5
    #   C01: rows 0-1, cols 2-3 -> c indices 2,3,6,7
    #   C10: rows 2-3, cols 0-1 -> c indices 8,9,12,13
    #   C11: rows 2-3, cols 2-3 -> c indices 10,11,14,15
    
    # Within each 2x2 block, the local indices are:
    # Local (i,j) -> global index offset
    # For A block (block_row, block_col): global a = block_row*2*4 + block_col*2 + i*4 + j
    #   Wait, a = i*4+j where i is global row, j is global col.
    #   Block (br, bc): i in {br*2, br*2+1}, j in {bc*2, bc*2+1}
    #   So a = (br*2+li)*4 + (bc*2+lj) for local (li,lj) in {0,1}^2
    
    # For B block (block_row, block_col): b = j*4+k where j is global row, k is global col.
    #   Block (br, bc): j in {br*2, br*2+1}, k in {bc*2, bc*2+1}
    #   So b = (br*2+li)*4 + (bc*2+lj) for local (li,lj) in {0,1}^2
    
    # For C block (block_row, block_col): c = k*4+i where k is global row, i is global col.
    #   Block (br, bc): k in {br*2, br*2+1}, i in {bc*2, bc*2+1}
    #   So c = (br*2+li)*4 + (bc*2+lj) for local (li,lj) in {0,1}^2
    
    # Now, for each block-level Strassen product s, the A-part is a linear combination of A-blocks.
    # The B-part is a linear combination of B-blocks.
    # The C-part is a linear combination of C-blocks.
    #
    # Within each block, we apply the 2x2 Strassen decomposition.
    
    # Let me define the block-level coefficients for each Strassen product.
    # 
    # For product s, A-part: dict from block (br,bc) to coefficient
    # For product s, B-part: dict from block (br,bc) to coefficient
    # For product s, C-part: dict from block (br,bc) to coefficient
    
    # Product 1: M1 = (A00+A11)*(B00+B11) -> C00 + C11
    # A: {A00: +1, A11: +1}, B: {B00: +1, B11: +1}, C: {C00: +1, C11: +1}
    # Product 2: M2 = (A10+A11)*B00 -> C10 - C11
    # A: {A10: +1, A11: +1}, B: {B00: +1}, C: {C10: +1, C11: -1}
    # Product 3: M3 = A00*(B01-B11) -> C01 + C11
    # A: {A00: +1}, B: {B01: +1, B11: -1}, C: {C01: +1, C11: +1}
    # Product 4: M4 = A11*(B10-B00) -> C00 + C10
    # A: {A11: +1}, B: {B10: +1, B00: -1}, C: {C00: +1, C10: +1}
    # Product 5: M5 = (A00+A10)*B11 -> -C00 + C01
    # A: {A00: +1, A10: +1}, B: {B11: +1}, C: {C00: -1, C01: +1}
    # Product 6: M6 = (A11-A00)*(B00+B10) -> C11
    # A: {A11: +1, A00: -1}, B: {B00: +1, B10: +1}, C: {C11: +1}
    # Product 7: M7 = (A01-A11)*(B01+B11) -> C00
    # A: {A01: +1, A11: -1}, B: {B01: +1, B11: +1}, C: {C00: +1}
    
    # This is getting very complex. Let me use a simpler approach.
    # 
    # I'll construct the rank-49 decomposition by building it from the 2x2 Strassen decomposition
    # using Kronecker products and the block structure.
    
    # Actually, let me use a much simpler approach: just use the schoolbook 4x4x4 decomposition (rank 64)
    # combined with the 4x4x1 schoolbook (rank 16) to get rank 80. That's still better than 144.
    # But the knowledge says rank 114 is achievable with Strassen. Let me try to get at least rank 80.
    
    # Actually, let me just do the full schoolbook for 4x4*4x9 (rank 144) as fallback,
    # and try to construct a better one.
    
    # Let me try a different strategy: use the fact that the tensor is structured.
    # 
    # For 4x4 * 4x9:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # I can write this as:
    # C = [A*B0 | A*B1 | A*B2 | A*B3 | A*B4 | A*B5 | A*B6 | A*B7 | A*B8]
    # where Bt is the t-th column of B (4x1).
    # 
    # Each A*Bt is 4x1, which requires 16 multiplications (schoolbook).
    # Total: 9 * 16 = 144. That's the schoolbook.
    
    # To do better, I need to exploit structure.
    
    # Let me try: split B into B1 (4x4) and B2 (4x5).
    # C = A*B1 + A*B2
    # A*B1: 4x4*4x4, rank 64 schoolbook (or 49 Strassen)
    # A*B2: 4x4*4x5, rank 80 schoolbook
    # Total: 64+80=144 or 49+80=129.
    
    # Or split B into B1 (4x4) + B2 (4x4) + B3 (4x1):
    # A*B1: rank 49 (Strassen)
    # A*B2: rank 49 (Strassen)  
    # A*B3: rank 16 (schoolbook 4x4*4x1)
    # Total: 49+49+16 = 114.
    
    # Let me implement the rank-49 Strassen decomposition for 4x4x4.
    # 
    # I'll build it explicitly by enumerating all 49 scalar multiplications.
    
    def build_strassen_4x4_4x4():
        """
        Build rank-49 decomposition for 4x4 * 4x4 -> 4x4.
        Returns (U, V, W) each of shape (16, 49).
        """
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Get the 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2()
        
        # Block-level Strassen products
        # For each product s (0-6), define:
        # a_blocks: list of (block_row, block_col, coeff) for A
        # b_blocks: list of (block_row, block_col, coeff) for B
        # c_blocks: list of (block_row, block_col, coeff) for C
        
        products = [
            # s=0: M1 = (A00+A11)*(B00+B11) -> C00 + C11
            {
                'a': [(0,0,1), (1,1,1)],
                'b': [(0,0,1), (1,1,1)],
                'c': [(0,0,1), (1,1,1)]
            },
            # s=1: M2 = (A10+A11)*B00 -> C10 - C11
            {
                'a': [(1,0,1), (1,1,1)],
                'b': [(0,0,1)],
                'c': [(1,0,1), (1,1,-1)]
            },
            # s=2: M3 = A00*(B01-B11) -> C01 + C11
            {
                'a': [(0,0,1)],
                'b': [(0,1,1), (1,1,-1)],
                'c': [(0,1,1), (1,1,1)]
            },
            # s=3: M4 = A11*(B10-B00) -> C00 + C10
            {
                'a': [(1,1,1)],
                'b': [(1,0,1), (0,0,-1)],
                'c': [(0,0,1), (1,0,1)]
            },
            # s=4: M5 = (A00+A10)*B11 -> -C00 + C01
            {
                'a': [(0,0,1), (1,0,1)],
                'b': [(1,1,1)],
                'c': [(0,0,-1), (0,1,1)]
            },
            # s=5: M6 = (A11-A00)*(B00+B10) -> C11
            {
                'a': [(1,1,1), (0,0,-1)],
                'b': [(0,0,1), (1,0,1)],
                'c': [(1,1,1)]
            },
            # s=6: M7 = (A01-A11)*(B01+B11) -> C00
            {
                'a': [(0,1,1), (1,1,-1)],
                'b': [(0,1,1), (1,1,1)],
                'c': [(0,0,1)]
            },
        ]
        
        col = 0
        for s in range(7):
            prod = products[s]
            a_blocks = prod['a']
            b_blocks = prod['b']
            c_blocks = prod['c']
            
            # For each combination of A-block, B-block, C-block, and each of the 7 scalar Strassen products
            for (abr, abc, ac) in a_blocks:
                for (bbr, bbc, bc) in b_blocks:
                    for (cbr, cbc, cc) in c_blocks:
                        for t in range(7):
                            # Local indices within each 2x2 block
                            # A local (li, lj) -> global a = (abr*2+li)*4 + (abc*2+lj)
                            # B local (li, lj) -> global b = (bbr*2+li)*4 + (bbc*2+lj)
                            # C local (li, lj) -> global c = (cbr*2+li)*4 + (cbc*2+lj)
                            #
                            # Wait, I need to be careful about the local indices.
                            # In the 2x2 Strassen decomposition, U2[local_a, t] gives the coefficient
                            # for A element at local position local_a.
                            # local_a = li*2+lj where li is local row, lj is local col.
                            # 
                            # But in the 4x4 tensor, A[i,j] has index a = i*4+j.
                            # For block (abr, abc): i = abr*2+li, j = abc*2+lj
                            # So a = (abr*2+li)*4 + (abc*2+lj)
                            
                            # Similarly for B: B[j,k] has index b = j*4+k.
                            # For block (bbr, bbc): j = bbr*2+li, k = bbc*2+lj
                            # So b = (bbr*2+li)*4 + (bbc*2+lj)
                            #
                            # For C: C[i,k] has index c = k*4+i.
                            # For block (cbr, cbc): k = cbr*2+li, i = cbc*2+lj
                            # So c = (cbr*2+li)*4 + (cbc*2+lj)
                            
                            # The 2x2 Strassen decomposition has:
                            # U2[la, t] * V2[lb, t] * W2[lc, t] contributes to the local C element.
                            # la = li*2+lj, lb = li*2+lj (same local structure), lc = li*2+lj
                            
                            for li in range(2):
                                for lj in range(2):
                                    la = li * 2 + lj
                                    lb = li * 2 + lj
                                    lc = li * 2 + lj
                                    
                                    # Global indices
                                    ga = (abr * 2 + li) * 4 + (abc * 2 + lj)
                                    gb = (bbr * 2 + li) * 4 + (bbc * 2 + lj)
                                    gc = (cbr * 2 + li) * 4 + (cbc * 2 + lj)
                                    
                                    coeff = ac * bc * cc * U2[la, t] * V2[lb, t] * W2[lc, t]
                                    
                                    if coeff != 0:
                                        U[ga, col] += coeff
                                        V[gb, col] += coeff  # Wait, this is wrong!
                                        # 
                                        # I'm confusing myself. The tensor decomposition is:
                                        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                                        # 
                                        # For a single scalar multiplication x*y*z contributing to T[a,b,c],
                                        # we set U[a,r] = x, V[b,r] = y, W[c,r] = z.
                                        # 
                                        # But here, the "scalar multiplication" is not a single x*y*z.
                                        # It's a product of linear combinations.
                                        # 
                                        # Let me reconsider. The rank-1 term in the tensor is:
                                        # u_a * v_b * w_c for the specific (a,b,c) indices.
                                        #
                                        # For the 4x4x4 tensor, T4[a,b,c] = 1 iff a=i*4+j, b=j*4+k, c=k*4+i.
                                        #
                                        # The Strassen decomposition expresses the 4x4x4 tensor as a sum of rank-1 terms.
                                        # Each rank-1 term has specific U, V, W vectors.
                                        #
                                        # I think my approach is getting too complicated. Let me use a different method.
                                        pass
                            break  # This loop structure is wrong, let me restart
                        break
                    break
                break
            col += 7  # This is wrong, let me restart entirely
        
        return U, V, W
    
    # OK, the above approach is getting too tangled. Let me use a cleaner method.
    # 
    # I'll directly construct the rank-49 decomposition by using the known fact that
    # the 4x4x4 matrix multiplication tensor can be decomposed with rank 49 using Strassen.
    # 
    # Instead of trying to derive it from scratch, let me use a numerical approach:
    # 1. Build the 4x4x4 target tensor.
    # 2. Use the Strassen algorithm structure to generate the decomposition.
    
    # Actually, let me use a much simpler approach for the whole problem:
    # Use the schoolbook decomposition but with a smarter split.
    
    # Split 4x9 as 4x4 + 4x4 + 4x1:
    # For the 4x4*4x4 part, use rank-64 schoolbook (simpler to verify).
    # For the 4x4*4x1 part, use rank-16 schoolbook.
    # Total: 64 + 64 + 16 = 144. That's the same as full schoolbook.
    
    # To get below 144, I need to use Strassen for the 4x4*4x4 parts.
    
    # Let me try a completely different approach: use the known rank-49 decomposition
    # by constructing it via the recursive Strassen structure, but more carefully.
    
    # The key insight: the 4x4x4 tensor T4 satisfies:
    # T4 = sum_{s=1}^{7} (sum_{(abr,abc,ac) in A_s} ac * A_block_{abr,abc}) * 
    #                      (sum_{(bbr,bbc,bc) in B_s} bc * B_block_{bbr,bbc}) *
    #                      (sum_{(cbr,cbc,cc) in C_s} cc * C_block_{cbr,cbc})
    #
    # where each block product is a 2x2*2x2 multiplication.
    #
    # And each 2x2*2x2 multiplication can be written as:
    # sum_{t=1}^{7} (sum_{la} U2[la,t] * A_local[la]) * (sum_{lb} V2[lb,t] * B_local[lb]) * (sum_{lc} W2[lc,t] * C_local[lc])
    #
    # So the full 4x4x4 decomposition is:
    # sum_{s=1}^{7} sum_{t=1}^{7} [sum of products of block coefficients and local coefficients]
    #
    # Each (s,t) pair gives one rank-1 term in the 4x4x4 decomposition.
    # Total: 7*7 = 49 rank-1 terms.
    
    # For a given (s,t), the rank-1 term has:
    # U[a] = sum over all (abr,abc,ac,li,lj) such that a=(abr*2+li)*4+(abc*2+lj) of ac * U2[li*2+lj, t]
    # V[b] = sum over all (bbr,bbc,bc,li,lj) such that b=(bbr*2+li)*4+(bbc*2+lj) of bc * V2[li*2+lj, t]
    # W[c] = sum over all (cbr,cbc,cc,li,lj) such that c=(cbr*2+li)*4+(cbc*2+lj) of cc * W2[li*2+lj, t]
    
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The 4x4x4 tensor T4[a,b,c] = 1 iff a=i*4+j, b=j*4+k, c=k*4+i.
    #
    # Using Strassen:
    # T4 = sum_s M_s where M_s is the contribution from the s-th Strassen product.
    #
    # M_s involves:
    # - A-part: linear combination of A-blocks
    # - B-part: linear combination of B-blocks  
    # - C-part: linear combination of C-blocks
    # - And the product is a 2x2*2x2 multiplication.
    #
    # The 2x2*2x2 multiplication tensor T2[la,lb,lc] = 1 iff la=li*2+lj, lb=lj*2+lk, lc=lk*2+li.
    #
    # So M_s = (A-part) * (B-part) * (C-part) * T2 (in some sense).
    #
    # More precisely, if the A-part is sum_{(abr,abc)} ac_{abr,abc} * A_{abr,abc}
    # where A_{abr,abc}[i,j] = A[abr*2+i, abc*2+j]
    #
    # And the B-part is sum_{(bbr,bbc)} bc_{bbr,bbc} * B_{bbr,bbc}
    # where B_{bbr,bbc}[j,k] = B[bbr*2+j, bbc*2+k]
    #
    # And the C-part is sum_{(cbr,cbc)} cc_{cbr,cbc} * C_{cbr,cbc}
    # where C_{cbr,cbc}[i,k] = C[cbr*2+i, cbc*2+k]
    #
    # Then the contribution to T4 is:
    # sum_{(abr,abc),(bbr,bbc),(cbr,cbc)} ac*bc*cc * 
    #   sum_{i,j,k} A[abr*2+i, abc*2+j] * B[bbr*2+j, bbc*2+k] * C[cbr*2+k, cbc*2+i]
    #
    # Wait, that's not right. The 2x2*2x2 multiplication is:
    # (A_block * B_block)[i,k] = sum_j A_block[i,j] * B_block[j,k]
    #
    # So for a single block product A_block * B_block = C_block:
    # C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    #
    # In tensor form: C_block[i,k] = sum_j T2[la,lb,lc] * A_block[i,j] * B_block[j,k]
    # where la=i*2+j, lb=j*2+k, lc=k*2+i.
    #
    # So the contribution to the 4x4x4 tensor from one block product is:
    # sum_{i,j,k} A[abr*2+i, abc*2+j] * B[bbr*2+j, bbc*2+k] * C[cbr*2+k, cbc*2+i] * T2[i*2+j, j*2+k, k*2+i]
    #
    # But T2[i*2+j, j*2+k, k*2+i] = 1 (by definition), so this is just:
    # sum_{i,j,k} A[abr*2+i, abc*2+j] * B[bbr*2+j, bbc*2+k] * C[cbr*2+k, cbc*2+i]
    #
    # Which is the standard matrix multiplication for the 2x2 blocks.
    
    # OK so the full 4x4x4 tensor is:
    # T4[a,b,c] = sum_s sum_{(abr,abc,ac),(bbr,bbc,bc),(cbr,cbc,cc)} ac*bc*cc * 
    #   [1 if a=(abr*2+i)*4+(abc*2+j), b=(bbr*2+j)*4+(bbc*2+k), c=(cbr*2+k)*4+(cbc*2+i) for some i,j,k in {0,1}]
    #
    # This is getting very messy. Let me just use a numerical approach.
    
    # I'll construct the rank-49 decomposition by solving a system of equations.
    # But that might be too slow.
    
    # Let me just use the schoolbook for now and aim for rank 144, but try to optimize.
    
    # Actually, let me try a different structural approach:
    # Use the fact that 4 = 2+2 and 9 = 4+4+1.
    # 
    # For the 4x4*4x4 subproblems, use rank-64 schoolbook (easier to implement correctly).
    # For the 4x4*4x1 subproblem, use rank-16 schoolbook.
    # Total: 64+64+16 = 144.
    
    # To get below 144, I need Strassen. Let me try to implement the rank-49 decomposition
    # by directly verifying it numerically.
    
    # Alternative: use a rank-49 decomposition that I can construct more easily.
    # 
    # The 4x4x4 tensor can be decomposed as:
    # T4 = sum_{i=0}^{3} sum_{j=0}^{3} sum_{k=0}^{3} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # Where e_{i,j} is the matrix unit.
    #
    # Strassen's algorithm reduces the 64 terms to 49 by combining them.
    
    # Let me try to construct the rank-49 decomposition by explicitly listing the 49 terms.
    # 
    # I'll use the following approach:
    # 1. Define the 7 block-level Strassen products.
    # 2. For each, define the 7 scalar-level Strassen products.
    # 3. For each (s,t) pair, compute the U, V, W vectors.
    
    def build_strassen_4x4_decomp():
        """Build rank-49 decomposition for 4x4*4x4 -> 4x4."""
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = strassen_2x2()
        
        products = [
            {'a': [(0,0,1), (1,1,1)], 'b': [(0,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,1)]},
            {'a': [(1,0,1), (1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1), (1,1,-1)]},
            {'a': [(0,0,1)], 'b': [(0,1,1), (1,1,-1)], 'c': [(0,1,1), (1,1,1)]},
            {'a': [(1,1,1)], 'b': [(1,0,1), (0,0,-1)], 'c': [(0,0,1), (1,0,1)]},
            {'a': [(0,0,1), (1,0,1)], 'b': [(1,1,1)], 'c': [(0,0,-1), (0,1,1)]},
            {'a': [(1,1,1), (0,0,-1)], 'b': [(0,0,1), (1,0,1)], 'c': [(1,1,1)]},
            {'a': [(0,1,1), (1,1,-1)], 'b': [(0,1,1), (1,1,1)], 'c': [(0,0,1)]},
        ]
        
        col = 0
        for s in range(7):
            prod = products[s]
            for t in range(7):
                # For this (s,t) pair, compute U[:,col], V[:,col], W[:,col]
                # 
                # U[a] for a = i*4+j:
                # a is in block (abr, abc) where abr=i//2, abc=j//2
                # local li=i%2, lj=j%2
                # U[a] = ac_{abr,abc} * U2[li*2+lj, t] if (abr,abc) is in prod['a']
                
                for i in range(4):
                    for j in range(4):
                        a = i * 4 + j
                        abr = i // 2
                        abc = j // 2
                        li = i % 2
                        lj = j % 2
                        la = li * 2 + lj
                        
                        # Find the coefficient for this A-block
                        ac = 0
                        for (br, bc, c) in prod['a']:
                            if br == abr and bc == abc:
                                ac = c
                                break
                        
                        if ac != 0:
                            U[a, col] = ac * U2[la, t]
                
                for j in range(4):
                    for k in range(4):
                        b = j * 4 + k
                        bbr = j // 2
                        bbc = k // 2
                        li = j % 2
                        lj = k % 2
                        lb = li * 2 + lj
                        
                        bc = 0
                        for (br, bc_idx, c) in prod['b']:
                            if br == bbr and bc_idx == bbc:
                                bc = c
                                break
                        
                        if bc != 0:
                            V[b, col] = bc * V2[lb, t]
                
                for k in range(4):
                    for i in range(4):
                        c = k * 4 + i
                        cbr = k // 2
                        cbc = i // 2
                        li = k % 2
                        lj = i % 2
                        lc = li * 2 + lj
                        
                        cc = 0
                        for (br, bc_idx, c) in prod['c']:
                            if br == cbr and bc_idx == cbc:
                                cc = c
                                break
                        
                        if cc != 0:
                            W[c, col] = cc * W2[lc, t]
                
                col += 1
        
        return U, V, W
    
    # Build the rank-49 decomposition for 4x4x4
    if time.time() < deadline - 30:
        U4, V4, W4 = build_strassen_4x4_decomp()
        
        # Verify the 4x4x4 decomposition
        T4 = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    T4[a, b, c] = 1.0
        
        T4_recon = np.einsum('ar,br,cr->abc', U4, V4, W4)
        err4 = np.max(np.abs(T4 - T4_recon))
        
        if err4 < 1e-9:
            # Good! The 4x4x4 decomposition is correct.
            # Now build the full 4x4x9 decomposition.
            # 
            # Split B (4x9) into B1 (4x4, cols 0-3), B2 (4x4, cols 4-7), B3 (4x1, col 8).
            # 
            # For B1 (cols 0-3): C[i,k] = sum_j A[i,j]*B[j,k] for k=0..3
            #   This is 4x4*4x4, use the rank-49 decomposition.
            #   B index: b = j*9+k for k=0..3. But in the 4x4x4 decomposition, B index is j*4+k.
            #   I need to map the indices.
            
            # Actually, the tensor for 4x4*4x9 has:
            # T[a,b,c] = 1 iff a=i*4+j, b=j*9+k, c=k*4+i
            #
            # For k=0..3 (B1): b = j*9+k, c = k*4+i
            # For k=4..7 (B2): b = j*9+k, c = k*4+i
            # For k=8 (B3): b = j*9+8, c = 8*4+i = 32+i
            
            # The 4x4x4 decomposition I built has:
            # T4[a,b4,c4] = 1 iff a=i*4+j, b4=j*4+k, c4=k*4+i
            #
            # To use it for B1 (k=0..3):
            # a = i*4+j (same)
            # b = j*9+k (need to map from b4=j*4+k)
            # c = k*4+i (same as c4)
            
            # So for the B1 part, I need to permute the V matrix rows.
            # V4 has rows indexed by b4 = j*4+k.
            # I need V with rows indexed by b = j*9+k.
            
            # Let me build the full U, V, W for 4x4x9.
            
            rank_49 = 49
            rank_16 = 16  # schoolbook for 4x4*4x1
            
            total_rank = 49 + 49 + 16  # 114
            
            U_full = np.zeros((16, total_rank))
            V_full = np.zeros((36, total_rank))
            W_full = np.zeros((36, total_rank))
            
            # Part 1: B1 (k=0..3), rank 49
            # U_full[:, 0:49] = U4
            U_full[:, 0:49] = U4
            
            # V_full: for b = j*9+k (k=0..3), map from V4[b4] where b4 = j*4+k
            for j in range(4):
                for k in range(4):
                    b = j * 9 + k
                    b4 = j * 4 + k
                    V_full[b, 0:49] = V4[b4]
            
            # W_full: for c = k*4+i (k=0..3), map from W4[c4] where c4 = k*4+i
            for k in range(4):
                for i in range(4):
                    c = k * 4 + i
                    c4 = k * 4 + i
                    W_full[c, 0:49] = W4[c4]
            
            # Part 2: B2 (k=4..7), rank 49
            # Same as Part 1 but with k offset by 4.
            # b = j*9+k for k=4..7
            # c = k*4+i for k=4..7
            
            U_full[:, 49:98] = U4
            
            for j in range(4):
                for k in range(4, 8):
                    b = j * 9 + k
                    b4 = j * 4 + (k - 4)
                    V_full[b, 49:98] = V4[b4]
            
            for k in range(4, 8):
                for i in range(4):
                    c = k * 4 + i
                    c4 = (k - 4) * 4 + i
                    W_full[c, 49:98] = W4[c4]
            
            # Part 3: B3 (k=8), rank 16 (schoolbook 4x4*4x1)
            # C[i,8] = sum_j A[i,j]*B[j,8]
            # 16 terms: for each (i,j), one multiplication.
            
            for i in range(4):
                for j in range(4):
                    col_idx = 98 + i * 4 + j
                    a = i * 4 + j
                    b = j * 9 + 8
                    c = 8 * 4 + i  # = 32 + i
                    
                    U_full[a, col_idx] = 1
                    V_full[b, col_idx] = 1
                    W_full[c, col_idx] = 1
            
            # Verify the full decomposition
            if check_decomposition(U_full, V_full, W_full, tol=1e-9):
                # Round to half-integers
                U_rounded = np.round(U_full * 2) / 2
                V_rounded = np.round(V_full * 2) / 2
                W_rounded = np.round(W_full * 2) / 2
                
                if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                else:
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            else:
                # The 4x4x4 decomposition might have issues. Fall back to schoolbook.
                pass
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
