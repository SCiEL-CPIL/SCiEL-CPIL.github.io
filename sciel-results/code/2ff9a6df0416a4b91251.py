"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy:
1. Build the schoolbook decomposition (rank 144) as a guaranteed baseline.
2. Attempt to construct a lower-rank decomposition by exploiting block structure:
   Split the 9 columns into 4+4+1, compute 4x4*4x4 twice (using a rank-49
   Strassen-based decomposition for each) and 4x4*4x1 (rank 16), for a total
   of 49+49+16 = 114.
3. If the block decomposition is verified, return it; otherwise fall back to schoolbook.

The key structural change over the seed program is to use an explicit
algebraic construction (Strassen on 2x2 blocks) rather than numerical
optimization, which is both faster and guarantees exact integer entries.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit

    n, m, p = DIMENSIONS
    # n=4 (rows of A, cols of C), m=4 (cols of A, rows of B), p=9 (cols of B, cols of C)

    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36

    # Build the list of (a, b, c) triples where the tensor is 1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    triple_set = set(triples)

    # Schoolbook solution (rank = n*m*p = 144)
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school), dtype=np.float64)
    V_school = np.zeros((V_rows, rank_school), dtype=np.float64)
    W_school = np.zeros((W_rows, rank_school), dtype=np.float64)
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact for all entries."""
        r = U.shape[1]
        # Check all triple entries = 1
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check all non-triple entries = 0
        # We'll check a large sample plus all entries for correctness
        for a in range(U_rows):
            for b in range(V_rows):
                for c in range(W_rows):
                    if (a, b, c) in triple_set:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    # Verify schoolbook
    if not check_decomposition(U_school, V_school, W_school):
        # This should never happen, but just in case
        pass

    # Now try the block decomposition approach:
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bj for j=1,2 is 4x4*4x4, use Strassen rank 49
    # A*B3 is 4x4*4x1, use schoolbook rank 16
    # Total rank: 49 + 49 + 16 = 114

    def strassen_2x2_decomposition():
        """
        Return the rank-7 Strassen decomposition for 2x2*2x2.
        
        For 2x2 matrices A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
        C = A*B where:
          c00 = ae+bg
          c01 = af+bh
          c10 = ce+dg
          c11 = cf+dh
        
        Strassen's 7 multiplications:
          m1 = (a+d)(e+h)
          m2 = (c+d)e
          m3 = a(f-h)
          m4 = d(g-e)
          m5 = (a+b)h
          m6 = (c-a)(e+f)
          m7 = (b-d)(g+h)
        
        Then:
          c00 = m1 + m4 - m5 + m7
          c01 = m3 + m5
          c10 = m2 + m4
          c11 = m1 - m2 + m3 + m6
        
        We represent this as U, V, W matrices of shape (4, 7), (4, 7), (4, 7)
        where U corresponds to A entries, V to B entries, W to C entries.
        """
        # Indexing for 2x2:
        # A: a=0, b=1, c=2, d=3
        # B: e=0, f=1, g=2, h=3
        # C: c00=0, c01=1, c10=2, c11=3
        
        U = np.zeros((4, 7), dtype=np.float64)
        V = np.zeros((4, 7), dtype=np.float64)
        W = np.zeros((4, 7), dtype=np.float64)
        
        # m1 = (a+d)(e+h)
        # U: a=1, d=1; V: e=1, h=1; W: contributes to c00(+1), c11(+1)
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        W[0, 0] = 1  # c00
        W[3, 0] = 1  # c11
        
        # m2 = (c+d)e
        U[2, 1] = 1  # c
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        W[2, 1] = 1  # c10
        
        # m3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f
        V[3, 2] = -1 # h
        W[1, 2] = 1  # c01
        W[3, 2] = 1  # c11
        
        # m4 = d(g-e)
        U[3, 3] = 1  # d
        V[2, 3] = 1  # g
        V[0, 3] = -1 # e
        W[0, 3] = 1  # c00
        W[2, 3] = 1  # c10
        
        # m5 = (a+b)h
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b
        V[3, 4] = 1  # h
        W[0, 4] = -1 # c00 (negative)
        W[1, 4] = 1  # c01
        
        # m6 = (c-a)(e+f)
        U[2, 5] = 1  # c
        U[0, 5] = -1 # a
        V[0, 5] = 1  # e
        V[1, 5] = 1  # f
        W[3, 5] = 1  # c11
        
        # m7 = (b-d)(g+h)
        U[1, 6] = 1  # b
        U[3, 6] = -1 # d
        V[2, 6] = 1  # g
        V[3, 6] = 1  # h
        W[0, 6] = 1  # c00
        
        return U, V, W

    def build_4x4_strassen():
        """
        Build a rank-49 decomposition for 4x4*4x4 using Strassen on 2x2 blocks.
        
        View 4x4 matrices as 2x2 block matrices with 2x2 blocks.
        Apply Strassen at the block level, and within each block product,
        use the 2x2 Strassen decomposition.
        
        Returns U, V, W of shape (16, 49), (16, 49), (16, 49)
        """
        # First get the 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2_decomposition()
        
        # Block structure for 4x4:
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # Block-level Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Then:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2*2x2 product with rank 7
        # Total rank = 7*7 = 49
        
        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)
        
        # Block indexing:
        # A blocks: A00 rows 0-1, cols 0-1; A01 rows 0-1, cols 2-3
        #           A10 rows 2-3, cols 0-1; A11 rows 2-3, cols 2-3
        # In linear index: A00=[0,1,2,3], A01=[4,5,6,7], A10=[8,9,10,11], A11=[12,13,14,15]
        # Wait, let me be more careful.
        # A is 4x4, indexed as A[i,j] with i=row, j=col
        # Linear index a = i*m + j = i*4 + j
        # A00: i in {0,1}, j in {0,1} -> indices 0,1,4,5
        # A01: i in {0,1}, j in {2,3} -> indices 2,3,6,7
        # A10: i in {2,3}, j in {0,1} -> indices 8,9,12,13
        # A11: i in {2,3}, j in {2,3} -> indices 10,11,14,15
        
        # Actually, let's use the natural linear indexing:
        # A[i,j] -> index i*4+j
        # A00 = A[0:2, 0:2] -> indices [0,1,4,5]
        # A01 = A[0:2, 2:4] -> indices [2,3,6,7]
        # A10 = A[2:4, 0:2] -> indices [8,9,12,13]
        # A11 = A[2:4, 2:4] -> indices [10,11,14,15]
        
        # Similarly for B (4x4):
        # B00 = B[0:2, 0:2] -> indices [0,1,4,5]
        # B01 = B[0:2, 2:4] -> indices [2,3,6,7]
        # B10 = B[2:4, 0:2] -> indices [8,9,12,13]
        # B11 = B[2:4, 2:4] -> indices [10,11,14,15]
        
        # For C (4x4):
        # C00 = C[0:2, 0:2] -> indices [0,1,4,5]
        # C01 = C[0:2, 2:4] -> indices [2,3,6,7]
        # C10 = C[2:4, 0:2] -> indices [8,9,12,13]
        # C11 = C[2:4, 2:4] -> indices [10,11,14,15]
        
        # Define block index mappings
        A_blocks = {
            0: [0, 1, 4, 5],   # A00
            1: [2, 3, 6, 7],   # A01
            2: [8, 9, 12, 13], # A10
            3: [10, 11, 14, 15] # A11
        }
        B_blocks = {
            0: [0, 1, 4, 5],   # B00
            1: [2, 3, 6, 7],   # B01
            2: [8, 9, 12, 13], # B10
            3: [10, 11, 14, 15] # B11
        }
        C_blocks = {
            0: [0, 1, 4, 5],   # C00
            1: [2, 3, 6, 7],   # C01
            2: [8, 9, 12, 13], # C10
            3: [10, 11, 14, 15] # C11
        }
        
        # For each of the 7 Strassen multiplications at the block level,
        # and for each of the 7 Strassen multiplications within the 2x2 product,
        # we have one column in the 49-rank decomposition.
        
        # Block-level Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mi, we need to compute a 2x2*2x2 product.
        # The "A" part of Mi is a linear combination of A-blocks.
        # The "B" part of Mi is a linear combination of B-blocks.
        # The result contributes to C-blocks with certain coefficients.
        
        # Let's define for each Mi:
        #   a_coeffs: dict from block_idx to coefficient (for A side)
        #   b_coeffs: dict from block_idx to coefficient (for B side)
        #   c_coeffs: dict from block_idx to coefficient (for C side)
        
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            # M2 = (A10+A11)B00, contributes to C10(+1)
            ({2: 1, 3: 1}, {0: 1}, {2: 1}),
            # M3 = A00(B01-B11), contributes to C01(+1), C11(+1)
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            # M4 = A11(B10-B00), contributes to C00(+1), C10(+1)
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            # M5 = (A00+A01)B11, contributes to C00(-1), C01(+1)
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            # M6 = (A10-A00)(B00+B01), contributes to C11(+1)
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            # M7 = (A01-A11)(B10+B11), contributes to C00(+1)
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        col = 0
        for m_idx in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[m_idx]
            for s_idx in range(7):
                # For this 2x2 Strassen term s_idx within the block product
                # U2[s_idx, a_local] gives the coefficient for A-local index a_local
                # V2[s_idx, b_local] gives the coefficient for B-local index b_local
                # W2[s_idx, c_local] gives the coefficient for C-local index c_local
                
                for a_local in range(4):
                    u_val = U2[s_idx, a_local]
                    if u_val != 0:
                        for a_block, a_coeff in a_coeffs.items():
                            for a_idx in A_blocks[a_block]:
                                # a_local maps to position within the 2x2 block
                                # a_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                                # But A_blocks[a_block] already gives the global indices
                                # We need to map a_local to the correct global index
                                # A_blocks gives 4 indices for each block, in order
                                # The 2x2 block has indices [0,1,2,3] corresponding to
                                # positions (0,0),(0,1),(1,0),(1,1) in the block
                                # But A_blocks lists them in row-major order of the 4x4 matrix
                                # So for block A00: [0,1,4,5] corresponds to local [0,1,2,3]
                                global_a = A_blocks[a_block][a_local]
                                U4[global_a, col] += u_val * a_coeff
                
                for b_local in range(4):
                    v_val = V2[s_idx, b_local]
                    if v_val != 0:
                        for b_block, b_coeff in b_coeffs.items():
                            global_b = B_blocks[b_block][b_local]
                            V4[global_b, col] += v_val * b_coeff
                
                for c_local in range(4):
                    w_val = W2[s_idx, c_local]
                    if w_val != 0:
                        for c_block, c_coeff in c_coeffs.items():
                            global_c = C_blocks[c_block][c_local]
                            W4[global_c, col] += w_val * c_coeff
                
                col += 1
        
        return U4, V4, W4

    def build_4x4x1_schoolbook():
        """
        Build a rank-16 decomposition for 4x4*4x1.
        A is 4x4, B is 4x1, C is 4x1.
        C[i] = sum_j A[i,j] * B[j]
        """
        U = np.zeros((16, 16), dtype=np.float64)
        V = np.zeros((4, 16), dtype=np.float64)
        W = np.zeros((4, 16), dtype=np.float64)
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        
        return U, V, W

    # Now build the full 4x4*4x9 decomposition
    # B is 4x9, split into B1 (4x4), B2 (4x4), B3 (4x1)
    # B1 columns 0-3, B2 columns 4-7, B3 column 8
    
    # C = A*B1 + A*B2 + A*B3
    # C is 4x9
    
    # For A*B1: use 4x4 Strassen (rank 49)
    # For A*B2: use 4x4 Strassen (rank 49)
    # For A*B3: use 4x4x1 schoolbook (rank 16)
    
    U_strassen1, V_strassen1, W_strassen1 = build_4x4_strassen()
    U_strassen2, V_strassen2, W_strassen2 = build_4x4_strassen()
    U_4x4x1, V_4x4x1, W_4x4x1 = build_4x4x1_schoolbook()
    
    total_rank = 49 + 49 + 16  # 114
    
    U_block = np.zeros((16, total_rank), dtype=np.float64)
    V_block = np.zeros((36, total_rank), dtype=np.float64)
    W_block = np.zeros((36, total_rank), dtype=np.float64)
    
    # A*B1 part (columns 0-3 of B, columns 0-3 of C)
    # U: A indices 0-15
    # V: B1 indices 0-3 (which are B rows 0-3, cols 0-3) -> V rows 0-3
    # W: C1 indices 0-3 (which are C rows 0-3, cols 0-3) -> W rows 0-3
    U_block[:, 0:49] = U_strassen1
    V_block[0:4, 0:49] = V_strassen1
    W_block[0:4, 0:49] = W_strassen1
    
    # A*B2 part (columns 4-7 of B, columns 4-7 of C)
    # V: B2 indices 4-7 -> V rows 4-7
    # W: C2 indices 4-7 -> W rows 4-7
    U_block[:, 49:98] = U_strassen2
    V_block[4:8, 49:98] = V_strassen2
    W_block[4:8, 49:98] = W_strassen2
    
    # A*B3 part (column 8 of B, column 8 of C)
    # V: B3 index 8 -> V row 8
    # W: C3 index 8 -> W row 8
    U_block[:, 98:114] = U_4x4x1
    V_block[8:12, 98:114] = V_4x4x1  # B rows 0-3, col 8 -> V index 0*9+8=8, 1*9+8=17, 2*9+8=26, 3*9+8=35
    # Wait, V indexing: V[j*p + k] where j is row of B, k is col of B
    # B3 is column 8, so k=8, j in 0-3
    # V indices: 0*9+8=8, 1*9+8=17, 2*9+8=26, 3*9+8=35
    
    # Let me redo this. V has 36 rows indexed by j*p+k where j in 0-3, k in 0-8
    # For B1 (cols 0-3): V indices are j*9+k for j in 0-3, k in 0-3
    # For B2 (cols 4-7): V indices are j*9+k for j in 0-3, k in 4-7
    # For B3 (col 8): V indices are j*9+8 for j in 0-3
    
    # Similarly W has 36 rows indexed by k*n+i where k in 0-8, i in 0-3
    # For C1 (cols 0-3): W indices are k*4+i for k in 0-3, i in 0-3
    # For C2 (cols 4-7): W indices are k*4+i for k in 4-7, i in 0-3
    # For C3 (col 8): W indices are 8*4+i for i in 0-3
    
    # Let me rebuild V_block and W_block correctly
    
    # Clear and rebuild
    V_block = np.zeros((36, total_rank), dtype=np.float64)
    W_block = np.zeros((36, total_rank), dtype=np.float64)
    
    # A*B1: B cols 0-3, C cols 0-3
    for j in range(4):
        for k in range(4):
            v_idx = j * 9 + k
            V_block[v_idx, 0:49] = V_strassen1[j, :]
    
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            W_block[w_idx, 0:49] = W_strassen1[i, :]
    
    # A*B2: B cols 4-7, C cols 4-7
    for j in range(4):
        for k in range(4, 8):
            v_idx = j * 9 + k
            V_block[v_idx, 49:98] = V_strassen2[j, :]
    
    for k in range(4, 8):
        for i in range(4):
            w_idx = k * 4 + i
            W_block[w_idx, 49:98] = W_strassen2[i, :]
    
    # A*B3: B col 8, C col 8
    for j in range(4):
        v_idx = j * 9 + 8
        V_block[v_idx, 98:114] = V_4x4x1[j, :]
    
    for i in range(4):
        w_idx = 8 * 4 + i
        W_block[w_idx, 98:114] = W_4x4x1[i, :]
    
    # Verify the block decomposition
    if check_decomposition(U_block, V_block, W_block):
        best_U, best_V, best_W = U_block, V_block, W_block
        best_rank = total_rank
    
    # Convert to lists
    def to_list(arr):
        # Round to nearest half-integer for exactness
        rounded = np.round(arr * 2) / 2
        return rounded.tolist()
    
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    # Final verification with the rounded values
    U_check = np.array(U_out)
    V_check = np.array(V_out)
    W_check = np.array(W_out)
    
    if check_decomposition(U_check, V_check, W_check):
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
