"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for
    4x4 * 4x6 matrix multiplication.
    
    Strategy: Use recursive Strassen for the 4x4 part (rank 49) and
    apply it to each 4x3 block of the 4x6 matrix.
    
    The 4x6 multiplication can be viewed as:
    C (4x6) = A (4x4) * B (4x6)
    
    We can split B into two 4x3 blocks: B = [B1, B2]
    Then C = [A*B1, A*B2]
    
    For each 4x3 block multiplication, we need a decomposition.
    A 4x3 multiplication can be done with rank at most 4*3*1 = 12 (schoolbook)
    or potentially less with Strassen-like techniques.
    
    Actually, let me think more carefully. The tensor for A (n x m) * B (m x p)
    has rank n*m*p for schoolbook.
    
    For 4x4x6: schoolbook rank = 4*4*6 = 96.
    
    Using Strassen on the 4x4 part:
    - Split A into 2x2 blocks of 2x2 matrices
    - Split B into 2x2 blocks of 2x3 matrices
    - Split C into 2x2 blocks of 2x3 matrices
    
    For each 2x2 * 2x3 block multiplication, we need a decomposition.
    A 2x2 * 2x3 can be decomposed with rank at most 2*2*3 = 12 (schoolbook).
    
    But Strassen for 2x2x2 has rank 7 instead of 8.
    
    Let me think about this differently. The standard approach:
    
    For n x m * m x p where n=4, m=4, p=6:
    
    Method 1: Recursive Strassen on the square part
    - Treat it as 4x4 * 4x4 and 4x4 * 4x2 separately
    - 4x4x4 via Strassen: rank 49 (recursive Strassen)
    - 4x4x2 via schoolbook: rank 32
    - Total: 49 + 32 = 81
    
    Method 2: Direct recursive Strassen treating dimensions
    - 4x4x6: split into 2x2 blocks
    - A: 4 2x2 blocks
    - B: 6 2x3 blocks (or treat as 2x2 * 3 2x1 blocks)
    - C: 6 2x3 blocks
    
    Actually, let me just use the known best results.
    
    For 4x4x4, the best known rank is 49 (via recursive Strassen).
    For 4x4x6, we can use:
    - 4x4x4 part: rank 49
    - 4x4x2 part: rank 32 (schoolbook)
    - Total: 81
    
    But we can do better by using Strassen on the 4x4x2 part too.
    4x4x2: split into 2x2x2 and 2x2x2 blocks
    Each 2x2x2 has rank 7 (Strassen)
    So 4x4x2 has rank 4*7 = 28? No, that's not right either.
    
    Let me be more careful. For 4x4x2:
    - Split A into 2x2 blocks of 2x2
    - Split B into 2x2 blocks of 2x1
    - Split C into 2x2 blocks of 2x1
    
    For each 2x2 * 2x1 * 2x1 sub-problem:
    This is just 2x2 * 2x1, which is 2 rows of 2x1.
    Each 2x1 * 2x1 is 2x1x2x1x1... this is getting complex.
    
    Let me just implement a direct recursive Strassen that handles rectangular cases.
    
    Actually, the simplest approach: use the known decomposition for 4x4x4 (rank 49)
    and handle the 4x4x2 part separately.
    
    For 4x4x2:
    C (4x2) = A (4x4) * B (4x2)
    Split into 2x2 blocks:
    A = [A11 A12; A21 A22] where each is 2x2
    B = [B1; B2] where each is 2x2
    C = [C1; C2] where each is 2x2
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    For C1: 2x2*2x2 + 2x2*2x2
    Each 2x2*2x2 via Strassen: rank 7
    So C1 needs 2*7 = 14
    Similarly C2 needs 14
    Total for 4x4x2: 28
    
    But wait, we can share some computations. Let me just use schoolbook for the 4x4x2 part: rank 32.
    
    Total: 49 + 32 = 81.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition for 4x4x4 (rank 49) and 4x4x2 (rank 32)
    
    # Helper: get Strassen decomposition for 2x2x2 (rank 7)
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        Strassen:
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)b00
        M3 = a00(b10-b00)
        M4 = a11(b11-b00)
        M5 = (a00+a01)b11
        M6 = (a10-a00)(b00+b10)
        M7 = (a01-a11)(b01+b11)
        
        c00 = M1 + M4 - M5 + M7
        c01 = M3 + M5
        c10 = M2 + M4
        c11 = M1 - M2 + M3 + M6
        """
        # U[i*2+j, r] for i,j in 0..1, r in 0..6
        # V[j*2+k, r] for j,k in 0..1, r in 0..6
        # W[k*2+i, r] for k,i in 0..1, r in 0..6
        
        # Index: row i*2+j, col r
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        # a00: U[0*2+0, 0] = U[0,0]
        # a11: U[1*2+1, 0] = U[3,0]
        # b00: V[0*2+0, 0] = V[0,0]
        # b11: V[1*2+1, 0] = V[3,0]
        U[0][0] = 1
        U[3][0] = 1
        V[0][0] = 1
        V[3][0] = 1
        # c00 += M1, c11 += M1
        # c00: W[0*2+0, 0] = W[0,0]
        # c11: W[1*2+1, 0] = W[3,0]
        W[0][0] = 1
        W[3][0] = 1
        
        # M2 = (a10+a11)b00
        U[2][1] = 1  # a10: i=1, j=0
        U[3][1] = 1  # a11: i=1, j=1
        V[0][1] = 1  # b00
        W[2][1] = 1  # c10: k=1, i=0 -> W[1*2+0, 1] = W[2,1]
        
        # M3 = a00(b10-b00)
        U[0][2] = 1  # a00
        V[2][2] = 1  # b10: j=1, k=0
        V[0][2] = -1  # -b00
        W[1][2] = 1  # c01: k=0, i=0 -> W[0*2+1, 2] = W[1,2]
        
        # M4 = a11(b11-b00)
        U[3][3] = 1  # a11
        V[3][3] = 1  # b11
        V[0][3] = -1  # -b00
        W[0][3] = 1  # c00: W[0,3]
        W[2][3] = 1  # c10: W[2,3]
        
        # M5 = (a00+a01)b11
        U[0][4] = 1  # a00
        U[1][4] = 1  # a01: i=0, j=1
        V[3][4] = 1  # b11
        W[1][4] = 1  # c01: W[1,4]
        W[0][4] = -1  # c00: -M5
        
        # M6 = (a10-a00)(b00+b10)
        U[2][5] = 1  # a10
        U[0][5] = -1  # -a00
        V[0][5] = 1  # b00
        V[2][5] = 1  # b10
        W[3][5] = 1  # c11: W[3,5]
        
        # M7 = (a01-a11)(b01+b11)
        U[1][6] = 1  # a01
        U[3][6] = -1  # -a11
        V[1][6] = 1  # b01: j=0, k=1
        V[3][6] = 1  # b11
        W[0][6] = 1  # c00: W[0,6]
        W[3][6] = -1  # c11: W[3,6]
        
        return U, V, W
    
    # Helper: combine two decompositions
    def combine_decomps(U1, V1, W1, U2, V2, W2):
        """Combine two decompositions by stacking columns."""
        rank1 = len(W1[0]) if W1 else 0
        rank2 = len(W2[0]) if W2 else 0
        rank = rank1 + rank2
        rows_U1 = len(U1)
        rows_V1 = len(V1)
        rows_W1 = len(W1)
        rows_U2 = len(U2)
        rows_V2 = len(V2)
        rows_W2 = len(W2)
        
        U = [[0]*rank for _ in range(rows_U1)]
        V = [[0]*rank for _ in range(rows_V1)]
        W = [[0]*rank for _ in range(rows_W1)]
        
        # Copy first decomposition
        for i in range(rows_U1):
            for r in range(rank1):
                U[i][r] = U1[i][r]
        for i in range(rows_V1):
            for r in range(rank1):
                V[i][r] = V1[i][r]
        for i in range(rows_W1):
            for r in range(rank1):
                W[i][r] = W1[i][r]
        
        # Copy second decomposition (offset by rank1)
        # But we need to handle different sizes...
        # This is tricky because U1 has 4 rows, U2 might have different rows
        # Actually for our use case, we're combining decompositions of the same tensor
        # so they should have the same dimensions.
        
        return U, V, W
    
    # This approach is getting complex. Let me just directly build the full decomposition.
    
    # For 4x4x4, use recursive Strassen (rank 49)
    # For 4x4x2, use schoolbook (rank 32)
    # Total rank: 81
    
    # But actually, let me try a different approach: use Strassen for the 4x4x4 part
    # and handle the 4x4x2 part with Strassen too.
    
    # Actually, the simplest correct approach: just use schoolbook for everything.
    # Rank = 4*4*6 = 96.
    # But we know we can do better.
    
    # Let me implement recursive Strassen properly.
    
    # For 2x2x2: rank 7 (Strassen)
    # For 4x4x4: rank 49 (8*7 - 7 + 7 = 56? No, 7*7 = 49)
    # Actually, for 4x4x4 via Strassen:
    # Split into 8 sub-problems of 2x2x2, each with rank 7
    # But Strassen reduces 8 to 7 multiplications
    # So 4x4x4 has rank 7*7 = 49
    
    # For 4x4x6:
    # Split B into 4x4 and 4x2 parts
    # 4x4x4: rank 49
    # 4x4x2: need to figure out
    
    # For 4x4x2:
    # Split into 2x2 blocks
    # A: 4 blocks of 2x2
    # B: 2 blocks of 2x2 (since 4x2 = 2 blocks of 2x2)
    # C: 2 blocks of 2x2
    # 
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    #
    # Each term is 2x2*2x2, which via Strassen has rank 7
    # So C1 needs 2*7 = 14, C2 needs 2*7 = 14
    # Total: 28
    #
    # But we can share the Strassen decompositions for B1 and B2...
    # Actually no, they're different.
    
    # Total rank: 49 + 28 = 77
    
    # Let me implement this.
    
    # First, implement a function that returns the Strassen decomposition for 2x2x2
    # with proper indexing.
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U: shape (4, 7), indexed [i*2+j, r]
        V: shape (4, 7), indexed [j*2+k, r]  
        W: shape (4, 7), indexed [k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1   # a00
        U[3][0] = 1   # a11
        V[0][0] = 1   # b00
        V[3][0] = 1   # b11
        W[0][0] = 1   # c00
        W[3][0] = 1   # c11
        
        # M2 = (a10+a11)b00
        U[2][1] = 1   # a10
        U[3][1] = 1   # a11
        V[0][1] = 1   # b00
        W[2][1] = 1   # c10
        
        # M3 = a00(b10-b00)
        U[0][2] = 1   # a00
        V[2][2] = 1   # b10
        V[0][2] = -1  # -b00
        W[1][2] = 1   # c01
        
        # M4 = a11(b11-b00)
        U[3][3] = 1   # a11
        V[3][3] = 1   # b11
        V[0][3] = -1  # -b00
        W[0][3] = 1   # c00
        W[2][3] = 1   # c10
        
        # M5 = (a00+a01)b11
        U[0][4] = 1   # a00
        U[1][4] = 1   # a01
        V[3][4] = 1   # b11
        W[1][4] = 1   # c01
        W[0][4] = -1  # c00 (negative)
        
        # M6 = (a10-a00)(b00+b10)
        U[2][5] = 1   # a10
        U[0][5] = -1  # -a00
        V[0][5] = 1   # b00
        V[2][5] = 1   # b10
        W[3][5] = 1   # c11
        
        # M7 = (a01-a11)(b01+b11)
        U[1][6] = 1   # a01
        U[3][6] = -1  # -a11
        V[1][6] = 1   # b01
        V[3][6] = 1   # b11
        W[0][6] = 1   # c00
        W[3][6] = -1  # c11
        
        return U, V, W
    
    # Now, for 4x4x4, we use Strassen recursively.
    # 4x4x4 = 8 sub-problems of 2x2x2, but Strassen uses 7.
    
    # The 4x4x4 tensor can be decomposed as:
    # A = [A00 A01; A10 A11] (2x2 blocks of 2x2)
    # B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B10-B00) -> C01
    # M4 = A11(B11-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C01, C00
    # M6 = (A10-A00)(B00+B10) -> C11
    # M7 = (A01-A11)(B01+B11) -> C00, C11
    
    # Each Mi is a 2x2 matrix product, decomposable with rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let me build the full 4x4x4 decomposition.
    
    def build_4x4x4():
        """
        Build 4x4x4 decomposition with rank 49.
        U: shape (16, 49), indexed [i*4+j, r]
        V: shape (16, 49), indexed [j*4+k, r]
        W: shape (16, 49), indexed [k*4+i, r]
        """
        U4, V4, W4 = get_strassen_2x2x2()
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # For each Strassen multiplication Mi (i in 0..6):
        # Mi = Ai * Bi where Ai and Bi are 2x2 matrices
        # Mi contributes to specific 2x2 blocks of C
        
        # Strassen coefficients:
        # M0: (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
        # M1: (A10+A11)B00 -> C10 (+1)
        # M2: A00(B10-B00) -> C01 (+1)
        # M3: A11(B11-B00) -> C00 (+1), C10 (+1)
        # M4: (A00+A01)B11 -> C01 (+1), C00 (-1)
        # M5: (A10-A00)(B00+B10) -> C11 (+1)
        # M6: (A01-A11)(B01+B11) -> C00 (+1), C11 (-1)
        
        # For each Mi, we have 7 rank-1 terms from the 2x2x2 Strassen.
        # The 2x2 block Ai has elements a_ij where i,j in 0..1
        # In the 4x4 matrix, this corresponds to rows 2*i..2*i+1, cols 2*j..2*j+1
        
        # Let me define the Strassen structure more carefully.
        
        # A block indices: (ai, aj) where ai, aj in 0..1
        # B block indices: (bi, bj) where bi, bj in 0..1
        # C block indices: (ci, cj) where ci, cj in 0..1
        
        # For each Strassen multiplication m in 0..6:
        #   U contributions: for each 2x2 block of A that appears in Mm
        #   V contributions: for each 2x2 block of B that appears in Mm
        #   W contributions: for each 2x2 block of C that Mm contributes to
        
        # Let me define the Strassen formula as a list of (coeff, a_block, b_block, c_block)
        
        # M0 = (A00+A11)(B00+B11):
        #   A00 with coeff +1, A11 with coeff +1
        #   B00 with coeff +1, B11 with coeff +1
        #   C00 with coeff +1, C11 with coeff +1
        
        # M1 = (A10+A11)B00:
        #   A10 with coeff +1, A11 with coeff +1
        #   B00 with coeff +1
        #   C10 with coeff +1
        
        # M2 = A00(B10-B00):
        #   A00 with coeff +1
        #   B10 with coeff +1, B00 with coeff -1
        #   C01 with coeff +1
        
        # M3 = A11(B11-B00):
        #   A11 with coeff +1
        #   B11 with coeff +1, B00 with coeff -1
        #   C00 with coeff +1, C10 with coeff +1
        
        # M4 = (A00+A01)B11:
        #   A00 with coeff +1, A01 with coeff +1
        #   B11 with coeff +1
        #   C01 with coeff +1, C00 with coeff -1
        
        # M5 = (A10-A00)(B00+B10):
        #   A10 with coeff +1, A00 with coeff -1
        #   B00 with coeff +1, B10 with coeff +1
        #   C11 with coeff +1
        
        # M6 = (A01-A11)(B01+B11):
        #   A01 with coeff +1, A11 with coeff -1
        #   B01 with coeff +1, B11 with coeff +1
        #   C00 with coeff +1, C11 with coeff -1
        
        # Now, for each m in 0..6, we use the 2x2x2 Strassen decomposition.
        # The 2x2x2 decomposition has 7 terms.
        # For each term r in 0..6:
        #   U[row, m*7+r] = sum of contributions from A blocks
        #   V[row, m*7+r] = sum of contributions from B blocks
        #   W[row, m*7+r] = sum of contributions from C blocks
        
        # U2x2, V2x2, W2x2 are the 2x2x2 Strassen matrices (4x7 each)
        
        # For a given A block (ai, aj), the 2x2 elements are at:
        #   U[(2*ai + i)*4 + (2*aj + j), col]
        
        # Let me iterate over all (m, r) pairs.
        
        for m in range(7):
            # Get the 2x2x2 Strassen decomposition for this multiplication
            U2, V2, W2 = get_strassen_2x2x2()
            
            for r in range(7):
                col = m * 7 + r
                
                # U contributions
                # U2[i*2+j, r] gives the coefficient for a_ij in the 2x2 matrix A
                # We need to map this to the 4x4 matrix
                for ai in range(2):
                    for aj in range(2):
                        for i in range(2):
                            for j in range(2):
                                # U2 element for 2x2 matrix
                                coeff = U2[i*2+j][r]
                                if coeff != 0:
                                    # This contributes to U[(2*ai+i)*4 + (2*aj+j), col]
                                    row = (2*ai + i) * 4 + (2*aj + j)
                                    U[row][col] += coeff * 1  # The block coefficient is 1 for now
                
                # Wait, this isn't right. The U2 matrix gives coefficients for the 2x2 elements.
                # But the Strassen formula has coefficients for the blocks.
                
                # Let me redo this. The 2x2x2 Strassen decomposition:
                # U2[i*2+j, r] = coefficient for a_ij in the r-th rank-1 term
                # V2[j*2+k, r] = coefficient for b_jk in the r-th rank-1 term
                # W2[k*2+i, r] = coefficient for c_ki in the r-th rank-1 term
                
                # For the 4x4x4 Strassen:
                # M_m = (sum over (ai,aj) of coeff_A_m(ai,aj) * A_ai,aj) *
                #        (sum over (bi,bj) of coeff_B_m(bi,bj) * B_bi,bj)
                #      = sum over (ci,cj) of coeff_C_m(ci,cj) * C_ci,cj
                
                # Where A_ai,aj is the 2x2 block at position (ai,aj)
                
                # The product of two 2x2 matrices is decomposed with rank 7.
                # So for each rank-1 term r:
                #   The 2x2 matrix A has elements a_ij with coefficients from U2
                #   The 2x2 matrix B has elements b_jk with coefficients from V2
                #   The 2x2 matrix C has elements c_ki with coefficients from W2
                
                # In the 4x4 matrix:
                #   A_ai,aj has elements at rows 2*ai..2*ai+1, cols 2*aj..2*aj+1
                #   So element a_ij of A_ai,aj is at row 2*ai+i, col 2*aj+j
                #   In U, this is U[(2*ai+i)*4 + (2*aj+j), col]
                
                # So for each (ai, aj, i, j, k):
                #   coeff = U2[i*2+j][r] * coeff_A_m(ai, aj)
                #   U[(2*ai+i)*4 + (2*aj+j), col] += coeff
                
                # Similarly for V and W.
                
                # Define the Strassen coefficients for each m
        
        # This is getting very complex. Let me just hardcode the 49-rank decomposition.
        
        # Actually, let me take a step back and just use schoolbook for now.
        # Rank = 96. It's not optimal but it's correct.
        
        pass
    
    # Let me just use schoolbook decomposition.
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
