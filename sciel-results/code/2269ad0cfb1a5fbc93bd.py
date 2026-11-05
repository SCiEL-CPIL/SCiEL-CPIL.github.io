"""Matrix multiplication 6x6x6 using Kronecker product of two 3x3 decompositions."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _get_bcl3_decomp():
    """
    Bini-Capovani-Lotti-Romani rank-21 decomposition for 3x3 matrix multiplication.
    
    This is a well-known exact decomposition. The factors U, V, W are 27x21 matrices
    (flattened row-major: index = row * n + col for U and V, col * n + row for W).
    
    U: shape (9, 21) - maps A[i,j] to intermediate
    V: shape (9, 21) - maps B[j,k] to intermediate
    W: shape (9, 21) - maps intermediate to C[k*3+i]
    
    Using the trace convention: W[k*n + i, r] where n=3
    """
    # The Bini-Capovani-Lotti-Romani decomposition for 3x3
    # Each rank-1 component corresponds to a triple (U_r, V_r, W_r)
    # U_r[i,j] V_r[j,k] W_r[k,i] for appropriate i,j,k
    
    # The 21 components are indexed by triples from {0,1,2}x{0,1,2}x{0,1,2}
    # minus the 6 diagonal triples, giving 27 - 6 = 21
    
    # For each non-diagonal triple (a, b, c):
    # U has 1 at position (a, b) if a != b, else 0... 
    # Actually, the standard BCL-R decomposition:
    # For each (i,j,k) with i,j,k in {0,1,2}, i+j+k not in {0,3,6}... 
    
    # Let me use the explicit construction:
    # The decomposition uses the 21 off-diagonal terms.
    # Component r corresponds to a specific pattern.
    
    # Simpler: use the known explicit BCL-R factors.
    # Each of the 21 components is defined by (p, q, r) where p,q,r in {0,1,2}, p+q+r not all equal and not in certain excluded set.
    # Actually, the 21 components correspond to all (i,j,k) in {0,1,2}^3 except (0,0,0), (1,1,1), (2,2,2) and (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0).
    # That gives 27 - 6 - 6 = 15... no that's not 21.
    
    # The BCL-R decomposition has exactly 21 rank-1 terms.
    # Let me use the standard construction based on the formula:
    # For each (s,t,u) in {0,1,2}^3 with s+t+u not in {0, 3, 6}:
    # Actually, I recall the construction involves 21 specific index triples.
    
    # Let me just use the well-known explicit factors.
    # BCL-R 3x3 decomposition, rank 21:
    
    # The indices are (i, j, k) for i,j,k in {0,1,2}
    # Excluded: (0,0,0), (1,1,1), (2,2,2), and the 6 permutations of (0,1,2)
    # Wait, that gives 27 - 3 - 6 = 18. Not 21.
    
    # The actual BCL-R excludes only (0,0,0), (1,1,1), (2,2,2) from the 27 diagonal-like terms? No.
    
    # Let me use a different well-known approach. The Strassen-like 3x3 decomposition of rank 21:
    # We can use the known rank-21 decomposition from Bini et al.
    
    # Explicit BCL-R decomposition:
    # The 21 components are indexed by triples (a,b,c) where a,b,c in {0,1,2}
    # and a != b or b != c or c != a (i.e., not all equal), PLUS 6 more terms.
    # This is getting complicated. Let me use the actual published factors.
    
    # I'll use the construction from the literature where each component r
    # is associated with an index triple, and the factors are:
    # U_r[i, j] = 1 if (i,j) matches a specific pattern for component r
    # V_r[j, k] = 1 if (j,k) matches a specific pattern
    # W_r[k, i] = 1 if (k,i) matches a specific pattern
    
    # Actually, let me just hardcode the known BCL-R rank-21 decomposition.
    # The decomposition can be expressed as follows:
    
    # For the 3x3 case, we use 21 "gates" (rank-1 tensors).
    # Gate r is defined by indices (i_r, j_r, k_r).
    
    # Standard BCL-R: the 21 gates are:
    # (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
    # (1,0,0), (1,0,1), (1,0,2), (1,1,0), (1,1,1), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
    # (2,0,0), (2,0,1), (2,0,2)
    # Wait, that's 21 but let me verify this is correct.
    
    # Actually, I need to be more careful. Let me use the explicit numerical BCL-R factors.
    # The Bini-Capovani-Lotti-Romani decomposition for 3x3 has rank 21.
    
    # One standard representation:
    # U, V, W are 9x21 matrices (rows indexed by (row,col) in row-major order)
    
    # I'll use the well-known explicit form. Let me define it via the gate structure.
    # For BCL-R, each component r corresponds to a choice of:
    # - For U: which (i,j) entry to pick from A
    # - For V: which (j',k) entry to pick from B  
    # - For W: which (k',i') entry to add to C
    
    # The key insight: for component r with "shift" parameters, 
    # U picks A[i, j], V picks B[j+delta1, k], W adds to C[k+delta2, i]
    
    # Let me just use the verified Strassen-block approach which is simpler and known to work.
    
    # Actually, for 6x6x6, I'll use the known rank-189 construction:
    # Block Strassen: partition 6x6 into 2x2 blocks of size 3x3.
    # Use Strassen's 7-multiplication algorithm at the block level (2x2 blocks).
    # Within each block, use schoolbook multiplication (27 multiplications for 3x3).
    # Total rank: 7 * 27 = 189.
    
    # This is the construction that achieved score 1.143 in previous observations.
    
    return None  # We'll implement the block Strassen directly

def construct():
    """
    Construct rank-189 decomposition for 6x6 matrix multiplication
    using block Strassen algorithm with 3x3 schoolbook sub-blocks.
    
    The 6x6 matrices are partitioned into 2x2 blocks of 3x3 matrices.
    Strassen's algorithm computes the 4 output blocks using 7 block multiplications.
    Each block multiplication of 3x3 matrices is done via schoolbook (27 scalar mults).
    Total rank: 7 * 27 = 189.
    """
    import time
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 3
    num_blocks = 2  # 2x2 blocks
    
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen's 7 multiplications for 2x2 block matrices:
    # Given A blocks A00, A01, A10, A11 and B blocks B00, B01, B10, B11:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # 
    # Output blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_s is a 3x3 matrix product, requiring 27 rank-1 terms.
    # Total: 7 * 27 = 189 rank-1 terms.
    
    # Strassen factor definitions (for the 2x2 block level):
    # For each of the 7 Strassen products, we need:
    # - U_block[i_block, s]: coefficient for A block i_block in product s
    # - V_block[j_block, s]: coefficient for B block j_block in product s  
    # - W_block[k_block, s]: coefficient for C block k_block in product s
    
    # Product 1: (A00 + A11) * (B00 + B11)
    # U_block: block 0 -> 1, block 1 -> 1
    # V_block: block 0 -> 1, block 1 -> 1
    # W_block: contributes to C00 and C11
    strassen_factors = [
        # (U_block_coeffs, V_block_coeffs, W_block_coeffs)
        # U_block_coeffs[i] = coefficient of A block i in this product
        # V_block_coeffs[j] = coefficient of B block j in this product
        # W_block_coeffs[k] = coefficient added to C block k
        
        # Product 1: (A00+A11)*(B00+B11)
        # Contributes +1 to C00, +1 to C11
        ([1, 1], [1, 1], [1, 1]),  # U: blocks 0,1; V: blocks 0,1; W: blocks 0,1 (C00,C11)
        
        # Product 2: (A10+A11)*B00
        # Contributes +1 to C10
        ([0, 1], [1, 0], [0, 1]),  # Wait, this needs careful indexing
    ]
    
    # Let me redo this more carefully.
    # The 2x2 block decomposition:
    # A = [[A00, A01], [A10, A11]] where each is 3x3
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For each Strassen product Ms, the factorization at block level is:
    # Ms = sum over A blocks and B blocks with appropriate coefficients
    #
    # The rank-1 decomposition at the block level for each Ms:
    # Ms = L_s(A) * R_s(B) where L_s is a linear combination of A blocks
    # and R_s is a linear combination of B blocks.
    #
    # For the full 6x6 rank-1 decomposition, each (block-level) product Ms
    # is itself a 3x3 matrix multiplication, which expands to 27 rank-1 terms.
    
    # Let me define the block-level coefficients for each Strassen product:
    # strassen[s] = (a_coeffs, b_coeffs, c_coeffs)
    # where a_coeffs[i] = coeff of A block i in L_s
    # b_coeffs[j] = coeff of B block j in R_s
    # c_coeffs[k] = coeff of C block k that Ms contributes to
    
    # But wait - the output distribution also matters. Each product Ms contributes
    # to multiple C blocks with specific coefficients. So W_block needs to encode
    # both which C block AND what coefficient.
    
    # Let me structure it differently. For each of the 7 Strassen products,
    # and for each output C block it contributes to, we have a (possibly different)
    # contribution. But actually, in the rank-1 decomposition, each rank-1 term
    # contributes to exactly one (C_block, A_entry, B_entry) triple with unit weight
    # (or we can have fractional/integer weights in U, V, W).
    
    # Let me think about this more carefully using the trace convention.
    # C[i,k] = sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k*n+i, r]
    # where n=m=p=6.
    
    # In the block Strassen approach:
    # Global row i = 3*I + i_local where I in {0,1} (block row), i_local in {0,1,2}
    # Global col j = 3*J + j_local where J in {0,1} (block col), j_local in {0,1,2}
    # For B: row j = 3*J + j_local, col k = 3*K + k_local
    # For C: row i = 3*I + i_local, col k = 3*K + k_local
    # W index: k*6+i = (3*K+k_local)*6 + (3*I+i_local)
    
    # For Strassen product s, A coefficient for block I is u_s[I],
    # B coefficient for block J is v_s[J], and C block K gets coefficient w_s[K].
    
    # The 7 Strassen products with their coefficients:
    strassen_U = [
        [1, 1],  # s=0 (M1): A00 + A11
        [0, 1],  # s=1 (M2): A10 + A11  -> blocks: A10 is block row 1, A11 is block row 1
        [1, 0],  # s=2 (M3): A00
        [0, 1],  # s=3 (M4): A11
        [1, 1],  # s=4 (M5): A00 + A01
        [1, -1], # s=5 (M6): A10 - A00
        [0, 1],  # s=6 (M7): A01 - A11
    ]
    # Wait, I need to be more careful. The A blocks are:
    # A00: block row 0, block col 0
    # A01: block row 0, block col 1
    # A10: block row 1, block col 0
    # A11: block row 1, block col 1
    
    # In the matrix multiplication C=AB, the relevant indices are:
    # A is 6x6, so A[row, col] where row = 3*I + i, col = 3*J + j
    # The "block" of A used is determined by (I, J).
    
    # For Strassen, the A factor for product s involves blocks with certain (I,J) indices.
    # But in the rank-1 decomposition, U[i,j,r] selects which (row,col) of A to use.
    # The block structure means that for a given Strassen product, the A factor
    # is a linear combination of specific A blocks.
    
    # Let me redefine. For each Strassen product s (0..6):
    # The A-side is a linear combination of A blocks: sum_{I,J} a_s[I,J] * A[I,J]
    # The B-side is a linear combination of B blocks: sum_{J',K} b_s[J',K] * B[J',K]
    # The result contributes to C blocks: sum_{K,I} c_s[K,I] * C[K,I]
    
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    #   A blocks: A00 (a=1), A11 (a=1)
    #   B blocks: B00 (b=1), B11 (b=1)
    #   C blocks: C00 (c=1), C11 (c=1) [from C00=M1+M4-M5+M7, C11=M1-M2+M3+M6]
    
    # M2 = (A10+A11)*B00
    #   A blocks: A10 (a=1), A11 (a=1)
    #   B blocks: B00 (b=1)
    #   C blocks: C10 (c=1)
    
    # M3 = A00*(B01-B11)
    #   A blocks: A00 (a=1)
    #   B blocks: B01 (b=1), B11 (b=-1)
    #   C blocks: C01 (c=1), C11 (c=1)
    
    # M4 = A11*(B10-B00)
    #   A blocks: A11 (a=1)
    #   B blocks: B10 (b=1), B00 (b=-1)
    #   C blocks: C00 (c=1), C10 (c=1)
    
    # M5 = (A00+A01)*B11
    #   A blocks: A00 (a=1), A01 (a=1)
    #   B blocks: B11 (b=1)
    #   C blocks: C00 (c=-1), C01 (c=1)
    
    # M6 = (A10-A00)*(B00+B01)
    #   A blocks: A10 (a=1), A00 (a=-1)
    #   B blocks: B00 (b=1), B01 (b=1)
    #   C blocks: C11 (c=1)
    
    # M7 = (A01-A11)*(B10+B11)
    #   A blocks: A01 (a=1), A11 (a=-1)
    #   B blocks: B10 (b=1), B11 (b=1)
    #   C blocks: C00 (c=1), C11 (c=1)
    
    # Now, for the full rank-1 decomposition:
    # Each Strassen product s corresponds to a 3x3 matrix multiplication.
    # The 3x3 multiplication has 27 rank-1 terms (schoolbook).
    # For term (i,j,k) in the 3x3 schoolbook:
    #   U_local[i,j] = 1, V_local[j,k] = 1, W_local[k,i] = 1
    #   where i,j,k are local indices 0,1,2 within the 3x3 block.
    
    # The global indices:
    # For A: global row = 3*I_A + i, global col = 3*J_A + j
    # For B: global row = 3*J_B + j, global col = 3*K_B + k
    # For C: global row = 3*I_C + i, global col = 3*K_C + k
    # W index = global_col * 6 + global_row = (3*K_C+k)*6 + (3*I_C+i)
    
    # The constraint: J_A = J_B (the column block of A must match the row block of B)
    # In Strassen, the A blocks and B blocks are coupled through the middle index.
    
    # For each Strassen product s, the A blocks are specific (I_A, J_A) pairs,
    # and B blocks are specific (J_B, K_B) pairs. The constraint is J_A = J_B.
    
    # For product s, let's define:
    # a_blocks_s = list of (I_A, J_A, coeff_a)
    # b_blocks_s = list of (J_B, K_B, coeff_b)  
    # c_blocks_s = list of (I_C, K_C, coeff_c)
    
    # The rank-1 terms for product s are all valid combinations:
    # For each (I_A, J_A, coeff_a) in a_blocks_s:
    #   For each (J_B, K_B, coeff_b) in b_blocks_s where J_A == J_B:
    #     For each (I_C, K_C, coeff_c) in c_blocks_s:
    #       For each (i, j, k) in {0,1,2}^3:
    #         rank-1 term with:
    #           U[3*I_A+i, 3*J_A+j] = coeff_a
    #           V[3*J_B+j, 3*K_B+k] = coeff_b
    #           W[(3*K_C+k)*6 + (3*I_C+i)] = coeff_c
    
    # Let me define the block structures:
    block_structures = [
        # s=0 (M1):
        {
            'a': [(0,0,1), (1,1,1)],  # (I_A, J_A, coeff)
            'b': [(0,0,1), (1,1,1)],  # (J_B, K_B, coeff)
            'c': [(0,0,1), (1,1,1)],  # (I_C, K_C, coeff)
        },
        # s=1 (M2):
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1)],
        },
        # s=2 (M3):
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1), (1,1,1)],
        },
        # s=3 (M4):
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)],
        },
        # s=4 (M5):
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)],
        },
        # s=5 (M6):
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)],
        },
        # s=6 (M7):
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)],
        },
    ]
    
    # Now build the full decomposition.
    # For each Strassen product s, for each valid (a_block, b_block) pair with matching middle index,
    # for each c_block, and for each local (i,j,k), add one rank-1 term.
    
    # Each such combination is one rank-1 term in the overall decomposition.
    # The U, V, W entries for that term are:
    #   U[3*I_A + i][r] = coeff_a
    #   V[3*J_B + j][r] = coeff_b
    #   W[(3*K_C + k)*6 + (3*I_C + i)][r] = coeff_c
    
    # Let's count how many terms this generates:
    # s=0: a has 2 blocks, b has 2 blocks. Matching J: (0,0) with (0,0) -> match; (1,1) with (0,0) -> no; (0,0) with (1,1) -> no; (1,1) with (1,1) -> match. So 2 matching pairs. c has 2 blocks. 2*2*27 = 108.
    # Hmm, that's already too many. Total would exceed 189.
    
    # I think the issue is that the C blocks should not multiply independently with each A-B pair.
    # The output C block is determined by the Strassen formula, not by free choice.
    
    # Let me reconsider. In the Strassen algorithm at the block level:
    # M_s is a single 3x3 matrix. It contributes to multiple C blocks with specific coefficients.
    # So for each Strassen product s and each local (i,j,k) in the 3x3 multiplication,
    # we have contributions to each C block that M_s feeds into.
    
    # But in a rank-1 decomposition, each rank-1 term contributes to exactly one C entry.
    # So we need to "split" the contribution of M_s to each C block into separate rank-1 terms.
    
    # Alternative approach: For each Strassen product s:
    # The product M_s = L_s(A) * R_s(B) is a 3x3 matrix.
    # M_s contributes to C blocks with coefficients. For a specific C block K_C,
    # the contribution is coeff_c * M_s.
    # So for each (s, c_block_entry, i, j, k):
    #   We create a rank-1 term where:
    #     U picks from A blocks according to L_s
    #     V picks from B blocks according to R_s
    #     W adds to the specific C block entry
    
    # But L_s is a sum over multiple A blocks, so U needs to encode that sum.
    # In a single rank-1 term, U[i,j,r] can have different values for different (i,j).
    
    # Let me construct it differently. For each Strassen product s:
    # L_s(A) = sum_{(I_A,J_A)} a_s[I_A,J_A] * A[I_A,J_A]
    # R_s(B) = sum_{(J_B,K_B)} b_s[J_B,K_B] * B[J_B,K_B]
    # M_s = L_s(A) * R_s(B) (matrix product of the aggregated 3x3 blocks)
    # 
    # M_s[i,k] = sum_{j=0}^2 sum_{(I_A,J_A)} sum_{(J_B,K_B)} a_s[I_A,J_A] * b_s[J_B,K_B] * A[3*I_A+i, 3*J_A+j] * B[3*J_B+j, 3*K_B+k]
    #   with constraint J_A = J_B (for the inner product to make sense at block level)
    #
    # Wait, this isn't right either. The block matrix product is:
    # (L_s(A))_{i,k} = sum_j sum_{I_A,J_A} a_s[I_A,J_A] * A_{3*I_A+i, 3*J_A+j} * delta_{J_A, J_B} ... 
    
    # I'm overcomplicating this. Let me use a cleaner formulation.
    
    # The key insight for block Strassen: 
    # We treat the 2x2 block structure algebraically. The Strassen formula gives
    # C blocks in terms of 7 block products. Each block product is a standard 3x3 matmul.
    
    # For the rank-1 decomposition, I'll enumerate:
    # For each Strassen product s (0..6):
    #   For each A-block index (I_A, J_A) with a_s[I_A,J_A] != 0:
    #     For each B-block index (J_B, K_B) with b_s[J_B,K_B] != 0 and J_A == J_B:
    #       For each local (i, j, k) in {0,1,2}^3:
    #         This defines intermediate values. But we also need to distribute to C blocks.
    
    # I think the cleanest way is:
    # For each (s, I_A, J_A, J_B, K_B, i, j, k) with J_A == J_B and a_s[I_A,J_A] != 0 and b_s[J_B,K_B] != 0:
    #   This combination defines a "partial product" that should be added to each C block
    #   with coefficient c_s[I_C, K_C].
    
    # For each such combination AND each C block (I_C, K_C) with c_s[I_C,K_C] != 0:
    #   Create a rank-1 term:
    #     U[3*I_A + i, 3*J_A + j, r] = a_s[I_A, J_A]
    #     V[3*J_B + j, 3*K_B + k, r] = b_s[J_B, K_B] 
    #     W[(3*K_C+k)*6 + (3*I_C+i), r] = c_s[I_C, K_C]
    
    # But wait, this would overcount because the same U and V entries would be used
    # across multiple rank-1 terms. That's fine - they just get added.
    
    # Let me count the total number of rank-1 terms:
    # For each s, the number is: (# A blocks in s) * (# B blocks in s with matching J) * (# C blocks in s) * 27
    # 
    # s=0: A blocks: 2, B blocks: 2, matching pairs: 2 (as computed), C blocks: 2. 2*2*27 = 108
    # That's already more than 189/7 = 27 per product on average.
    
    # The issue: I'm not correctly modeling the block-level structure. In Strassen,
    # M_s is a SINGLE 3x3 matrix product, not multiple independent ones. The A blocks
    # are combined into a single 3x3 matrix, and similarly for B.
    
    # So for product s:
    # L_s = sum a_s[I_A,J_A] * A[I_A,J_A] is a single 3x3 matrix.
    # R_s = sum b_s[J_B,K_B] * B[J_B,K_B] is a single 3x3 matrix.
    # M_s = L_s * R_s is the 3x3 product.
    
    # The schoolbook decomposition of M_s = L_s * R_s has 27 rank-1 terms:
    # M_s[i,k] = sum_j L_s[i,j] * R_s[j,k]
    # L_s[i,j] = sum_{I_A,J_A} a_s[I_A,J_A] * A[3*I_A+i, 3*J_A+j]
    # R_s[j,k] = sum_{J_B,K_B} b_s[J_B,K_B] * B[3*J_B+j, 3*K_B+k]
    
    # So the 27 rank-1 terms for product s are, for each (i,j,k):
    # U entry: for each (I_A,J_A), U[3*I_A+i, 3*J_A+j, r] += a_s[I_A,J_A]
    # V entry: for each (J_B,K_B), V[3*J_B+j, 3*K_B+k, r] += b_s[J_B,K_B]
    # W entry: for each (I_C,K_C), W[(3*K_C+k)*6+(3*I_C+i), r] += c_s[I_C,K_C]
    
    # This uses ONE rank-1 term per (s, i, j, k) = 7 * 27 = 189 total. 
    
    # Let's implement this:
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    a_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},              # M3
        {(1,1): 1},              # M4
        {(0,0): 1, (0,1): 1},   # M5
        {(1,0): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): -1},  # M7
    ]
    b_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},              # M2
        {(0,1): 1, (1,1): -1},  # M3
        {(1,0): 1, (0,0): -1},  # M4
        {(1,1): 1},              # M5
        {(0,0): 1, (0,1): 1},   # M6
        {(1,0): 1, (1,1): 1},   # M7
    ]
    c_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1 -> C00, C11
        {(1,0): 1},              # M2 -> C10
        {(0,1): 1, (1,1): 1},   # M3 -> C01, C11
        {(0,0): 1, (1,0): 1},   # M4 -> C00, C10
        {(0,0): -1, (0,1): 1},  # M5 -> C00, C01
        {(1,1): 1},              # M6 -> C11
        {(0,0): 1, (1,1): 1},   # M7 -> C00, C11
    ]
    
    col = 0
    for s in range(7):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if col >= rank:
                        break
                    # Set U entries
                    for (I_A, J_A), a_val in a_coeffs[s].items():
                        U[3*I_A + i][3*J_A + j] = a_val if U[3*I_A + i][3*J_A + j] == 0 else U[3*I_A + i][3*J_A + j] + a_val
                    # Hmm, U is indexed by [row, col] not [I_A,i, J_A,j]. Let me fix.
    
    # I made an indexing error. U is shape (36, 189), where row index = 3*I_A + i (global row of A)
    # But U[row, r] where row encodes both the block and local index.
    # Actually in the problem, U has shape (n*m, rank) = (36, rank).
    # U[i, j, r] in the tensor sense maps to U[i*m+j, r] = U[6i+j, r].
    # For A[i,j], the row index in U is i*m + j = 6*i + j.
    # With block structure: i = 3*I_A + i_local, j = 3*J_A + j_local.
    # So U row index = 6*(3*I_A + i_local) + (3*J_A + j_local) = 18*I_A + 6*i_local + 3*J_A + j_local.
    
    # Similarly, V has shape (m*p, rank) = (36, rank).
    # V[j, k, r] -> V[j*p + k, r] = V[6*j + k, r].
    # j = 3*J_B + j_local, k = 3*K_B + k_local.
    # V row index = 6*(3*J_B + j_local) + (3*K_B + k_local) = 18*J_B + 6*j_local + 3*K_B + k_local.
    
    # W has shape (p*n, rank) = (36, rank).
    # W[k, i, r] -> W[k*n + i, r] = W[6*k + i, r].
    # k = 3*K_C + k_local, i = 3*I_C + i_local.
    # W row index = 6*(3*K_C + k_local) + (3*I_C + i_local) = 18*K_C + 6*k_local + 3*I_C + i_local.
    
    # Let me redo:
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        for i_loc in range(3):
            for j_loc in range(3):
                for k_loc in range(3):
                    if col >= rank:
                        break
                    # U entries: for each (I_A, J_A, a_val) in a_coeffs[s]
                    for (I_A, J_A), a_val in a_coeffs[s].items():
                        u_idx = 6 * (3 * I_A + i_loc) + (3 * J_A + j_loc)
                        U[u_idx][col] = a_val
                    
                    # V entries: for each (J_B, K_B, b_val) in b_coeffs[s]
                    for (J_B, K_B), b_val in b_coeffs[s].items():
                        v_idx = 6 * (3 * J_B + j_loc) + (3 * K_B + k_loc)
                        V[v_idx][col] = b_val
                    
                    # W entries: for each (I_C, K_C, c_val) in c_coeffs[s]
                    for (I_C, K_C), c_val in c_coeffs[s].items():
                        w_idx = 6 * (3 * K_C + k_loc) + (3 * I_C + i_loc)
                        W[w_idx][col] = c_val
                    
                    col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
    
    # Verify we used exactly 189 columns
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to the required format
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
