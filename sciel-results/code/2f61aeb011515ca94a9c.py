"""Tensor-train / hierarchical compression for 4x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Strategy: Split k-index (0..6) into two groups:
    #   Group A: k in {0,1,2,3} (4 values) -> use Strassen for 4x4x4, rank 7
    #   Group B: k in {4,5,6} (3 values)   -> use schoolbook for 4x4x3, rank 12
    # Total rank = 7 + 12 = 19
    
    # Strassen decomposition for 2x2x2 (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4x4, split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2.
    # Block indices: I,J in {0,1}, sub-indices i',j' in {0,1}
    # i = 2*I + i', j = 2*J + j'
    #
    # Each Strassen term M_k is a 2x2 matrix product, so each M_k has rank 7
    # (using Strassen on the 2x2x2 sub-problem).
    # Total for 4x4x4: 7 terms * 7 rank each = 49? No wait.
    #
    # Actually, the 4x4x4 multiplication tensor can be decomposed using
    # Strassen's algorithm at the block level. The 7 Strassen multiplications
    # are each 2x2 matrix multiplications. If we use schoolbook for each 2x2
    # multiplication (rank 4 each), total = 7*4 = 28.
    # If we use Strassen recursively for each 2x2 (rank 7 each), total = 7*7 = 49.
    # Schoolbook for 4x4x4 is rank 16. So 28 > 16, not helpful.
    #
    # Let me just use schoolbook for 4x4x4 (rank 16) + schoolbook for 4x4x3 (rank 12)
    # = total rank 28. That's better than 112.
    
    # Actually, let me think about this differently.
    # Schoolbook for 4x4x7: rank = 4*4*7 = 112
    # We want to do better.
    
    # Approach: Use the fact that 4x4x4 can be done in rank 7 (Strassen) 
    # and 4x4x3 can be done in rank 12 (schoolbook).
    # But the Strassen decomposition for 4x4x4 gives rank 7, not 49.
    # Wait, Strassen gives rank 7 for 2x2x2, not 4x4x4.
    # For 4x4x4, the best known is much higher.
    
    # Let me just do a simple block decomposition:
    # Split the 7 columns of B into a 4-column part and a 3-column part.
    # 4x4x4: schoolbook rank 16
    # 4x4x3: schoolbook rank 12
    # Total: 28
    
    # This is a solid improvement from 112 to 28.
    
    U_rows = n * m  # 16
    V_rows = m * p  # 28
    W_rows = p * n  # 28
    
    # Build schoolbook factors for 4x4x4 part (k=0..3)
    rank_444 = n * m * 4  # 64... no, 4*4*4=64? No, n*m*4 = 16*4=64? 
    # Wait, schoolbook rank for n x m x p is n*m*p.
    # For 4x4x4: 4*4*4 = 64? No, that's wrong.
    # Schoolbook: for each (i,j,k) triple, one rank-1 term.
    # Number of triples = n*m*p = 4*4*4 = 64? No!
    # The multiplication C = A*B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Rank-1 term for (i,j,k): U[i,j,term]=1, V[j,k,term]=1, W[k,i,term]=1
    # Number of terms = n*m*p = 4*4*4 = 64? No, that's n*m*p but...
    # Actually yes, schoolbook rank = n*m*p.
    # For 4x4x7: 4*4*7 = 112. Correct.
    # For 4x4x4: 4*4*4 = 64. 
    # For 4x4x3: 4*4*3 = 48.
    # Total: 64+48 = 112. Same as before!
    
    # So splitting doesn't help with schoolbook. I need actual Strassen-like savings.
    
    # Let me implement Strassen for the 4x4x4 part properly.
    # Strassen reduces 2x2x2 from rank 8 to rank 7.
    # For 4x4x4, I can view it as 2x2 blocks of 2x2 matrices.
    # The 7 Strassen products are each 2x2x2 multiplications.
    # If I use schoolbook (rank 4) for each 2x2x2: total = 7*4 = 28.
    # If I use Strassen (rank 7) for each 2x2x2: total = 7*7 = 49.
    # Schoolbook for 4x4x4: rank 64.
    # So 28 < 64, Strassen at block level with schoolbook sub-multiplications helps!
    
    # Let me construct the Strassen-based decomposition for 4x4x4.
    
    # Strassen terms for 2x2x2:
    # Each term t in {0..6} has coefficients for A blocks and B blocks,
    # and the result contributes to C blocks with certain signs.
    
    # A blocks: A[0,0], A[0,1], A[1,0], A[1,1] indexed by (I,J) in {0,1}^2
    # B blocks: B[0,0], B[0,1], B[1,0], B[1,1]
    # C blocks: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each M_t, it's a product of a 2x2 linear combo of A blocks and a 2x2
    # linear combo of B blocks. The product is a 2x2 matrix.
    # To compute this 2x2 product, we use schoolbook: 4 rank-1 terms.
    # For each of the 4 sub-multiplications (i',j',k' in {0,1}^3),
    # we get a rank-1 term in the full 4x4x4 tensor.
    
    # Total: 7 Strassen terms * 4 sub-terms = 28 rank-1 terms for 4x4x4.
    
    # Now for each Strassen term t and sub-term (i',j',k'),
    # I need to determine which A-rows, B-columns, and C-entries are involved.
    
    # A block (I,J) covers rows i = 2*I + i', cols j = 2*J + j' of A.
    # B block (J,K) covers rows j = 2*J + j', cols k = 2*K + k' of B.
    # C block (I,K) covers rows i = 2*I + i', cols k = 2*K + k' of C.
    
    # Strassen term t specifies:
    # - Which A blocks are combined and with what signs
    # - Which B blocks are combined and with what signs
    # - Which C blocks receive the result and with what signs
    
    # Let me define the Strassen structure:
    # term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (I,J) -> sign
    # b_coeffs: dict from (J,K) -> sign  
    # c_coeffs: dict from (I,K) -> sign
    
    strassen_terms = [
        # M0 = (A00+A11)(B00+B11)
        (
            {(0,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1, (1,1): 1}    # C coeffs (C00 +=, C11 +=)
        ),
        # M1 = (A10+A11)B00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1}              # C10 +=
        ),
        # M2 = A00(B10-B11)
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1}              # C01 +=
        ),
        # M3 = A11(B01-B00)
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}   # C00 +=, C10 +=
        ),
        # M4 = (A00+A01)B11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}  # C00 -=, C01 +=
        ),
        # M5 = (A10-A00)(B00+B01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}              # C11 +=
        ),
        # M6 = (A01-A11)(B10+B11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1}  # C00 +=, C11 -=
        ),
    ]
    
    # Now build the factors.
    # For the 4x4x4 part, rank = 28.
    # For the 4x4x3 part, rank = 48 (schoolbook).
    # Total rank = 76.
    
    # Actually, let me just use schoolbook for everything to start,
    # then optimize. But 112 is the baseline. Let me try the Strassen approach.
    
    total_rank = 28 + 48  # 76
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 76
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 76
    
    col = 0
    # Part 1: 4x4x4 via Strassen (28 terms)
    for t_idx in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_terms[t_idx]
        # For each sub-multiplication (i', j', k') in {0,1}^3
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    # A blocks involved: those (I,J) in a_coeffs
                    # B blocks involved: those (J,K) in b_coeffs where J=jp? 
                    # No, J is the middle index. Let me be more careful.
                    #
                    # The product M = (sum_{I,J} a_IJ * A[I,J]) * (sum_{J,K} b_JK * B[J,K])
                    # = sum_{I,J,K} a_IJ * b_JK * A[I,J] * B[J,K]
                    # This is a 2x2 matrix where entry (i',k') is:
                    # sum_J (sum_I a_IJ * A[2I+i', 2J+jp]) * (sum_K b_JK * B[2J+jp, 2K+kp])
                    # Wait, this doesn't decompose nicely into rank-1 terms of the full tensor.
                    #
                    # Let me think again. The Strassen algorithm computes:
                    # M = A_combined * B_combined where A_combined is 2x2, B_combined is 2x2.
                    # M[i',k'] = sum_j' A_combined[i',j'] * B_combined[j',k']
                    # A_combined[i',j'] = sum_{I,J: 2I+i'=i', 2J+j'=j'} a_IJ * A[2I+i', 2J+j']
                    # This is getting complicated. Let me just enumerate.
                    
                    # For a given (ip, jp, kp), the contribution to C[2I+ip, 2K+kp]
                    # from Strassen term t is:
                    # sum_{I,J,K} a_IJ * b_JK * c_IK * A[2I+ip, 2J+jp] * B[2J+jp, 2K+kp]
                    
                    # This is a sum of products, not a single product.
                    # To make it rank-1, I'd need to fix J as well.
                    # So I should iterate over J too.
                    
                    pass  # Skip this approach, it's not giving rank-1 terms directly.
    
    # The issue is that Strassen's linear combinations mean each "term" is itself
    # a sum of multiple rank-1 contributions. The total rank is still the sum of
    # all individual rank-1 contributions.
    
    # Let me just do a clean schoolbook decomposition but with a different
    # ordering to see if there's any structure I can exploit.
    
    # Actually, the simplest improvement: just use schoolbook. Rank = 112.
    # That's the same as the seed. I need to find a genuine improvement.
    
    # Let me try: for the 4x4x7 case, note that m=4 and we're multiplying
    # 4x4 by 4x7. The tensor has symmetry in a sense.
    
    # Simplest valid improvement: none obvious with pure schoolbook.
    # Let me just return the schoolbook decomposition to ensure validity,
    # then the search can try to find better.
    
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
