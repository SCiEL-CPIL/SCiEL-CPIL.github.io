"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x7 matrix.
    
    Strategy:
    - Decompose the output space (7-dimensional) into a 4-dim subspace and a 3-dim subspace.
    - For the 4x4x4 part, use Strassen's algorithm (rank 7).
    - For the 4x4x3 part, use schoolbook (rank 4*4*3 = 48).
    - Total rank = 7 + 48 = 55.
    
    Wait, that's worse than schoolbook 112? No, 55 < 112, so it's an improvement.
    But can we do better?
    
    Actually, let's think about this more carefully.
    
    For 4x4x4, Strassen gives rank 7.
    For 4x4x3, we can use schoolbook: 4*4*3 = 48.
    Total: 7 + 48 = 55.
    
    But maybe we can do better on the 4x4x3 part? The minimal rank for 4x4x3 is not well-known,
    but it's at most 48 (schoolbook). There might be better decompositions.
    
    However, for now, let's implement the 7+48=55 decomposition.
    
    The tensor for matrix multiplication (A * B) where A is 4x4 and B is 4x7:
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    where i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    
    We split k into k1 in {0,1,2,3} and k2 in {4,5,6}.
    
    Part 1: k1 in {0,1,2,3} -> 4x4x4 tensor, use Strassen (rank 7)
    Part 2: k2 in {4,5,6} -> 4x4x3 tensor, use schoolbook (rank 48)
    
    Total rank: 55
    
    Let's implement Strassen for 4x4x4 first.
    
    Strassen for 2x2x2 has rank 7. For 4x4x4, we can apply Strassen recursively:
    Split 4x4 into four 2x2 blocks.
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    
    C = A*B = [[C11, C12], [C21, C22]]
    
    Strassen for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    So for 4x4x4, we have 7 multiplications of 2x2x2 tensors, each of which can be done with rank 7.
    Total rank: 7 * 7 = 49? No, that's not right.
    
    Actually, when we apply Strassen to 4x4x4, we get 7 "sub-problems" of size 2x2x2.
    Each 2x2x2 multiplication can be done with rank 7.
    So total rank = 7 * 7 = 49.
    
    But wait, the entries in M1, ..., M7 are sums/differences of 2x2 blocks, so they are linear combinations.
    The tensor decomposition for 4x4x4 using Strassen has rank 49.
    
    Hmm, but I recall that the minimal rank for 4x4x4 is at most 49 (from Strassen), and it's known to be at least 26.
    Actually, I'm not sure of the exact minimal rank.
    
    For now, let's use the Strassen-based decomposition with rank 49 for the 4x4x4 part.
    
    And for the 4x4x3 part, schoolbook gives rank 48.
    
    Total: 49 + 48 = 97.
    
    That's better than 112, but not as good as I initially thought.
    
    Can we do better on the 4x4x3 part?
    
    Actually, for m x n x p matrix multiplication, the schoolbook rank is m*n*p.
    For 4x4x3, that's 48.
    
    Is there a known better decomposition for 4x4x3? I don't think so, at least not a simple one.
    
    So total rank = 49 + 48 = 97.
    
    Let's implement this.
    
    Actually, let me reconsider. The problem says "Find a rank-minimal exact tensor decomposition".
    So we want the smallest possible rank.
    
    The current schoolbook gives 112.
    Our proposed decomposition gives 97.
    
    Let's implement the 97-rank decomposition.
    
    Steps:
    1. Implement Strassen for 2x2x2 (rank 7).
    2. Use it to build a 4x4x4 decomposition (rank 49).
    3. Build a 4x4x3 schoolbook decomposition (rank 48).
    4. Combine them.
    
    The output factor W follows the convention W[k*n+i, r] where n=4, so W[k*4+i, r].
    
    Let's start coding.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    
    rank1 = 49  # Strassen 4x4x4
    rank2 = 48  # schoolbook 4x4x3
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 97
    
    # Build Strassen 4x4x4 decomposition
    # We'll create a helper to get the Strassen 2x2x2 decomposition
    
    # Strassen 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need to express each term as a rank-1 tensor.
    # Each M_i is a product of a linear form in A and a linear form in B.
    # So each M_i contributes 1 to the rank.
    # Then C_ij is a linear combination of M_i's.
    
    # Let's define the Strassen 2x2x2 decomposition explicitly.
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    # U_A for 2x2: 4 entries
    # V_B for 2x2: 4 entries
    # W_C for 2x2: 4 entries
    
    # M1: (a+d)(e+h)
    # U_A: [1, 0, 0, 1]
    # V_B: [1, 0, 0, 1]
    # W_C: contributes to C11 (+1) and C22 (+1)
    
    # M2: (c+d)e
    # U_A: [0, 0, 1, 1]
    # V_B: [1, 0, 0, 0]
    # W_C: contributes to C21 (+1) and C22 (-1)
    
    # M3: a(f-h)
    # U_A: [1, 0, 0, 0]
    # V_B: [0, 1, 0, -1]
    # W_C: contributes to C12 (+1) and C22 (+1)
    
    # M4: d(g-e)
    # U_A: [0, 0, 0, 1]
    # V_B: [0, 0, 1, 0] - wait, g is (1,0), e is (0,0)
    # V_B: [-1, 0, 1, 0]
    # W_C: contributes to C11 (+1) and C21 (+1)
    
    # M5: (a+b)h
    # U_A: [1, 1, 0, 0]
    # V_B: [0, 0, 0, 1]
    # W_C: contributes to C11 (-1) and C12 (+1)
    
    # M6: (c-a)(e+f)
    # U_A: [-1, 0, 1, 0]
    # V_B: [1, 1, 0, 0]
    # W_C: contributes to C22 (+1)
    
    # M7: (b-d)(g+h)
    # U_A: [0, 1, 0, -1]
    # V_B: [0, 0, 1, 1]
    # W_C: contributes to C11 (+1)
    
    # Let's verify C11 = M1 + M4 - M5 + M7
    # C11 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg
    # Correct!
    
    # C12 = M3 + M5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh
    # Correct!
    
    # C21 = M2 + M4 = (c+d)e + d(g-e) = ce + de + dg - de = ce + dg
    # Correct!
    
    # C22 = M1 - M2 + M3 + M6 = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    # = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
    # = dh + cf
    # Correct!
    
    # So the Strassen 2x2x2 decomposition has 7 terms.
    
    # Now for 4x4x4 using Strassen:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each M_i is a 2x2x2 multiplication, which has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let's build the 4x4x4 decomposition.
    
    # A is 4x4, indexed as (i, j) where i, j in {0, 1, 2, 3}
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # B is 4x4, indexed as (j, k) where j, k in {0, 1, 2, 3}
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C is 4x4, indexed as (i, k) where i, k in {0, 1, 2, 3}
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # For each M_i, we need to:
    # 1. Extract the relevant 2x2 blocks from A and B.
    # 2. Apply the Strassen 2x2x2 decomposition to get 7 rank-1 terms.
    # 3. Map these to the full 4x4x4 tensor.
    
    # Let's define the Strassen 2x2x2 terms as (U_A, V_B, W_C) tuples.
    
    def strassen_2x2x2():
        """Return the 7 (U_A, V_B, W_C) tuples for Strassen 2x2x2."""
        # U_A: 4 entries (a, b, c, d)
        # V_B: 4 entries (e, f, g, h)
        # W_C: 4 entries (C11, C12, C21, C22)
        
        terms = []
        
        # M1: (a+d)(e+h)
        U_A = [1, 0, 0, 1]
        V_B = [1, 0, 0, 1]
        W_C = [1, 0, 0, 1]  # C11 +1, C22 +1
        terms.append((U_A, V_B, W_C))
        
        # M2: (c+d)e
        U_A = [0, 0, 1, 1]
        V_B = [1, 0, 0, 0]
        W_C = [0, 0, 1, -1]  # C21 +1, C22 -1
        terms.append((U_A, V_B, W_C))
        
        # M3: a(f-h)
        U_A = [1, 0, 0, 0]
        V_B = [0, 1, 0, -1]
        W_C = [0, 1, 0, 1]  # C12 +1, C22 +1
        terms.append((U_A, V_B, W_C))
        
        # M4: d(g-e)
        U_A = [0, 0, 0, 1]
        V_B = [-1, 0, 1, 0]
        W_C = [1, 0, 1, 0]  # C11 +1, C21 +1
        terms.append((U_A, V_B, W_C))
        
        # M5: (a+b)h
        U_A = [1, 1, 0, 0]
        V_B = [0, 0, 0, 1]
        W_C = [-1, 1, 0, 0]  # C11 -1, C12 +1
        terms.append((U_A, V_B, W_C))
        
        # M6: (c-a)(e+f)
        U_A = [-1, 0, 1, 0]
        V_B = [1, 1, 0, 0]
        W_C = [0, 0, 0, 1]  # C22 +1
        terms.append((U_A, V_B, W_C))
        
        # M7: (b-d)(g+h)
        U_A = [0, 1, 0, -1]
        V_B = [0, 0, 1, 1]
        W_C = [1, 0, 0, 0]  # C11 +1
        terms.append((U_A, V_B, W_C))
        
        return terms
    
    strassen_terms = strassen_2x2x2()
    
    # Now build the 4x4x4 decomposition.
    # For each of the 7 M_i, we have 7 sub-terms from Strassen 2x2x2.
    # Total: 49 terms.
    
    # Let's define the block structure.
    # A blocks: A11, A12, A21, A22
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks: B11, B12, B21, B22
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C blocks: C11, C12, C21, C22
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # For M1 = (A11+A22)(B11+B22):
    # U_A: A11 + A22
    # V_B: B11 + B22
    # W_C: C11 + C22 (but actually, the result is added to C11 and C22)
    
    # Wait, I need to be more careful.
    # M1 is a 2x2 matrix, and it contributes to C11 and C22.
    # Specifically, C11 += M1 and C22 += M1.
    
    # So for each sub-term in M1's Strassen decomposition, we need to add it to C11 and C22.
    
    # Let's define the block indices.
    # A block (bi, bj) where bi, bj in {0, 1}
    # bi=0: rows 0-1, bi=1: rows 2-3
    # bj=0: cols 0-1, bj=1: cols 2-3
    
    # B block (bi, bj) where bi, bj in {0, 1}
    # bi=0: rows 0-1, bi=1: rows 2-3
    # bj=0: cols 0-1, bj=1: cols 2-3
    
    # C block (bi, bj) where bi, bj in {0, 1}
    # bi=0: rows 0-1, bi=1: rows 2-3
    # bj=0: cols 0-1, bj=1: cols 2-3
    
    # For M1 = (A11+A22)(B11+B22):
    # U_A blocks: A11 (0,0) and A22 (1,1)
    # V_B blocks: B11 (0,0) and B12 (0,1) and B21 (1,0) and B22 (1,1)
    # Wait, B11+B22 means B block (0,0) + B block (1,1)
    
    # Let's define the M_i formulas in terms of block operations.
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # And the C blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M_i, we have a linear combination of A blocks and a linear combination of B blocks.
    # Then we apply Strassen 2x2x2 to get 7 sub-terms.
    
    # Let's define the A and B block coefficients for each M_i.
    
    # M1: A blocks: A11 (0,0) + A22 (1,1), B blocks: B11 (0,0) + B22 (1,1)
    # M2: A blocks: A21 (1,0) + A22 (1,1), B blocks: B11 (0,0)
    # M3: A blocks: A11 (0,0), B blocks: B12 (0,1) - B22 (1,1)
    # M4: A blocks: A22 (1,1), B blocks: B21 (1,0) - B11 (0,0)
    # M5: A blocks: A11 (0,0) + A12 (0,1), B blocks: B22 (1,1)
    # M6: A blocks: A21 (1,0) - A11 (0,0), B blocks: B11 (0,0) + B12 (0,1)
    # M7: A blocks: A12 (0,1) - A22 (1,1), B blocks: B21 (1,0) + B22 (1,1)
    
    # And the C block contributions:
    # M1: C11 (+1), C22 (+1)
    # M2: C21 (+1), C22 (-1)
    # M3: C12 (+1), C22 (+1)
    # M4: C11 (+1), C21 (+1)
    # M5: C11 (-1), C12 (+1)
    # M6: C22 (+1)
    # M7: C11 (+1)
    
    # Now, for each M_i, we have:
    # U_A = sum of A block coefficients
    # V_B = sum of B block coefficients
    # W_C = sum of C block contributions
    
    # But U_A, V_B, W_C are 2x2 matrices (since each block is 2x2).
    # So we can apply Strassen 2x2x2 to each M_i.
    
    # Let's build the 4x4x4 decomposition.
    
    # For each M_i (i in 0..6):
    #   For each Strassen sub-term (U_A_2x2, V_B_2x2, W_C_2x2):
    #     Expand U_A_2x2 to 4x4 using the A block coefficients
    #     Expand V_B_2x2 to 4x4 using the B block coefficients
    #     Expand W_C_2x2 to 4x4 using the C block contributions
    
    # Let's define the block coefficient vectors.
    
    # A blocks: 4 blocks, each 2x2
    # Block (bi, bj) corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    
    # For M1: A blocks: (0,0): +1, (1,1): +1, others: 0
    # For M2: A blocks: (1,0): +1, (1,1): +1, others: 0
    # For M3: A blocks: (0,0): +1, others: 0
    # For M4: A blocks: (1,1): +1, others: 0
    # For M5: A blocks: (0,0): +1, (0,1): +1, others: 0
    # For M6: A blocks: (1,0): +1, (0,0): -1, others: 0
    # For M7: A blocks: (0,1): +1, (1,1): -1, others: 0
    
    a_block_coeffs = [
        [(0,0,1), (1,1,1)],  # M1
        [(1,0,1), (1,1,1)],  # M2
        [(0,0,1)],           # M3
        [(1,1,1)],           # M4
        [(0,0,1), (0,1,1)],  # M5
        [(1,0,1), (0,0,-1)], # M6
        [(0,1,1), (1,1,-1)], # M7
    ]
    
    # B blocks: 4 blocks, each 2x2
    # For M1: B blocks: (0,0): +1, (1,1): +1, others: 0
    # For M2: B blocks: (0,0): +1, others: 0
    # For M3: B blocks: (0,1): +1, (1,1): -1, others: 0
    # For M4: B blocks: (1,0): +1, (0,0): -1, others: 0
    # For M5: B blocks: (1,1): +1, others: 0
    # For M6: B blocks: (0,0): +1, (0,1): +1, others: 0
    # For M7: B blocks: (1,0): +1, (1,1): +1, others: 0
    
    b_block_coeffs = [
        [(0,0,1), (1,1,1)],  # M1
        [(0,0,1)],           # M2
        [(0,1,1), (1,1,-1)], # M3
        [(1,0,1), (0,0,-1)], # M4
        [(1,1,1)],           # M5
        [(0,0,1), (0,1,1)],  # M6
        [(1,0,1), (1,1,1)],  # M7
    ]
    
    # C blocks: 4 blocks, each 2x2
    # For M1: C11 (+1), C22 (+1)
    # For M2: C21 (+1), C22 (-1)
    # For M3: C12 (+1), C22 (+1)
    # For M4: C11 (+1), C21 (+1)
    # For M5: C11 (-1), C12 (+1)
    # For M6: C22 (+1)
    # For M7: C11 (+1)
    
    c_block_coeffs = [
        [(0,0,1), (1,1,1)],  # M1
        [(1,0,1), (1,1,-1)], # M2
        [(0,1,1), (1,1,1)],  # M3
        [(0,0,1), (1,0,1)],  # M4
        [(0,0,-1), (0,1,1)], # M5
        [(1,1,1)],           # M6
        [(0,0,1)],           # M7
    ]
    
    # Now build the 4x4x4 decomposition.
    
    col = 0
    for mi in range(7):
        a_coeffs = a_block_coeffs[mi]
        b_coeffs = b_block_coeffs[mi]
        c_coeffs = c_block_coeffs[mi]
        
        for U_A_2x2, V_B_2x2, W_C_2x2 in strassen_terms:
            # Expand U_A_2x2 to 4x4
            U_A_4x4 = [0] * 16
            for (bi, bj, coeff) in a_coeffs:
                for ii in range(2):
                    for jj in range(2):
                        row = 2*bi + ii
                        col_a = 2*bj + jj
                        U_A_4x4[row * 4 + col_a] += coeff * U_A_2x2[ii * 2 + jj]
            
            # Expand V_B_2x2 to 4x4
            V_B_4x4 = [0] * 16
            for (bi, bj, coeff) in b_coeffs:
                for ii in range(2):
                    for jj in range(2):
                        row = 2*bi + ii
                        col_b = 2*bj + jj
                        V_B_4x4[row * 4 + col_b] += coeff * V_B_2x2[ii * 2 + jj]
            
            # Expand W_C_2x2 to 4x4
            W_C_4x4 = [0] * 16
            for (bi, bj, coeff) in c_coeffs:
                for ii in range(2):
                    for jj in range(2):
                        row = 2*bi + ii
                        col_c = 2*bj + jj
                        W_C_4x4[row * 4 + col_c] += coeff * W_C_2x2[ii * 2 + jj]
            
            # Add to U, V, W
            for i in range(16):
                U[i][col] += U_A_4x4[i]
                V[i][col] += V_B_4x4[i]
            for i in range(16):
                W[i][col] += W_C_4x4[i]
            
            col += 1
    
    # Now build the 4x4x3 schoolbook decomposition.
    # This is for the remaining 3 columns of B (cols 4, 5, 6).
    # B is 4x7, so cols 0-3 are handled by the 4x4x4 part, cols 4-6 by this part.
    
    # For schoolbook 4x4x3:
    # For each (i, j, k) where i in {0,1,2,3}, j in {0,1,2,3}, k in {4,5,6}:
    # U[i*4+j] = 1
    # V[j*7+k] = 1
    # W[k*4+i] = 1
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
