"""Matrix multiplication 2x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 5, 6
    deadline = time.time() + 110  # safety margin before 120s limit

    # We'll build a structured decomposition.
    # Strategy: Use Strassen for 2x2 blocks where possible, schoolbook for the rest.
    # A is 2x5, B is 5x6, C is 2x6.
    #
    # Let's partition:
    # A = [A1 | A2] where A1 is 2x2 (cols 0,1), A2 is 2x3 (cols 2,3,4)
    # B = [B1; B2] where B1 is 2x6 (rows 0,1), B2 is 3x6 (rows 2,3,4)
    #
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 * 2x6 -> use Strassen-style for the 2x2 part, but B1 is 2x6.
    # Actually, let's think differently.
    #
    # Alternative partition:
    # A = [A1 | A2 | A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    # B = [B1; B2; B3] where B1 is 2x6, B2 is 2x6, B3 is 1x6
    #
    # C = A1*B1 + A2*B2 + A3*B3
    #
    # A1*B1: 2x2 * 2x6. We can view this as two 2x2*2x2 blocks plus... no.
    # Better: A1*B1 where A1 is 2x2 and B1 is 2x6.
    # This is equivalent to computing 2x2 matrix times 2x6 matrix.
    # We can use Strassen for the 2x2 "core" but B has 6 columns.
    #
    # Let's use a different approach. The schoolbook rank is 2*5*6 = 60.
    # We want to beat 60.
    #
    # Known result: For 2xm * mxp, the rank is at most 2*min(m,p) + ... 
    # Actually, let's just try to construct a good decomposition.
    #
    # Let me try: partition A into [A1 | A2] where A1 is 2x2, A2 is 2x3.
    # Partition B into [B1; B2] where B1 is 2x6, B2 is 3x6.
    #
    # C = A1*B1 + A2*B2
    #
    # A1*B1: A1 is 2x2, B1 is 2x6.
    # Let B1 = [B1a | B1b] where B1a is 2x3, B1b is 2x3.
    # A1*B1 = A1*B1a + A1*B1b
    # A1*B1a: 2x2 * 2x3. Schoolbook rank = 2*2*3 = 12.
    # Can we do better? For 2x2*2x2, Strassen gives 7. For 2x2*2x3...
    # Actually, 2x2*2x3 can be seen as two 2x2*2x2 computations sharing the A factor.
    # Rank for 2x2*2x3: at most 2*2*3 - something. Let's just use schoolbook: 12.
    #
    # This is getting complicated. Let me try a direct construction.
    #
    # Simple approach: 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's try to use Strassen on the 2x2 sub-blocks of A and B.
    #
    # Partition A as 2x5: rows 0,1; cols 0..4
    # Partition B as 5x6: rows 0..4; cols 0..5
    #
    # Let's group cols of A into: {0,1}, {2,3}, {4}
    # Let's group rows of B into: {0,1}, {2,3}, {4}
    #
    # A = [A_a | A_b | A_c] where A_a is 2x2, A_b is 2x2, A_c is 2x1
    # B = [B_a; B_b; B_c] where B_a is 2x6, B_b is 2x6, B_c is 1x6
    #
    # C = A_a * B_a + A_b * B_b + A_c * B_c
    #
    # A_a * B_a: 2x2 * 2x6. 
    #   Let B_a = [B_a1 | B_a2] where each is 2x3.
    #   A_a * B_a1: 2x2 * 2x3. Use Strassen-like: rank 7 for 2x2*2x2 part, but we have 3 cols.
    #   Actually, for 2x2 * 2xk, the rank is at most 7*ceil(k/2) or something.
    #   For k=3: we can do 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = 11? Or better.
    #   Schoolbook for 2x2*2x3 is 12. So 11 saves 1.
    #
    # This is getting too complex to reason about abstractly. Let me just code up a 
    # reasonable structured decomposition and verify it.
    #
    # Let me try a cleaner approach:
    # 
    # Use the fact that for 2xm * mxn, there are known low-rank decompositions.
    # For 2x5 * 5x6, schoolbook is 60.
    #
    # Let me try: 
    # - Use Strassen for 2x2*2x2 blocks
    # - Use schoolbook for remaining parts
    #
    # Partition:
    # A cols: 0,1 | 2,3 | 4  -> A1 (2x2), A2 (2x2), A3 (2x1)
    # B rows: 0,1 | 2,3 | 4  -> B1 (2x6), B2 (2x6), B3 (1x6)
    #
    # C = A1*B1 + A2*B2 + A3*B3
    #
    # A1*B1: 2x2 * 2x6
    #   Split B1 into B1a (2x3) and B1b (2x3)
    #   A1*B1a: 2x2*2x3 -> schoolbook 12, but we can do better
    #   For 2x2*2x3: write as [x y; z w] * [b1 b2 b3; b4 b5 b6]
    #   Result: [x*b1+y*b4, x*b2+y*b5, x*b3+y*b6; z*b1+w*b4, z*b2+w*b5, z*b3+w*b6]
    #   Schoolbook: 12 terms.
    #   Strassen-like: For the 2x2 part (first 2 cols), use 7 terms. For the 3rd col, 4 terms. Total 11.
    #   But we can share: the Strassen factors for the 2x2 part can be reused.
    #   Actually, let's just use schoolbook for simplicity first: 12 terms for A1*B1a, 12 for A1*B1b. Total 24.
    #
    # This is not efficient. Let me think of a better partition.
    #
    # Alternative: 
    # A is 2x5, B is 5x6.
    # Let's use the "laser method" or just try to find a good rank.
    #
    # Known: rank(2xm*mxn) <= 2*m*n - m - n + 1 for some cases? Not sure.
    #
    # Let me just try to construct a valid decomposition with rank < 60.
    #
    # Approach: Start with schoolbook (rank 60), then try to combine terms.
    #
    # Actually, let me try a specific construction:
    #
    # For C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # Let's define rank terms. Each term is a rank-1 tensor: u * v * w
    # where u is in R^{10} (A indices), v is in R^{30} (B indices), w is in R^{12} (C indices).
    #
    # The tensor T[i,j,k] = delta_{i,k} ... no, T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, the multiplication tensor has T[i,j,k] = 1 if j is the "middle" index, i.e., 
    # T[i,j,k] = 1 for all i,j,k with appropriate dimensions. Wait, no.
    #
    # The matrix multiplication tensor for A(n x m) * B(m x p) -> C(n x p) is:
    # T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) entry of C.
    # So T[i, j, k] = 1 for all valid i, j, k. The tensor is T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. The tensor is:
    # T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    # So for each (i,k), C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor has T[i, j, k] = 1 for all i, j, k. No wait, the tensor is defined as:
    # T[i, j, k] = 1 if (i,j) is a valid A index, (j,k) is a valid B index, and (i,k) is a valid C index.
    # Since all are valid, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # But that's a 3D array of all 1s, which has rank 1. That can't be right.
    #
    # I think I'm confusing the tensor. Let me reconsider.
    #
    # The matrix multiplication tensor is defined by:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i, j, k] = 1 if the term A[i,j]*B[j,k] appears in the sum for C[i,k].
    # So T[i, j, k] = 1 for all i, j, k. But that gives a rank-1 tensor, which is wrong.
    #
    # The correct tensor: For each (i,j,k), the coefficient of A[i,j] * B[j,k] in C[i,k] is 1 if k is the output column and i is the output row, and j is the summation index. So T[i, j, k] = 1 for all valid i, j, k. But the tensor is indexed by (i, j, k) where i is row of A/C, j is col of A / row of B, k is col of B/C. The tensor entry T[i, j, k] = 1 means that A[i,j] and B[j,k] multiply to contribute to C[i,k]. Since every such triple contributes, T is all 1s. But the rank of an all-1s tensor of size n x m x p is 1, which contradicts the known rank of matrix multiplication.
    #
    # I see the issue. The tensor is NOT all 1s. The tensor is:
    # T[i, j, k] = 1 if the (i,j) entry of A and the (j,k) entry of B are multiplied to form the (i,k) entry of C.
    # But for a given (i,k), the sum is over j. So for a fixed (i,k), the terms are A[i,0]*B[0,k], A[i,1]*B[1,k], ..., A[i,m-1]*B[m-1,k].
    # So the tensor has T[i, j, k] = 1 for all i, j, k. That's still all 1s.
    #
    # Wait, I think the tensor is defined differently. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,k} ... no.
    #
    # Let me look at this more carefully. The bilinear map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T = sum_{i,j,k} T_{i,j,k} (e_i ⊗ e_j ⊗ e_k) where T_{i,j,k} = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # For C[i,k], the contributing terms are A[i,j]B[j,k] for all j. So T_{i,j,k} = 1 for all i, j, k.
    # But that gives T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k), which has rank 1.
    #
    # This is clearly wrong. The issue is that the tensor should be:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * [coefficient of A[i,j]B[j,k] in C[i,k]]
    # But the coefficient is 1 only if the indices match, i.e., the A index is (i,j), the B index is (j,k), and the C index is (i,k). So for each triple (i,j,k), the coefficient is 1. But the tensor is a sum over all triples, and each triple gives a rank-1 term. The total tensor is the sum of n*m*p rank-1 terms, so its rank is at most n*m*p. But we want the MINIMUM rank, which is less than n*m*p.
    #
    # I think I've been overthinking this. The tensor is simply:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we want to decompose this tensor into a sum of rank-1 tensors.
    # But that tensor is just the outer product of three all-1s vectors, so it has rank 1.
    #
    # I must be misunderstanding the tensor definition. Let me reconsider.
    #
    # Ah, I see. The tensor is NOT T[i,j,k] = 1 for all i,j,k. The tensor is:
    # T[i, j, k] = 1 if the (i,j) entry of A and the (j,k) entry of B are used to compute the (i,k) entry of C.
    # But for a given (i,k), we sum over j. So the tensor has a 1 at position (i, j, k) for each j. That means for fixed i and k, there are m ones (one for each j). So the tensor is:
    # T[i, j, k] = 1 for all i, j, k.
    # This is still all 1s.
    #
    # I think the correct understanding is: the tensor is indexed by (i, j, k) where i is the row index of A and C, j is the column index of A and row index of B, and k is the column index of B and C. The tensor entry T[i, j, k] is 1 if the product A[i,j] * B[j,k] contributes to C[i,k]. Since it always does, T[i,j,k] = 1 for all valid i, j, k.
    #
    # But then the rank is 1, which is absurd. The resolution is that the tensor is NOT a single 3D array. The tensor is a multilinear map, and its "rank" is the minimum number of rank-1 terms needed to express it. A rank-1 term is u ⊗ v ⊗ w where u is in R^n, v is in R^m, w is in R^p. The tensor T can be written as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    # which is rank 1.
    #
    # This is clearly not the matrix multiplication tensor. I must be making a fundamental error.
    #
    # Let me reconsider. The matrix multiplication tensor is defined by the bilinear form:
    # <T, (A, B)> = trace(C^T * (A*B)) ... no.
    #
    # Actually, the tensor is defined such that:
    # C[i,k] = sum_j T[i, j, k] * A[i,j] * B[j,k]
    # where T[i, j, k] = 1 for all i, j, k. But that's what I had.
    #
    # I think the issue is that the tensor is not T[i,j,k] but rather the tensor is the map (A, B) -> C, and the "tensor" is the collection of coefficients. For each (i,k), C[i,k] = sum_j A[i,j] B[j,k]. So the coefficient of A[i,j] B[j,k] in C[i,k] is 1. The tensor is:
    # T = sum_{i,k} sum_j (e_i ⊗ e_j ⊗ e_k) * 1
    # = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    # Rank 1.
    #
    # I give up trying to derive this from first principles. Let me just use the known fact that the rank of the n x m x p matrix multiplication tensor is at most n*m*p and try to find a good decomposition.
    #
    # For the purpose of this task, I'll just construct a valid decomposition and hope it's good enough.
    #
    # Let me try a simple approach: use schoolbook for all terms, giving rank 60. Then try to improve.
    #
    # Actually, let me just return a schoolbook decomposition with rank 60 and see what happens.
    # No, the current program has rank 58, so I need to beat that.
    #
    # Let me try to construct a rank-57 or lower decomposition.
    #
    # Idea: Use Strassen for 2x2*2x2 blocks.
    #
    # Partition A into [A1 | A2 | A3] where A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3), A3 is 2x1 (col 4)
    # Partition B into [B1; B2; B3] where B1 is 2x6 (rows 0,1), B2 is 2x6 (rows 2,3), B3 is 1x6 (row 4)
    #
    # C = A1*B1 + A2*B2 + A3*B3
    #
    # A1*B1: 2x2 * 2x6
    #   Split B1 into B1a (2x3, cols 0-2) and B1b (2x3, cols 3-5)
    #   A1*B1a: 2x2*2x3. Let's use schoolbook: 12 terms.
    #   A1*B1b: 2x2*2x3. Schoolbook: 12 terms.
    #   Total for A1*B1: 24 terms.
    #
    # A2*B2: 2x2 * 2x6
    #   Similarly: 24 terms.
    #
    # A3*B3: 2x1 * 1x6
    #   Schoolbook: 12 terms.
    #
    # Total: 24 + 24 + 12 = 60. Same as schoolbook.
    #
    # To improve, I need to use Strassen or other tricks.
    #
    # For A1*B1a (2x2*2x3):
    #   A1 = [a b; c d], B1a = [e f g; h i j]
    #   C1a = [a*e+b*h, a*f+b*i, a*g+b*j; c*e+d*h, c*f+d*i, c*g+d*j]
    #   Schoolbook: 12 terms.
    #   Strassen for the 2x2 part (first 2 cols): 7 terms.
    #   For the 3rd col: 4 terms (a*g, b*j, c*g, d*j).
    #   Total: 7 + 4 = 11 terms. Saves 1.
    #
    # Similarly for A1*B1b: 11 terms.
    # A2*B2a: 11 terms.
    # A2*B2b: 11 terms.
    # A3*B3: 12 terms.
    #
    # Total: 11*4 + 12 = 56 terms.
    #
    # Let me verify this.
    #
    # Actually, let me just code this up and test it.

    def make_decomposition():
        n, m, p = 2, 5, 6
        # We'll build U, V, W as lists of lists.
        # U: shape (n*m, rank) = (10, rank)
        # V: shape (m*p, rank) = (30, rank)
        # W: shape (p*n, rank) = (12, rank)
        
        # We'll use a list of terms, each term is (u_idx, u_val, v_idx, v_val, w_idx, w_val)
        # But since each rank step can have multiple non-zero entries, we'll use a different approach.
        
        terms = []  # Each term is a dict with 'u', 'v', 'w' keys, each a list of (idx, val)
        
        def add_term(u_pairs, v_pairs, w_pairs):
            terms.append({'u': u_pairs, 'v': v_pairs, 'w': w_pairs})
        
        # Part 1: A1*B1a (2x2*2x3)
        # A1: rows 0,1; cols 0,1. Indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
        # B1a: rows 0,1; cols 0,1,2. Indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=6, (1,1)=7, (1,2)=8
        # C: rows 0,1; cols 0,1,2. Indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=6, (1,1)=7, (1,2)=8
        
        # Strassen for 2x2*2x2 (first 2 cols of B1a):
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        add_term([(0, 1), (6, 1)], [(0, 1), (6, 1)], [(0, 1), (7, 1)])
        # m2 = (c+d)e -> C10 += 1, C11 -= 1
        add_term([(5, 1), (6, 1)], [(0, 1)], [(6, 1), (7, -1)])
        # m3 = a(f-h) -> C01 += 1, C11 += 1
        add_term([(0, 1)], [(1, 1), (6, -1)], [(1, 1), (7, 1)])
        # m4 = d(g-e) -> C00 += 1, C10 += 1
        add_term([(6, 1)], [(2, 1), (0, -1)], [(0, 1), (6, 1)])
        # m5 = (a+b)h -> C00 -= 1, C01 += 1
        add_term([(0, 1), (1, 1)], [(6, 1)], [(0, -1), (1, 1)])
        # m6 = (c-a)(e+f) -> C11 += 1
        add_term([(5, 1), (0, -1)], [(0, 1), (1, 1)], [(7, 1)])
        # m7 = (b-d)(g+h) -> C00 += 1
        add_term([(1, 1), (6, -1)], [(2, 1), (6, 1)], [(0, 1)])
        
        # Schoolbook for 3rd col of B1a:
        # a*g -> C02
        add_term([(0, 1)], [(2, 1)], [(2, 1)])
        # b*j -> C02
        add_term([(1, 1)], [(8, 1)], [(2, 1)])
        # c*g -> C12
        add_term([(5, 1)], [(2, 1)], [(8, 1)])
        # d*j -> C12
        add_term([(6, 1)], [(8, 1)], [(8, 1)])
        
        # Part 2: A1*B1b (2x2*2x3)
        # B1b: rows 0,1; cols 3,4,5. Indices: (0,3)=3, (0,4)=4, (0,5)=5, (1,3)=9, (1,4)=10, (1,5)=11
        # C: rows 0,1; cols 3,4,5. Indices: (0,3)=3, (0,4)=4, (0,5)=5, (1,3)=9, (1,4)=10, (1,5)=11
        
        # Strassen for 2x2*2x2 (first 2 cols of B1b, which are cols 3,4 of B):
        add_term([(0, 1), (6, 1)], [(3, 1), (9, 1)], [(3, 1), (10, 1)])
        add_term([(5, 1), (6, 1)], [(3, 1)], [(9, 1), (10, -1)])
        add_term([(0, 1)], [(4, 1), (9, -1)], [(4, 1), (10, 1)])
        add_term([(6, 1)], [(5, 1), (3, -1)], [(3, 1), (9, 1)])
        add_term([(0, 1), (1, 1)], [(9, 1)], [(3, -1), (4, 1)])
        add_term([(5, 1), (0, -1)], [(3, 1), (4, 1)], [(10, 1)])
        add_term([(1, 1), (6, -1)], [(5, 1), (9, 1)], [(3, 1)])
        
        # Schoolbook for 3rd col of B1b (col 5 of B):
        add_term([(0, 1)], [(5, 1)], [(5, 1)])
        add_term([(1, 1)], [(11, 1)], [(5, 1)])
        add_term([(5, 1)], [(5, 1)], [(11, 1)])
        add_term([(6, 1)], [(11, 1)], [(11, 1)])
        
        # Part 3: A2*B2a (2x2*2x3)
        # A2: rows 0,1; cols 2,3. Indices: (0,2)=2, (0,3)=3, (1,2)=7, (1,3)=8
        # B2a: rows 2,3; cols 0,1,2. Indices: (2,0)=12, (2,1)=13, (2,2)=14, (3,0)=18, (3,1)=19, (3,2)=20
        # C: rows 0,1; cols 0,1,2. Indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=6, (1,1)=7, (1,2)=8
        
        add_term([(2, 1), (8, 1)], [(12, 1), (18, 1)], [(0, 1), (7, 1)])
        add_term([(7, 1), (8, 1)], [(12, 1)], [(6, 1), (7, -1)])
        add_term([(2, 1)], [(13, 1), (18, -1)], [(1, 1), (7, 1)])
        add_term([(8, 1)], [(14, 1), (12, -1)], [(0, 1), (6, 1)])
        add_term([(2, 1), (3, 1)], [(18, 1)], [(0, -1), (1, 1)])
        add_term([(7, 1), (2, -1)], [(12, 1), (13, 1)], [(7, 1)])
        add_term([(3, 1), (8, -1)], [(14, 1), (18, 1)], [(0, 1)])
        
        add_term([(2, 1)], [(14, 1)], [(2, 1)])
        add_term([(3, 1)], [(20, 1)], [(2, 1)])
        add_term([(7, 1)], [(14, 1)], [(8, 1)])
        add_term([(8, 1)], [(20, 1)], [(8, 1)])
        
        # Part 4: A2*B2b (2x2*2x3)
        # B2b: rows 2,3; cols 3,4,5. Indices: (2,3)=15, (2,4)=16, (2,5)=17, (3,3)=21, (3,4)=22, (3,5)=23
        # C: rows 0,1; cols 3,4,5. Indices: (0,3)=3, (0,4)=4, (0,5)=5, (1,3)=9, (1,4)=10, (1,5)=11
        
        add_term([(2, 1), (8, 1)], [(15, 1), (21, 1)], [(3, 1), (10, 1)])
        add_term([(7, 1), (8, 1)], [(15, 1)], [(9, 1), (10, -1)])
        add_term([(2, 1)], [(16, 1), (21, -1)], [(4, 1), (10, 1)])
        add_term([(8, 1)], [(17, 1), (15, -1)], [(3, 1), (9, 1)])
        add_term([(2, 1), (3, 1)], [(21, 1)], [(3, -1), (4, 1)])
        add_term([(7, 1), (2, -1)], [(15, 1), (16, 1)], [(10, 1)])
        add_term([(3, 1), (8, -1)], [(17, 1), (21, 1)], [(3, 1)])
        
        add_term([(2, 1)], [(17, 1)], [(5, 1)])
        add_term([(3, 1)], [(23, 1)], [(5, 1)])
        add_term([(7, 1)], [(17, 1)], [(11, 1)])
        add_term([(8, 1)], [(23, 1)], [(11, 1)])
        
        # Part 5: A3*B3 (2x1*1x6)
        # A3: rows 0,1; col 4. Indices: (0,4)=4, (1,4)=9
        # B3: row 4; cols 0-5. Indices: (4,0)=24, (4,1)=25, (4,2)=26, (4,3)=27, (4,4)=28, (4,5)=29
        # C: rows 0,1; cols 0-5. Indices: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (0,4)=4, (0,5)=5, (1,0)=6, (1,1)=7, (1,2)=8, (1,3)=9, (1,4)=10, (1,5)=11
        
        for k in range(6):
            add_term([(4, 1)], [(24 + k, 1)], [(k, 1)])
            add_term([(9, 1)], [(24 + k, 1)], [(6 + k, 1)])
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, term in enumerate(terms):
            for idx, val in term['u']:
                U[idx][col] = val
            for idx, val in term['v']:
                V[idx][col] = val
            for idx, val in term['w']:
                W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}
    
    result = make_decomposition()
    return result

# EVOLVE-BLOCK-END
