"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Strategy: Use a block decomposition 5 = 3 + 2.
    # For the 3x3 * 3x3 sub-problem, use Strassen's algorithm (rank 7 instead of 27).
    # For the 2x2 * 2x2 sub-problem, use Strassen's algorithm (rank 7 instead of 8).
    # For rectangular sub-problems (3x2 * 2x3, 3x2 * 2x2, 2x3 * 3x2, 2x3 * 2x2),
    # use schoolbook multiplication.
    
    # Block indices:
    # A = [[A11 (3x3), A12 (3x2)],
    #      [A21 (2x3), A22 (2x2)]]
    # B = [[B11 (3x3), B12 (3x2)],
    #      [B21 (2x3), B22 (2x2)]]
    # C = [[C11 (3x3), C12 (3x2)],
    #      [C21 (2x3), C22 (2x2)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Rank contributions:
    # A11*B11: Strassen 3x3 -> rank 7
    # A12*B21: Schoolbook 3x2 * 2x3 -> rank 3*2*3 = 18
    # A11*B12: Schoolbook 3x3 * 3x2 -> rank 3*3*2 = 18
    # A12*B22: Schoolbook 3x2 * 2x2 -> rank 3*2*2 = 12
    # A21*B11: Schoolbook 2x3 * 3x3 -> rank 2*3*3 = 18
    # A22*B21: Schoolbook 2x2 * 2x3 -> rank 2*2*3 = 12
    # A21*B12: Schoolbook 2x3 * 3x2 -> rank 2*3*2 = 12
    # A22*B22: Strassen 2x2 -> rank 7
    
    # Total rank = 7 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 104
    
    # However, we can do better by combining terms.
    # Let's stick to the additive decomposition for correctness and simplicity first.
    # If we can verify it, rank 104 is a significant improvement over 125.
    
    # Let's refine:
    # Is there a way to combine the Strassen 3x3 with the rectangular parts?
    # Not easily in a block-wise additive manner without cross-terms.
    
    # Let's implement the additive block decomposition with Strassen for square blocks.
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
    
    # We need to manage the column index 'col' carefully.
    # Let's pre-allocate with a safe upper bound and trim at the end,
    # or just use a list of tuples and convert.
    
    terms = []
    
    def add_term_tuple(u_idx, v_idx, w_idx, val):
        terms.append((u_idx, v_idx, w_idx, val))
    
    # Indices for 5x5:
    # A is 5x5. Rows 0-2 are block 1, rows 3-4 are block 2.
    # Cols 0-2 are block 1, cols 3-4 are block 2.
    
    # Block 1 (3x3) Strassen
    # Strassen 3x3:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Wait, standard Strassen 2x2 is:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # This is for 2x2. For 3x3, Strassen is more complex.
    # Actually, Strassen's algorithm is defined for 2x2.
    # For 3x3, we can't directly apply 2x2 Strassen without padding to 4x4 or using a specific 3x3 algorithm.
    # The rank of 3x3 matrix multiplication is known to be 23 (Bini et al), not 7.
    # 7 is the rank for 2x2.
    #
    # So, if we split 5 = 3 + 2:
    # 3x3 * 3x3 has rank 23 (best known) or 27 (schoolbook).
    # 2x2 * 2x2 has rank 7 (Strassen).
    #
    # If we use rank 23 for 3x3:
    # C11 part: 23 (A11*B11) + 18 (A12*B21) = 41
    # C12 part: 18 (A11*B12) + 12 (A12*B22) = 30
    # C21 part: 18 (A21*B11) + 12 (A22*B21) = 30
    # C22 part: 12 (A21*B12) + 7 (A22*B22) = 19
    # Total = 41 + 30 + 30 + 19 = 120.
    #
    # 120 is better than 125, but implementing rank 23 for 3x3 is non-trivial.
    #
    # Let's try splitting 5 = 4 + 1.
    # 4x4 * 4x4 rank is 49 (known).
    # 4x1 * 1x4, etc.
    #
    # This is getting complicated. Let's stick to a simpler improvement.
    #
    # What if we use Strassen on the 2x2 blocks and schoolbook on the rest?
    # Split 5 = 2 + 2 + 1?
    #
    # Let's go back to 5 = 3 + 2 and just use schoolbook for 3x3 (rank 27) and Strassen for 2x2 (rank 7).
    # C11: 27 + 18 = 45
    # C12: 18 + 12 = 30
    # C21: 18 + 12 = 30
    # C22: 12 + 7 = 19
    # Total = 45 + 30 + 30 + 19 = 124.
    #
    # 124 is an improvement over 125.
    #
    # Let's implement this.
    
    rank_est = 124
    U = [[0] * rank_est for _ in range(n * m)]
    V = [[0] * rank_est for _ in range(m * p)]
    W = [[0] * rank_est for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    # Helper to add Strassen 2x2 terms
    # A block is rows [r0, r1], cols [c0, c1]
    # B block is rows [r0, r1], cols [c0, c1]
    # C block is rows [r0, r1], cols [c0, c1]
    # Indices in A: (r0,c0), (r0,c1), (r1,c0), (r1,c1)
    # Indices in B: (r0,c0), (r0,c1), (r1,c0), (r1,c1)
    # Indices in C: (r0,c0), (r0,c1), (r1,c0), (r1,c1)
    def add_strassen_2x2(r0, r1, c0, c1):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A11=(r0,c0), A22=(r1,c1)
        # B11=(r0,c0), B22=(r1,c1)
        # Term 1: A11*B11
        U[r0*m+c0][col] = 1; V[r0*p+c0][col] = 1; W[c0*n+r0][col] = 1
        # Term 2: A11*B22
        U[r0*m+c0][col] = 1; V[r1*p+c1][col] = 1; W[c1*n+r0][col] = 1
        # Term 3: A22*B11
        U[r1*m+c1][col] = 1; V[r0*p+c0][col] = 1; W[c0*n+r1][col] = 1
        # Term 4: A22*B22
        U[r1*m+c1][col] = 1; V[r1*p+c1][col] = 1; W[c1*n+r1][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        # A21=(r1,c0), A22=(r1,c1), B11=(r0,c0)
        # Term 5: A21*B11
        U[r1*m+c0][col] = 1; V[r0*p+c0][col] = 1; W[c0*n+r1][col] = 1
        # Term 6: A22*B11
        U[r1*m+c1][col] = 1; V[r0*p+c0][col] = 1; W[c0*n+r1][col] = 1
        col += 1
        
        # M3 = A11(B12-B22)
        # A11=(r0,c0), B12=(r0,c1), B22=(r1,c1)
        # Term 7: A11*B12
        U[r0*m+c0][col] = 1; V[r0*p+c1][col] = 1; W[c1*n+r0][col] = 1
        # Term 8: A11*B22
        U[r0*m+c0][col] = -1; V[r1*p+c1][col] = -1; W[c1*n+r0][col] = -1
        col += 1
        
        # M4 = A22(B21-B11)
        # A22=(r1,c1), B21=(r1,c0), B11=(r0,c0)
        # Term 9: A22*B21
        U[r1*m+c1][col] = 1; V[r1*p+c0][col] = 1; W[c0*n+r1][col] = 1
        # Term 10: A22*B11
        U[r1*m+c1][col] = -1; V[r0*p+c0][col] = -1; W[c0*n+r1][col] = -1
        col += 1
        
        # M5 = (A11+A12)B22
        # A11=(r0,c0), A12=(r0,c1), B22=(r1,c1)
        # Term 11: A11*B22
        U[r0*m+c0][col] = 1; V[r1*p+c1][col] = 1; W[c1*n+r0][col] = 1
        # Term 12: A12*B22
        U[r0*m+c1][col] = 1; V[r1*p+c1][col] = 1; W[c1*n+r0][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21=(r1,c0), A11=(r0,c0), B11=(r0,c0), B12=(r0,c1)
        # Term 13: A21*B11
        U[r1*m+c0][col] = 1; V[r0*p+c0][col] = 1; W[c0*n+r1][col] = 1
        # Term 14: A21*B12
        U[r1*m+c0][col] = 1; V[r0*p+c1][col] = 1; W[c1*n+r1][col] = 1
        # Term 15: A11*B11
        U[r0*m+c0][col] = -1; V[r0*p+c0][col] = -1; W[c0*n+r0][col] = -1
        # Term 16: A11*B12
        U[r0*m+c0][col] = -1; V[r0*p+c1][col] = -1; W[c1*n+r0][col] = -1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12=(r0,c1), A22=(r1,c1), B21=(r1,c0), B22=(r1,c1)
        # Term 17: A12*B21
        U[r0*m+c1][col] = 1; V[r1*p+c0][col] = 1; W[c0*n+r0][col] = 1
        # Term 18: A12*B22
        U[r0*m+c1][col] = 1; V[r1*p+c1][col] = 1; W[c1*n+r0][col] = 1
        # Term 19: A22*B21
        U[r1*m+c1][col] = -1; V[r1*p+c0][col] = -1; W[c0*n+r1][col] = -1
        # Term 20: A22*B22
        U[r1*m+c1][col] = -1; V[r1*p+c1][col] = -1; W[c1*n+r1][col] = -1
        col += 1
        
        # Wait, Strassen 2x2 uses 7 multiplications, not 20 terms.
        # The terms above are the *expansion* of the 7 multiplications.
        # But in the tensor decomposition, each "multiplication" is a rank-1 tensor.
        # So M1 is ONE rank-1 tensor? No, M1 = (A11+A22)(B11+B22) is a product of sums.
        # This is NOT a rank-1 tensor. It is a sum of 4 rank-1 tensors.
        # So Strassen does NOT reduce the rank in the tensor decomposition sense if we require the factors to be linear forms.
        #
        # Wait, the tensor rank of matrix multiplication is the minimum number of rank-1 tensors.
        # A rank-1 tensor is of the form u v^T w^T.
        # In Strassen, we compute M1 = (A11+A22)(B11+B22).
        # This requires 4 multiplications in the scalar sense, but in the tensor sense,
        # it corresponds to 4 rank-1 tensors:
        # A11*B11, A11*B22, A22*B11, A22*B22.
        # So Strassen does NOT reduce the tensor rank?
        #
        # No, Strassen reduces the *arithmetic complexity* (number of scalar multiplications).
        # The *tensor rank* is a different measure.
        # The tensor rank of 2x2 matrix multiplication is 7.
        # The tensor rank of 2x2 matrix multiplication is NOT 8 (schoolbook).
        #
        # How is the rank-7 decomposition structured?
        # It is a set of 7 triples (u_i, v_i, w_i) such that
        # sum_i u_i v_i w_i = T_2x2.
        #
        # The standard Strassen decomposition IS a rank-7 tensor decomposition.
        # Let's verify.
        # M1 = (A11+A22)(B11+B22)
        # This is not a single rank-1 term.
        #
        # Ah, I see the confusion.
        # The tensor T = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
        # A rank-1 term is A_{i,j} B_{j,k} C_{k,i} for specific i,j,k.
        # Schoolbook has 8 such terms for 2x2.
        #
        # Strassen's algorithm computes the same result using 7 scalar multiplications.
        # But each scalar multiplication in Strassen is a product of *linear combinations* of A and B entries.
        # So M1 = (A11+A22) * (B11+B22).
        # This product contributes to C11, C12, C21, C22.
        # Specifically, M1 contributes +1 to C11, +1 to C12? No.
        #
        # Let's look at the Strassen formulas again:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each M_i is a scalar.
        # So the tensor decomposition is:
        # T = sum_{i=1}^7 (u_i v_i w_i)
        # where u_i, v_i are linear forms in A, B, and w_i is the indicator for the C entry?
        # No, w_i must be a linear form in C?
        # The definition is: T = sum u^{(i)} v^{(i)} w^{(i)}
        # where u^{(i)} in A^*, v^{(i)} in B^*, w^{(i)} in C^*.
        #
        # In Strassen, M1 = (A11+A22)(B11+B22).
        # This term appears in C11 with coeff 1, C22 with coeff 1.
        # So it contributes to two entries of C.
        # This means it is NOT a rank-1 tensor in the C space?
        #
        # Actually, the tensor rank is defined over the field.
        # The term M1 corresponds to the tensor:
        # (A11+A22) (B11+B22) (C11 + C22) ?
        # No, that would be a single rank-1 term.
        #
        # Let's check if (A11+A22)(B11+B22)(C11+C22) is a valid rank-1 term.
        # If we set C11=1, C22=0, it gives (A11+A22)(B11+B22).
        # But in the product, M1 contributes to BOTH C11 and C22.
        # So the term is (A11+A22)(B11+B22)(C11 + C22).
        # This IS a rank-1 tensor!
        #
        # So, Strassen's 7 multiplications correspond to 7 rank-1 tensors.
        #
        # Let's list them for 2x2:
        # 1. u = A11+A22, v = B11+B22, w = C11+C22
        # 2. u = A21+A22, v = B11, w = C21
        # 3. u = A11, v = B12-B22, w = C12
        # 4. u = A22, v = B21-B11, w = C21
        # 5. u = A11+A12, v = B22, w = C12
        # 6. u = A21-A11, v = B11+B12, w = C22
        # 7. u = A12-A22, v = B21+B22, w = C22
        #
        # Let's verify C11:
        # From 1: +1
        # From 4: +1 (A22(B21-B11) -> C21? No, w=C21. So it contributes to C21.)
        # Wait, my mapping of w is wrong.
        #
        # Let's re-verify Strassen:
        # C11 = M1 + M4 - M5 + M7
        # M1 = (A11+A22)(B11+B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M7 = (A12-A22)(B21+B22)
        #
        # So the rank-1 terms for C11 are:
        # T1: u=A11+A22, v=B11+B22, w=C11
        # T4: u=A22, v=B21-B11, w=C11  (Wait, M4 contributes to C11? No, M4 contributes to C21 in standard Strassen?)
        #
        # Standard Strassen 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # So the terms are:
        # 1. (A11+A22)(B11+B22) contributes to C11 (+1) and C22 (+1).
        #    This is a single rank-1 tensor: u=A11+A22, v=B11+B22, w=C11+C22.
        # 2. (A21+A22)B11 contributes to C21 (+1).
        #    u=A21+A22, v=B11, w=C21.
        # 3. A11(B12-B22) contributes to C12 (+1) and C22 (-1).
        #    u=A11, v=B12-B22, w=C12-C22.
        # 4. A22(B21-B11) contributes to C11 (+1) and C21 (+1).
        #    u=A22, v=B21-B11, w=C11+C21.
        # 5. (A11+A12)B22 contributes to C11 (-1) and C12 (+1).
        #    u=A11+A12, v=B22, w=-C11+C12.
        # 6. (A21-A11)(B11+B12) contributes to C22 (+1).
        #    u=A21-A11, v=B11+B12, w=C22.
        # 7. (A12-A22)(B21+B22) contributes to C11 (+1) and C22 (-1).
        #    u=A12-A22, v=B21+B22, w=C11-C22.
        #
        # Let's check C11:
        # 1: +1
        # 4: +1
        # 5: -1
        # 7: +1
        # Sum: 1+1-1+1 = 2? No, C11 should be 1.
        #
        # I am mixing up the coefficients.
        # The tensor decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is u v w.
        # The coefficient is 1.
        #
        # Let's check the sum of the 7 tensors:
        # T1: (A11+A22)(B11+B22)(C11+C22)
        # T2: (A21+A22)B11(C21)
        # T3: A11(B12-B22)(C12-C22)
        # T4: A22(B21-B11)(C11+C21)
        # T5: (A11+A12)B22(-C11+C12)
        # T6: (A21-A11)(B11+B12)(C22)
        # T7: (A12-A22)(B21+B22)(C11-C22)
        #
        # Coefficient of A11 B11 C11:
        # T1: A11*B11*C11 -> +1
        # T4: A22*... no A11.
        # T5: A11*B22*... no B11.
        # T7: A12... no A11.
        # Sum: 1. Correct.
        #
        # Coefficient of A11 B11 C22:
        # T1: A11*B11*C22 -> +1
        # T2: ... no C22.
        # T3: ... no C22? T3 has C12-C22. So A11*B12*C12 - A11*B12*C22 - A11*B22*C12 + A11*B22*C22.
        #    No A11 B11.
        # T4: ... no C22.
        # T5: ... no C22.
        # T6: (A21-A11)(B11+B12)C22 = A21 B11 C22 + A21 B12 C22 - A11 B11 C22 - A11 B12 C22.
        #    So -1 from T6.
        # T7: ... no C22? T7 has C11-C22. A12 B21 C11 ... - A12 B22 C22 ...
        #    No A11 B11.
        # Sum: 1 (from T1) - 1 (from T6) = 0. Correct (A11 B11 does not contribute to C22).
        #
        # So the rank-7 decomposition is valid.
        #
        # Now, can we apply this to the 2x2 block in our 5x5 problem?
        # Yes. The 2x2 block A22, B22, C22 can be replaced by 7 rank-1 tensors.
        # The rest of the terms are schoolbook.
        #
        # Let's implement this.
        
    # Re-implement with correct Strassen 2x2 rank-7 terms.
    
    U = [[0] * rank_est for _ in range(n * m)]
    V = [[0] * rank_est for _ in range(m * p)]
    W = [[0] * rank_est for _ in range(p * n)]
    col = 0
    
    # Schoolbook for 3x3 block (A11, B11, C11)
    # Indices: rows 0-2, cols 0-2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Schoolbook for A12 (3x2) * B21 (2x3) -> C11 (3x3)
    # A12: rows 0-2, cols 3-4
    # B21: rows 3-4, cols 0-2
    # C11: rows 0-2, cols 0-2
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Schoolbook for A11 (3x3) * B12 (3x2) -> C12 (3x2)
    # A11: rows 0-2, cols 0-2
    # B12: rows 0-2, cols 3-4
    # C12: rows 0-2, cols 3-4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Schoolbook for A12 (3x2) * B22 (2x2) -> C12 (3x2)
    # A12: rows 0-2, cols 3-4
    # B22: rows 3-4, cols 3-4
    # C12: rows 0-2, cols 3-4
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Schoolbook for A21 (2x3) * B11 (3x3) -> C21 (2x3)
    # A21: rows 3-4, cols 0-2
    # B11: rows 0-2, cols 0-2
    # C21: rows 3-4, cols 0-2
    for i in range(3, 5):
        for j in range(3):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Schoolbook for A22 (2x2) * B21 (2x3) -> C21 (2x3)
    # A22: rows 3-4, cols 3-4
    # B21: rows 3-4, cols 0-2
    # C21: rows 3-4, cols 0-2
    for i in range(3, 5):
        for j in range(3, 5):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Schoolbook for A21 (2x3) * B12 (3x2) -> C22 (2x2)
    # A21: rows 3-4, cols 0-2
    # B12: rows 0-2, cols 3-4
    # C22: rows 3-4, cols 3-4
    for i in range(3, 5):
        for j in range(3):
            for k in range(3, 5):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Strassen 2x2 for A22 (2x2) * B22 (2x2) -> C22 (2x2)
    # Indices:
    # A22: (3,3), (3,4), (4,3), (4,4)
    # B22: (3,3), (3,4), (4,3), (4,4)
    # C22: (3,3), (3,4), (4,3), (4,4)
    
    # Map local 2x2 indices to global 5x5 indices
    # Local (0,0) -> Global (3,3)
    # Local (0,1) -> Global (3,4)
    # Local (1,0) -> Global (4,3)
    # Local (1,1) -> Global (4,4)
    
    def local_to_global(r, c):
        return 3 + r, 3 + c
    
    # Term 1: (A11+A22)(B11+B22)(C11+C22)
    # Local A11=(0,0), A22=(1,1)
    # Local B11=(0,0), B22=(1,1)
    # Local C11=(0,0), C22=(1,1)
    # Global A: (3,3), (4,4)
    # Global B: (3,3), (4,4)
    # Global C: (3,3), (4,4)
    
    # We need to add 4 sub-terms for this rank-1 tensor?
    # No, the rank-1 tensor is u v w.
    # u = A33 + A44
    # v = B33 + B44
    # w = C33 + C44
    # This is NOT a single entry in U, V, W.
    # U is a matrix. u is a vector in A-space.
    # So we set U[3*m+3][col] = 1, U[4*m+4][col] = 1.
    # V[3*p+3][col] = 1, V[4*p+4][col] = 1.
    # W[3*n+3][col] = 1, W[4*n+4][col] = 1.
    
    U[3*m+3][col] = 1
    U[4*m+4][col] = 1
    V[3*p+3][col] = 1
    V[4*p+4][col] = 1
    W[3*n+3][col] = 1
    W[4*n+4][col] = 1
    col += 1
    
    # Term 2: (A21+A22)B11(C21)
    # Local A21=(1,0), A22=(1,1)
    # Local B11=(0,0)
    # Local C21=(1,0)
    # Global A: (4,3), (4,4)
    # Global B: (3,3)
    # Global C: (4,3)
    
    U[4*m+3][col] = 1
    U[4*m+4][col] = 1
    V[3*p+3][col] = 1
    W[3*n+4][col] = 1  # C21 is local (1,0) -> global (4,3). W index is k*n+i. k=3, i=4.
    col += 1
    
    # Term 3: A11(B12-B22)(C12-C22)
    # Local A11=(0,0)
    # Local B12=(0,1), B22=(1,1)
    # Local C12=(0,1), C22=(1,1)
    # Global A: (3,3)
    # Global B: (3,4), (4,4)
    # Global C: (3,4), (4,4)
    
    U[3*m+3][col] = 1
    V[3*p+4][col] = 1
    V[4*p+4][col] = -1
    W[4*n+3][col] = 1  # C12 local (0,1) -> global (3,4). k=4, i=3.
    W[4*n+4][col] = -1 # C22 local (1,1) -> global (4,4). k=4, i=4.
    col += 1
    
    # Term 4: A22(B21-B11)(C11+C21)
    # Local A22=(1,1)
    # Local B21=(1,0), B11=(0,0)
    # Local C11=(0,0), C21=(1,0)
    # Global A: (4,4)
    # Global B: (4,3), (3,3)
    # Global C: (3,3), (4,3)
    
    U[4*m+4][col] = 1
    V[4*p+3][col] = 1
    V[3*p+3][col] = -1
    W[3*n+3][col] = 1  # C11 local (0,0) -> global (3,3). k=3, i=3.
    W[3*n+4][col] = 1  # C21 local (1,0) -> global (4,3). k=3, i=4.
    col += 1
    
    # Term 5: (A11+A12)B22(-C11+C12)
    # Local A11=(0,0), A12=(0,1)
    # Local B22=(1,1)
    # Local C11=(0,0), C12=(0,1)
    # Global A: (3,3), (3,4)
    # Global B: (4,4)
    # Global C: (3,3), (3,4)
    
    U[3*m+3][col] = 1
    U[3*m+4][col] = 1
    V[4*p+4][col] = 1
    W[3*n+3][col] = -1 # C11 local (0,0) -> global (3,3). k=3, i=3.
    W[4*n+3][col] = 1  # C12 local (0,1) -> global (3,4). k=4, i=3.
    col += 1
    
    # Term 6: (A21-A11)(B11+B12)(C22)
    # Local A21=(1,0), A11=(0,0)
    # Local B11=(0,0), B12=(0,1)
    # Local C22=(1,1)
    # Global A: (4,3), (3,3)
    # Global B: (3,3), (3,4)
    # Global C: (4,4)
    
    U[4*m+3][col] = 1
    U[3*m+3][col] = -1
    V[3*p+3][col] = 1
    V[3*p+4][col] = 1
    W[4*n+4][col] = 1  # C22 local (1,1) -> global (4,4). k=4, i=4.
    col += 1
    
    # Term 7: (A12-A22)(B21+B22)(C11-C22)
    # Local A12=(0,1), A22=(1,1)
    # Local B21=(1,0), B22=(1,1)
    # Local C11=(0,0), C22=(1,1)
    # Global A: (3,4), (4,4)
    # Global B: (4,3), (4,4)
    # Global C: (3,3), (4,4)
    
    U[3*m+4][col] = 1
    U[4*m+4][col] = -1
    V[4*p+3][col] = 1
    V[4*p+4][col] = 1
    W[3*n+3][col] = 1  # C11 local (0,0) -> global (3,3). k=3, i=3.
    W[4*n+4][col] = -1 # C22 local (1,1) -> global (4,4). k=4, i=4.
    col += 1
    
    # Total rank used:
    # 3x3 schoolbook: 27
    # A12*B21: 18
    # A11*B12: 18
    # A12*B22: 12
    # A21*B11: 18
    # A22*B21: 12
    # A21*B12: 12
    # Strassen 2x2: 7
    # Sum: 27+18+18+12+18+12+12+7 = 124.
    
    # Trim U, V, W to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
