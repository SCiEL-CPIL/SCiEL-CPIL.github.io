"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Trivial rank-48 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
    
    # Strategy: Use a structured block decomposition.
    # We partition the matrices into blocks:
    # A (2x4) = [A11 A12; A21 A22] where each block is 1x2
    # B (4x6) = [B11 B12; B13 B14] where each block is 2x3
    #
    # We want to compute C = A * B (2x6)
    # C = [A11 B11 + A12 B13, A11 B12 + A12 B14;
    #      A21 B11 + A22 B13, A21 B12 + A22 B14]
    #
    # Let's try to use a Strassen-like approach on the 2x2 block structure.
    # Standard Strassen for 2x2 matrices uses 7 multiplications.
    # Here, the "elements" are rectangular matrices.
    # Let X1 = A11, X2 = A22 (1x2)
    # Let Y1 = B11, Y2 = B22 (2x3)
    #
    # Strassen formulas for 2x2 block multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a matrix product of size (1x2) * (2x3) = 1x3.
    # The rank of a single matrix product of size 1x2 * 2x3 is 6 (trivial).
    # However, we are looking for the tensor rank of the whole operation.
    # The total rank would be 7 * 6 = 42.
    # This is better than 48.
    
    # Let's construct the U, V, W matrices for this rank-42 decomposition.
    # The indices for U, V, W correspond to the "scalar" multiplications inside the block products.
    
    # Block definitions:
    # A blocks (indices i, j):
    # A11: i=0, j=0,1
    # A12: i=0, j=2,3
    # A21: i=1, j=0,1
    # A22: i=1, j=2,3
    #
    # B blocks (indices j, k):
    # B11: j=0,1, k=0,1,2
    # B12: j=0,1, k=3,4,5
    # B21: j=2,3, k=0,1,2
    # B22: j=2,3, k=3,4,5
    
    # We define 7 "Strassen" terms.
    # Each term is a product of a linear combination of A elements and a linear combination of B elements.
    
    # Term 1: (A11 + A22) * (B11 + B22)
    # Term 2: (A21 + A22) * B11
    # Term 3: A11 * (B12 - B22)
    # Term 4: A22 * (B21 - B11)
    # Term 5: (A11 + A12) * B22
    # Term 6: (A21 - A11) * (B11 + B12)
    # Term 7: (A12 - A22) * (B21 + B22)
    
    # For each term t, we have a sub-tensor of rank 6 (since it's a 1x2 * 2x3 product).
    # We can use the trivial decomposition for each sub-product.
    
    rank_total = 42
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Helper to add a trivial rank-6 decomposition for a specific (i_block, j_block, k_block) product
    # where i_block is 0 or 1 (row of A), j_block is 0,1,2,3 (col of A / row of B), k_block is 0,1,2,3,4,5 (col of B)
    # But wait, the "elements" in Strassen are sums/differences.
    # We need to map the coefficients to U and V.
    
    # Let's define the coefficients for A and B for each Strassen term.
    # A coefficients: shape (2, 4) for each term (sparse)
    # B coefficients: shape (4, 6) for each term (sparse)
    
    # Term 1: A11 + A22
    # A11: (0,0)=1, (0,1)=1
    # A22: (1,2)=1, (1,3)=1
    # B11 + B22:
    # B11: (0,0..2)=1, (1,0..2)=1
    # B22: (2,3..5)=1, (3,3..5)=1
    
    # Let's explicitly build the U, V, W entries.
    # For each term t (0..6), and for each "scalar" multiplication in the 1x2*2x3 product (6 of them),
    # we assign a rank index r = t*6 + s.
    
    # The 1x2 * 2x3 product is:
    # [a0, a1] * [[b00, b01, b02], [b10, b11, b12]]
    # = [a0*b00 + a1*b10, a0*b01 + a1*b11, a0*b02 + a1*b12]
    
    # The trivial decomposition for this specific 1x2*2x3 product has rank 6.
    # The 6 multiplications are:
    # a0*b00, a0*b01, a0*b02
    # a1*b10, a1*b11, a1*b12
    
    # We need to map the "a" and "b" values to the global A and B matrices using the Strassen coefficients.
    
    # Define Strassen coefficients for A (2x4) and B (4x6)
    # A_coeffs[t][i][j]
    # B_coeffs[t][j][k]
    
    A_coeffs = []
    B_coeffs = []
    
    # Helper to create a 2x4 matrix of zeros
    def zero_A():
        return np.zeros((2, 4))
    
    def zero_B():
        return np.zeros((4, 6))
    
    # T1: (A11 + A22) * (B11 + B22)
    A1 = zero_A()
    A1[0,0] = 1; A1[0,1] = 1
    A1[1,2] = 1; A1[1,3] = 1
    B1 = zero_B()
    B1[0,0:3] = 1; B1[1,0:3] = 1
    B1[2,3:6] = 1; B1[3,3:6] = 1
    A_coeffs.append(A1)
    B_coeffs.append(B1)
    
    # T2: (A21 + A22) * B11
    A2 = zero_A()
    A2[1,0] = 1; A2[1,1] = 1
    A2[1,2] = 1; A2[1,3] = 1
    B2 = zero_B()
    B2[0,0:3] = 1; B2[1,0:3] = 1
    A_coeffs.append(A2)
    B_coeffs.append(B2)
    
    # T3: A11 * (B12 - B22)
    A3 = zero_A()
    A3[0,0] = 1; A3[0,1] = 1
    B3 = zero_B()
    B3[0,3:6] = 1; B3[1,3:6] = 1
    B3[2,3:6] = -1; B3[3,3:6] = -1
    A_coeffs.append(A3)
    B_coeffs.append(B3)
    
    # T4: A22 * (B21 - B11)
    A4 = zero_A()
    A4[1,2] = 1; A4[1,3] = 1
    B4 = zero_B()
    B4[2,0:3] = 1; B4[3,0:3] = 1
    B4[0,0:3] = -1; B4[1,0:3] = -1
    A_coeffs.append(A4)
    B_coeffs.append(B4)
    
    # T5: (A11 + A12) * B22
    A5 = zero_A()
    A5[0,0] = 1; A5[0,1] = 1
    A5[0,2] = 1; A5[0,3] = 1
    B5 = zero_B()
    B5[2,3:6] = 1; B5[3,3:6] = 1
    A_coeffs.append(A5)
    B_coeffs.append(B5)
    
    # T6: (A21 - A11) * (B11 + B12)
    A6 = zero_A()
    A6[1,0] = 1; A6[1,1] = 1
    A6[0,0] = -1; A6[0,1] = -1
    B6 = zero_B()
    B6[0,0:6] = 1; B6[1,0:6] = 1
    A_coeffs.append(A6)
    B_coeffs.append(B6)
    
    # T7: (A12 - A22) * (B21 + B22)
    A7 = zero_A()
    A7[0,2] = 1; A7[0,3] = 1
    A7[1,2] = -1; A7[1,3] = -1
    B7 = zero_B()
    B7[2,0:6] = 1; B7[3,0:6] = 1
    A_coeffs.append(A7)
    B_coeffs.append(B7)
    
    # Now, for each term t, we decompose the product of the 1x2 "A-vector" and 2x3 "B-matrix" into 6 scalar multiplications.
    # The "A-vector" for term t is effectively the row of A_coeffs[t] corresponding to the output row?
    # Wait, the Strassen terms are block products.
    # Term 1: (A11+A22) is a 2x2 block (rows 0,1 of A, cols 0,1,2,3? No, A11 is 1x2, A22 is 1x2. They stack vertically to form a 2x2 block?
    # No, A is 2x4. A11 is A[0,0:2], A12 is A[0,2:4], A21 is A[1,0:2], A22 is A[1,2:4].
    # A11 + A22 is NOT a matrix addition in the standard sense if they are different rows.
    # Strassen works on 2x2 block matrices where blocks are themselves matrices.
    # Here, A is partitioned into 4 blocks of size 1x2.
    # B is partitioned into 4 blocks of size 2x3.
    # The product C is 2x6, partitioned into 4 blocks of size 1x3.
    # C11 (1x3) = A11(1x2) * B11(2x3) + A12(1x2) * B13(2x3) ?? No.
    # C11 is C[0, 0:3].
    # C[0, k] = sum_j A[0, j] B[j, k].
    # For k in 0..2, C[0, k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k].
    # This is (A11 * B11) + (A12 * B21).
    # Let's re-verify the block structure.
    # A = [A11 A12]
    #     [A21 A22]
    # B = [B11 B12]
    #     [B13 B14]
    # Note: Standard block notation:
    # A11 is top-left, A12 top-right.
    # A21 bottom-left, A22 bottom-right.
    # B11 top-left, B12 top-right.
    # B13 bottom-left, B14 bottom-right.
    #
    # C11 = A11 B11 + A12 B13
    # C12 = A11 B12 + A12 B14
    # C21 = A21 B11 + A22 B13
    # C22 = A21 B12 + A22 B14
    #
    # My previous Strassen mapping used B21 and B22 which I mapped to B13 and B14.
    # Let's check the Strassen formulas again with correct indices.
    # M1 = (A11 + A22)(B11 + B14)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B14)
    # M4 = A22(B13 - B11)
    # M5 = (A11 + A12)B14
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B13 + B14)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify C11:
    # M1 = A11B11 + A11B14 + A22B11 + A22B14
    # M4 = A22B13 - A22B11
    # -M5 = -A11B14 - A12B14
    # M7 = A12B13 + A12B14 - A22B13 - A22B14
    # Sum:
    # A11B11 + A11B14 + A22B11 + A22B14 + A22B13 - A22B11 - A11B14 - A12B14 + A12B13 + A12B14 - A22B13 - A22B14
    # = A11B11 + A12B13
    # Correct.
    
    # So the coefficients I defined earlier need to be updated to match B13 and B14 instead of B21, B22 (which were just labels).
    # My previous B1, B2... definitions:
    # B1 (for M1): B11 + B14. B11 is rows 0,1 cols 0,1,2. B14 is rows 2,3 cols 3,4,5.
    # My B1 definition:
    # B1[0,0:3]=1, B1[1,0:3]=1 (B11)
    # B1[2,3:6]=1, B1[3,3:6]=1 (B14)
    # This matches.
    
    # B2 (for M2): B11.
    # B2[0,0:3]=1, B2[1,0:3]=1. Matches.
    
    # B3 (for M3): B12 - B14.
    # B12 is rows 0,1 cols 3,4,5. B14 is rows 2,3 cols 3,4,5.
    # My B3 definition:
    # B3[0,3:6]=1, B3[1,3:6]=1 (B12)
    # B3[2,3:6]=-1, B3[3,3:6]=-1 (B14)
    # Matches.
    
    # B4 (for M4): B13 - B11.
    # B13 is rows 2,3 cols 0,1,2. B11 is rows 0,1 cols 0,1,2.
    # My B4 definition:
    # B4[2,0:3]=1, B4[3,0:3]=1 (B13)
    # B4[0,0:3]=-1, B4[1,0:3]=-1 (B11)
    # Matches.
    
    # B5 (for M5): B14.
    # My B5 definition:
    # B5[2,3:6]=1, B5[3,3:6]=1. Matches.
    
    # B6 (for M6): B11 + B12.
    # B11 rows 0,1 cols 0,1,2. B12 rows 0,1 cols 3,4,5.
    # My B6 definition:
    # B6[0,0:6]=1, B6[1,0:6]=1. Matches.
    
    # B7 (for M7): B13 + B14.
    # B13 rows 2,3 cols 0,1,2. B14 rows 2,3 cols 3,4,5.
    # My B7 definition:
    # B7[2,0:6]=1, B7[3,0:6]=1. Matches.
    
    # A coefficients:
    # A1 (M1): A11 + A22.
    # A11: row 0, cols 0,1. A22: row 1, cols 2,3.
    # My A1:
    # A1[0,0]=1, A1[0,1]=1 (A11)
    # A1[1,2]=1, A1[1,3]=1 (A22)
    # Matches.
    
    # A2 (M2): A21 + A22.
    # A21: row 1, cols 0,1. A22: row 1, cols 2,3.
    # My A2:
    # A2[1,0]=1, A2[1,1]=1 (A21)
    # A2[1,2]=1, A2[1,3]=1 (A22)
    # Matches.
    
    # A3 (M3): A11.
    # My A3:
    # A3[0,0]=1, A3[0,1]=1. Matches.
    
    # A4 (M4): A22.
    # My A4:
    # A4[1,2]=1, A4[1,3]=1. Matches.
    
    # A5 (M5): A11 + A12.
    # A11: row 0, cols 0,1. A12: row 0, cols 2,3.
    # My A5:
    # A5[0,0]=1, A5[0,1]=1 (A11)
    # A5[0,2]=1, A5[0,3]=1 (A12)
    # Matches.
    
    # A6 (M6): A21 - A11.
    # A21: row 1, cols 0,1. A11: row 0, cols 0,1.
    # My A6:
    # A6[1,0]=1, A6[1,1]=1 (A21)
    # A6[0,0]=-1, A6[0,1]=-1 (A11)
    # Matches.
    
    # A7 (M7): A12 - A22.
    # A12: row 0, cols 2,3. A22: row 1, cols 2,3.
    # My A7:
    # A7[0,2]=1, A7[0,3]=1 (A12)
    # A7[1,2]=-1, A7[1,3]=-1 (A22)
    # Matches.
    
    # Now, for each term t, we have A_coeffs[t] (2x4) and B_coeffs[t] (4x6).
    # We want to compute the product C_t = A_coeffs[t] * B_coeffs[t].
    # This product is a 2x6 matrix.
    # However, in the Strassen algorithm, we are treating the blocks as scalars?
    # No, we are treating the block multiplication as a scalar multiplication of the "block values".
    # But the block values are matrices.
    # The rank of the tensor for multiplying a 1x2 matrix by a 2x3 matrix is 6.
    # So for each Strassen term, we need 6 rank-1 tensors.
    # Total rank = 7 * 6 = 42.
    
    # For each term t, let's decompose the product A_t * B_t into 6 rank-1 tensors.
    # A_t is 2x4. B_t is 4x6.
    # Wait, the Strassen term is a product of two "block sums".
    # The "block sum" for A is a 2x4 matrix (with many zeros).
    # The "block sum" for B is a 4x6 matrix (with many zeros).
    # Their product is a 2x6 matrix.
    # We need to decompose this specific 2x6 matrix product into rank-1 tensors?
    # No, we are decomposing the *operation* of matrix multiplication.
    # The operation is: given A and B, compute C.
    # The Strassen algorithm says:
    # C = M1 + ... - M7 (with signs).
    # Each Mi is a matrix product of two matrices derived from A and B.
    # Mi = Ai * Bi.
    # Ai is a 2x4 matrix (linear combo of A).
    # Bi is a 4x6 matrix (linear combo of B).
    # The product Ai * Bi is a 2x6 matrix.
    # We need to express the entries of Ai * Bi as a sum of rank-1 tensors.
    # The rank of the matrix multiplication tensor for 2x4 * 4x6 is 48 (trivial).
    # But here Ai and Bi are sparse!
    # Specifically, for each term t, Ai and Bi have a specific sparsity pattern.
    # We can use the trivial decomposition for the non-zero parts.
    
    # Let's analyze the sparsity of A_t and B_t.
    # For T1:
    # A1 has non-zeros at (0,0), (0,1), (1,2), (1,3).
    # B1 has non-zeros at (0,0..2), (1,0..2), (2,3..5), (3,3..5).
    # The product C1 = A1 * B1.
    # C1[i, k] = sum_j A1[i, j] B1[j, k].
    
    # We can decompose this product into rank-1 tensors corresponding to the non-zero paths i->j->k.
    # Number of non-zero (i, j) pairs in A1 is 4.
    # Number of non-zero (j, k) pairs in B1 is 12 (4 rows * 3 cols).
    # But we must match j.
    # For each j, if A1 has non-zeros in row i and B1 has non-zeros in col k for that row j, we have a term.
    
    # Let's just iterate over all i, j, k where A1[i,j] != 0 and B1[j,k] != 0.
    # Each such triplet (i, j, k) contributes a rank-1 tensor to the decomposition of the product for term t.
    # The coefficient for U is A1[i,j], for V is B1[j,k].
    # The output W is 1 at (k, i).
    
    # Let's count the rank for each term.
    # T1:
    # A1 non-zeros: (0,0), (0,1), (1,2), (1,3).
    # B1 non-zeros:
    # Row 0: (0,0), (0,1), (0,2) -> matches A1[0,0]
    # Row 1: (1,0), (1,1), (1,2) -> matches A1[0,1]
    # Row 2: (2,3), (2,4), (2,5) -> matches A1[1,2]
    # Row 3: (3,3), (3,4), (3,5) -> matches A1[1,3]
    # Total non-zero triplets: 4 * 3 = 12.
    # So T1 has rank 12.
    
    # T2:
    # A2 non-zeros: (1,0), (1,1), (1,2), (1,3).
    # B2 non-zeros:
    # Row 0: (0,0..2) -> matches A2[1,0]
    # Row 1: (1,0..2) -> matches A2[1,1]
    # Row 2: None -> A2[1,2] is non-zero but B2[2,k] is 0.
    # Row 3: None -> A2[1,3] is non-zero but B2[3,k] is 0.
    # Total non-zero triplets: 2 * 3 = 6.
    # So T2 has rank 6.
    
    # T3:
    # A3 non-zeros: (0,0), (0,1).
    # B3 non-zeros:
    # Row 0: (0,3..5) -> matches A3[0,0]
    # Row 1: (1,3..5) -> matches A3[0,1]
    # Row 2: (2,3..5) with -1 -> A3[0,2] is 0.
    # Row 3: (3,3..5) with -1 -> A3[0,3] is 0.
    # Total non-zero triplets: 2 * 3 = 6.
    # So T3 has rank 6.
    
    # T4:
    # A4 non-zeros: (1,2), (1,3).
    # B4 non-zeros:
    # Row 0: (0,0..2) with -1 -> A4[1,0] is 0.
    # Row 1: (1,0..2) with -1 -> A4[1,1] is 0.
    # Row 2: (2,0..2) -> matches A4[1,2]
    # Row 3: (3,0..2) -> matches A4[1,3]
    # Total non-zero triplets: 2 * 3 = 6.
    # So T4 has rank 6.
    
    # T5:
    # A5 non-zeros: (0,0), (0,1), (0,2), (0,3).
    # B5 non-zeros:
    # Row 0: None
    # Row 1: None
    # Row 2: (2,3..5) -> matches A5[0,2]
    # Row 3: (3,3..5) -> matches A5[0,3]
    # Total non-zero triplets: 2 * 3 = 6.
    # So T5 has rank 6.
    
    # T6:
    # A6 non-zeros: (1,0), (1,1), (0,0), (0,1).
    # B6 non-zeros:
    # Row 0: (0,0..5) -> matches A6[1,0] and A6[0,0]
    # Row 1: (1,0..5) -> matches A6[1,1] and A6[0,1]
    # Row 2: None
    # Row 3: None
    # Total non-zero triplets: 4 * 3 = 12.
    # So T6 has rank 12.
    
    # T7:
    # A7 non-zeros: (0,2), (0,3), (1,2), (1,3).
    # B7 non-zeros:
    # Row 0: None
    # Row 1: None
    # Row 2: (2,0..5) -> matches A7[0,2] and A7[1,2]
    # Row 3: (3,0..5) -> matches A7[0,3] and A7[1,3]
    # Total non-zero triplets: 4 * 3 = 12.
    # So T7 has rank 12.
    
    # Total rank = 12 + 6 + 6 + 6 + 6 + 12 + 12 = 60.
    # This is worse than 48!
    
    # Why? Because I am counting the rank of the *product* of the sparse matrices.
    # But the Strassen algorithm reduces the number of *block* multiplications from 8 to 7.
    # Each block multiplication is a 1x2 * 2x3 product, which has rank 6.
    # So 7 * 6 = 42.
    # My calculation above gave 60. Why the discrepancy?
    
    # Ah, in T1, I calculated rank 12.
    # But the block product A11*B11 is 1x2 * 2x3. Rank 6.
    # A11*B14 is 1x2 * 2x3. Rank 6.
    # A22*B11 is 1x2 * 2x3. Rank 6.
    # A22*B14 is 1x2 * 2x3. Rank 6.
    # Total 24?
    # But Strassen says M1 = (A11+A22)(B11+B14).
    # This is ONE multiplication of two 2x2 block matrices?
    # No, A11+A22 is a 2x2 matrix (if we view blocks as scalars).
    # But here the blocks are matrices.
    # (A11+A22) is a 2x4 matrix.
    # (B11+B14) is a 4x6 matrix.
    # Their product is a 2x6 matrix.
    # The rank of this product is the rank of the tensor for multiplying a 2x4 by 4x6 matrix, restricted to the support of A1 and B1.
    # My count of 12 for T1 seems correct for the restricted support.
    
    # So the total rank is 60.
    # This is not an improvement over 48.
    
    # Is there a better block partition?
    # What if we partition into 2x2 blocks of size 1x2 and 2x3?
    # We did that.
    
    # What if we use a different algorithm?
    # Maybe the trivial rank 48 is actually minimal for 2x4 * 4x6?
    # Let's check the lower bounds.
    # The rank of matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n).
    # Here max(8, 24, 12) = 24.
    # So rank >= 24.
    # 48 is not necessarily minimal.
    
    # Let's try a different block size.
    # Partition A (2x4) into two 1x4 blocks? No.
    # Partition A into 2 blocks of 1x4?
    # Partition B into 2 blocks of 4x3?
    # Then we have 2x2 block structure.
    # A11 (1x4), A12 (1x4)
    # A21 (1x4), A22 (1x4)
    # B11 (4x3), B12 (4x3)
    # B13 (4x3), B14 (4x3)
    #
    # Strassen terms:
    # M1 = (A11+A22)(B11+B14)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B14)
    # M4 = A22(B13-B11)
    # M5 = (A11+A12)B14
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B13+B14)
    #
    # Each term is a product of a 1x4 matrix and a 4x3 matrix.
    # Rank of 1x4 * 4x3 product is 12 (trivial).
    # Total rank 7 * 12 = 84. Worse.
    
    # What if we partition B into 4x6 blocks?
    # A into 2x6? No.
    
    # It seems the 2x2 block partition with 1x2 and 2x3 blocks is the best structured approach, but it yields rank 60.
    
    # Let's go back to the trivial rank 48.
    # Can we do better than 48 with a non-block approach?
    # Maybe a random search with a better heuristic?
    # Or maybe the rank 48 is the best we can easily find.
    
    # Given the time constraint, I will return the trivial rank 48 decomposition.
    # It is guaranteed to be correct.
    
    result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
