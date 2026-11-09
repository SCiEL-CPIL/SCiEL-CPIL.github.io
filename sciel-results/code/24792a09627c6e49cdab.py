import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    n, m, p = DIMENSIONS  # n=3, m=4, p=5

    # We want to find a low-rank decomposition of the matrix multiplication tensor
    # The tensor is defined as: T[i, j, k] = 1 if and only if the (i,k) entry of A*B uses
    # A[i,j] and B[j,k], which is always true for standard matrix multiplication.
    # Actually, the standard matrix multiplication tensor T[i, j, k] = 1 for all valid i,j,k.
    # Wait, let's re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix."
    # The tensor T has dimensions (n, m, p) and T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right. The tensor for matrix multiplication (A*B)_{ik} = sum_j A_{ij} B_{jk}.
    # The trilinear form is sum_{i,j,k} (A*B)_{ik} x A_{ij} y B_{jk} ? No.
    # The standard tensor is T[i, j, k] = 1 if we are summing over j.
    # Let's check the indices in the seed program.
    # U[i * m + j, r], V[j * p + k, r], W[k * n + i, r]
    # The product is sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This product should equal T[i, j, k].
    # For standard matrix multiplication, (A*B)_{ik} = sum_j A_{ij} B_{jk}.
    # The term A_{ij} B_{jk} contributes to (A*B)_{ik}.
    # So the tensor T[i, j, k] should be 1 for all i, j, k?
    # If T[i, j, k] = 1, then sum_r U V W = 1.
    # In the schoolbook method, for each (i, j, k), we have a term r = (i, j, k).
    # U[i, j, r] = 1, V[j, k, r] = 1, W[k, i, r] = 1.
    # Sum over r: Only one term is non-zero for a specific (i, j, k) in the schoolbook decomposition?
    # No, in the schoolbook decomposition, the term r corresponds to (i, j, k).
    # So for a fixed (i, j, k), only the term r=(i, j, k) has U,V,W all 1.
    # Thus sum_r ... = 1.
    # So the target tensor T[i, j, k] = 1 for all i, j, k.
    
    # Let's verify this interpretation.
    # If T[i, j, k] = 1, then the rank is at least max(n*m, m*p, p*n) = max(12, 20, 15) = 20?
    # No, the rank is the number of terms.
    # The schoolbook rank is n*m*p = 60.
    # Can we do better?
    # The matrix multiplication tensor is known to have rank less than n*m*p for some sizes.
    # For 2x2x2, Strassen's algorithm has rank 7 < 8.
    # For 3x3x3, the best known is 23 < 27.
    # For 3x4x5, we need to find a good decomposition.
    
    # The prompt says: "Find a rank-minimal exact tensor decomposition... The output factor follows the official trace convention W[k*n+i,r]."
    # This confirms the indices: U[i*m+j], V[j*p+k], W[k*n+i].
    # And the target is indeed T[i,j,k] = 1.
    
    # Strategy:
    # 1. Try to construct a decomposition using known fast matrix multiplication algorithms or block decomposition.
    # 2. Since we can't easily search for the minimal rank (it's an NP-hard problem in general), we should try to find a good upper bound.
    # 3. A simple upper bound is the schoolbook rank 60.
    # 4. We can try block decomposition.
    #    Split the matrices into blocks.
    #    A (3x4), B (4x5).
    #    Split B into two parts: B1 (4x3) and B2 (4x2).
    #    C = A * B = A * B1 + A * B2.
    #    A*B1 is (3x3). Schoolbook rank 9.
    #    A*B2 is (3x2). Schoolbook rank 6.
    #    Total rank 15.
    #    Wait, is this valid?
    #    The terms for A*B1 and A*B2 are separate.
    #    For A*B1: terms (i, j, k1) where k1 in 0..2.
    #    For A*B2: terms (i, j, k2) where k2 in 0..1 (mapped to global k=3,4).
    #    The tensor T[i, j, k] = 1 for all k.
    #    So summing the two decompositions:
    #    For a fixed (i, j), the sum over k is 1 for each k.
    #    The decomposition for A*B1 gives 1 for k=0,1,2.
    #    The decomposition for A*B2 gives 1 for k=3,4.
    #    So the total sum is 1 for all k.
    #    This works!
    #    Rank = 3*4*3 + 3*4*2 = 36 + 24 = 60?
    #    No, rank of A*B1 (3x4 * 4x3) is 3*4*3 = 36?
    #    Schoolbook rank for n x m * m x p is n*m*p.
    #    So 3*4*3 = 36.
    #    3*4*2 = 24.
    #    Total 60. Same as schoolbook.
    
    # Why? Because splitting the output dimension doesn't reduce the rank in the schoolbook method.
    # We need to use the structure of matrix multiplication to reduce rank.
    # Strassen's algorithm works for 2x2.
    # Can we use Strassen on blocks?
    # A (3x4) can be split into [A1 (2x4), A2 (1x4)]? No, Strassen needs square blocks.
    # We can pad to 4x4 or 5x5?
    # If we pad A to 4x4 (by adding zeros) and B to 4x4 (by adding zeros),
    # then we can use Strassen's 7-term algorithm for the 4x4 multiplication?
    # No, Strassen is for 2x2.
    # We can use the 7-term algorithm for 2x2 blocks.
    # A 4x4 matrix can be split into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B has 4 2x2 blocks.
    # Each 2x2 block multiplication can be done with 7 terms.
    # But the blocks share the input matrices.
    # Strassen's algorithm computes 7 products M1..M7 of linear combinations of the blocks.
    # Then the output blocks are linear combinations of these.
    # This is a rank 7 decomposition for 2x2.
    # For 4x4, the rank is 7^2 = 49?
    # Or is it 7 * (number of block interactions)?
    # Actually, the rank of 4x4 matrix multiplication is known to be 49 (from recursive Strassen).
    # But we have 3x4 and 4x5.
    # We can pad A to 4x4 and B to 4x4.
    # Then compute C (4x4).
    # Then extract the 3x4 part of C.
    # The rank of the 4x4 multiplication using Strassen is 49.
    # But we need to handle the padding.
    # The padding adds terms?
    # If we use the same 49 terms, they compute the full 4x4 product.
    # The 3x4 product is a subset of the entries.
    # So the rank should be 49.
    # 49 < 60.
    
    # Let's try to implement Strassen's algorithm for 4x4.
    # We need to define the 49 terms.
    # Each term corresponds to a product of a linear combination of A blocks and a linear combination of B blocks.
    # And then the output C blocks are linear combinations of these products.
    # This fits the tensor decomposition model:
    # U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Wait, the model is U(i,j) * V(j,k) * W(k,i).
    # This is exactly the structure of the tensor decomposition.
    # So if we have a rank R decomposition, we can write down U, V, W.
    
    # How to get the 49 terms?
    # Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we apply this to the 2x2 block structure.
    # A is 4x4, split into 2x2 blocks.
    # B is 4x4, split into 2x2 blocks.
    # The 7 products M1..M7 are themselves 2x2 matrix multiplications.
    # Each of these can be computed with 7 terms (Strassen again).
    # So total terms = 7 * 7 = 49.
    
    # Let's construct the U, V, W matrices for this.
    # This is complex but doable.
    # We need to track the indices carefully.
    
    # Alternative: Use a known low-rank decomposition for 3x4x5 or similar sizes.
    # Or, use a greedy algorithm to merge terms?
    # The current code tries to remove terms, which is unlikely to work.
    
    # Let's try the 4x4 Strassen approach.
    # Rank 49 is a significant improvement over 60.
    
    # We need to generate the U, V, W matrices for the 4x4 Strassen decomposition.
    # Then, we need to map them to the 3x4x5 problem.
    # A is 3x4. We pad it to 4x4 by adding a row of zeros? Or column?
    # A is (3x4). Pad to (4x4) by adding a zero row at the bottom.
    # B is (4x5). Pad to (4x4) by adding a zero column on the right.
    # C is (4x4). We take the top-left 3x4 part?
    # No, C = A_pad * B_pad.
    # A_pad (4x4) * B_pad (4x4) -> C_pad (4x4).
    # C[i, k] = sum_j A_pad[i, j] B_pad[j, k].
    # For i<3, k<5:
    # If k<4, C[i, k] = sum_{j=0}^3 A[i, j] B[j, k] (since A_pad[i, j]=A[i, j] for i<3, and B_pad[j, k]=B[j, k] for k<4).
    # If k=4, C[i, 4] = sum_{j=0}^3 A[i, j] B_pad[j, 4] = sum_{j=0}^3 A[i, j] * 0 = 0.
    # So we lose the 5th column of the result!
    # We need B_pad to be (4x5) or (4x6)?
    # Strassen requires square matrices.
    # We can pad B to (4x4) but we need 5 columns.
    # So we should pad to (5x5) or (6x6)?
    # If we pad to (5x5), the rank is 7^2 * 7? No.
    # 5x5 can be split into 2x2 blocks with some remainder.
    # This gets complicated.
    
    # Let's try a different approach.
    # Use the fact that 3x4x5 can be embedded in 4x4x4 or 5x5x5.
    # Or, use a known decomposition for 3x3x3 (rank 23) and extend.
    
    # Actually, the simplest improvement might be to use the "block" Strassen method correctly.
    # Let's pad A (3x4) to (4x4) and B (4x5) to (4x4) is not enough for the 5th column.
    # We can pad B to (4x6) and then split into two (4x3) blocks?
    # No, Strassen is for square.
    
    # Let's pad both to (5x5).
    # A_pad (5x5), B_pad (5x5).
    # C_pad (5x5).
    # Rank of 5x5 multiplication?
    # We can split 5x5 into a 4x4 part and a 1x1 part, plus cross terms.
    # Or use a recursive Strassen with padding to 8x8?
    # 8x8 rank is 7^3 = 343. Too high.
    # 4x4 rank is 49.
    # 5x5 can be done with 4x4 + 1x1 + cross terms.
    # Cross terms: 4x1 and 1x4.
    # Rank of 4x1 is 4. Rank of 1x4 is 4.
    # Total rank approx 49 + 4 + 4 + 1 = 58. Still close to 60.
    
    # What if we split the 3x4x5 problem differently?
    # Split A into 2x4 and 1x4.
    # Split B into 4x3 and 4x2.
    # Terms:
    # (2x4)*(4x3) -> 2x3. Rank 2*4*3 = 24.
    # (2x4)*(4x2) -> 2x2. Rank 2*4*2 = 16.
    # (1x4)*(4x3) -> 1x3. Rank 1*4*3 = 12.
    # (1x4)*(4x2) -> 1x2. Rank 1*4*2 = 8.
    # Total rank = 24 + 16 + 12 + 8 = 60.
    # Still 60.
    
    # It seems that simple block decomposition doesn't help.
    # We need to use the algebraic structure (like Strassen) to reduce the rank.
    
    # Let's try to implement the 4x4 Strassen decomposition and see if we can map it.
    # If we can't map it directly, we can try to find a decomposition for 3x4x5 using a search algorithm.
    # But search is hard.
    
    # Let's assume the 4x4 Strassen rank of 49 is a good target.
    # Can we embed 3x4x5 into 4x4x4?
    # A (3x4) -> A_pad (4x4) with last row zero.
    # B (4x5) -> B_pad (4x4) with last col zero? No, we need 5 cols.
    # We can split B into B1 (4x4) and B2 (4x1).
    # C = A*B1 + A*B2.
    # A*B1 (3x4) -> pad to 4x4. Rank 49.
    # A*B2 (3x1) -> schoolbook rank 3*4*1 = 12.
    # Total rank 61. Worse.
    
    # Split B into B1 (4x3) and B2 (4x2).
    # A*B1 (3x3). Pad to 4x4?
    # If we pad A to 4x4 and B1 to 4x4, rank 49.
    # A*B2 (3x2). Schoolbook rank 12.
    # Total 61.
    
    # It seems we can't beat 60 easily with padding.
    
    # What if we use a different algorithm?
    # Laderman's algorithm for 3x3 has rank 23.
    # We can use Laderman for the 3x3 part.
    # Split B into B1 (4x3) and B2 (4x2).
    # A*B1 (3x3). Use Laderman (rank 23).
    # A*B2 (3x2). Schoolbook (rank 12).
    # Total rank 35.
    # This is much better!
    
    # Let's implement Laderman's algorithm for 3x3.
    # Laderman's algorithm (1976) reduces 3x3 matrix multiplication to 23 scalar multiplications.
    # The algorithm is complex.
    # We need the explicit U, V, W matrices for the 23 terms.
    
    # Reference:
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # The 23 products are linear combinations of a's and b's.
    
    # This is a lot of work to hardcode.
    # Maybe we can find a simpler low-rank decomposition for 3x4x5?
    # Or, we can just return the schoolbook 60 and hope for the best?
    # No, the goal is to improve.
    
    # Let's try to construct a decomposition for 3x4x5 by combining Laderman (3x3) and schoolbook (3x2).
    # We need the Laderman matrices.
    # I will hardcode the Laderman decomposition for 3x3.
    # Then I will add the schoolbook terms for the 3x2 part.
    # Total rank 23 + 12 = 35.
    
    # Let's find the Laderman 3x3 decomposition.
    # The tensor T[i, j, k] = 1 for i, j, k in 0..2.
    # We need U, V, W of size (9, 23), (9, 23), (9, 23).
    # (Since 3*3=9).
    # For the 3x2 part, we need U, V, W of size (12, 12), (12, 12), (6, 12)?
    # No, the dimensions are fixed by the problem.
    # U is (n*m, rank) = (12, rank).
    # V is (m*p, rank) = (20, rank).
    # W is (p*n, rank) = (15, rank).
    
    # So we need to map the 3x3 and 3x2 decompositions to these dimensions.
    # For the 3x3 part (B1):
    # i in 0..2, j in 0..3, k in 0..2.
    # Wait, B1 is 4x3. So j is 0..3.
    # A is 3x4.
    # So A*B1 is (3x3).
    # The terms involve i (0..2), j (0..3), k (0..2).
    # Laderman is for 3x3 * 3x3.
    # Here we have 3x4 * 4x3.
    # We can split A into A1 (3x3) and A2 (3x1).
    # Split B1 into B11 (3x3) and B12 (1x3).
    # A*B1 = A1*B11 + A2*B12.
    # A1*B11 (3x3). Laderman rank 23.
    # A2*B12 (3x3) (since A2 is 3x1 and B12 is 1x3, the product is 3x3).
    # A2*B12 is a rank-1 matrix (outer product).
    # (A2 * B12)_{ik} = A2_i * B12_k.
    # This can be decomposed into 3 terms?
    # sum_r U_i * V_j * W_k.
    # Here j is fixed (j=3).
    # So we need terms where U[i, 3, r] * V[3, k, r] * W[k, i, r] = A2_i * B12_k.
    # This is a rank-1 matrix in the (i, k) space.
    # So it can be done with 1 term?
    # U[i, 3, r] = A2_i, V[3, k, r] = B12_k, W[k, i, r] = 1?
    # No, W is the output factor.
    # The product is sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    # For j=3, we want sum_r U[i, 3, r] V[3, k, r] W[k, i, r] = A2_i * B12_k.
    # If we set U[i, 3, r] = A2_i, V[3, k, r] = B12_k, W[k, i, r] = 1, then the product is A2_i * B12_k * 1.
    # But W[k, i, r] must be the same for all i, k?
    # No, W is a matrix.
    # So we can set W[k, i, r] = 1 for all i, k.
    # Then the term is A2_i * B12_k.
    # This works!
    # So A2*B12 requires 1 term.
    
    # So total for A*B1 (3x3 output):
    # A1*B11 (3x3*3x3): 23 terms.
    # A2*B12 (3x1*1x3): 1 term.
    # Total 24 terms.
    
    # Now for A*B2 (3x2 output):
    # A (3x4) * B2 (4x2).
    # Schoolbook rank 3*4*2 = 24.
    # Can we do better?
    # Split A into A1 (3x3) and A2 (3x1).
    # Split B2 into B21 (3x2) and B22 (1x2).
    # A1*B21 (3x2). Schoolbook 3*3*2 = 18.
    # A2*B22 (3x2) (3x1*1x2). Rank 1?
    # (A2*B22)_{ik} = A2_i * B22_k.
    # 1 term.
    # Total 19 terms.
    
    # So total rank = 24 + 19 = 43.
    # This is better than 60!
    
    # Let's verify the indices.
    # Global indices:
    # i in 0..2 (n=3)
    # j in 0..3 (m=4)
    # k in 0..4 (p=5)
    
    # Part 1: A1*B11 (3x3 * 3x3)
    # i in 0..2, j in 0..2, k in 0..2.
    # U[i, j, r] = Laderman_U[i, j, r]
    # V[j, k, r] = Laderman_V[j, k, r]
    # W[k, i, r] = Laderman_W[k, i, r]
    # r in 0..22.
    
    # Part 2: A2*B12 (3x1 * 1x3)
    # i in 0..2, j=3, k in 0..2.
    # U[i, 3, 23] = A2_i (which is A[i, 3])
    # V[3, k, 23] = B12_k (which is B[3, k])
    # W[k, i, 23] = 1
    # r = 23.
    
    # Part 3: A1*B21 (3x3 * 3x2)
    # i in 0..2, j in 0..2, k in 3..4.
    # Wait, B21 is 3x2. So j in 0..2, k in 3..4 (global k).
    # Local k' in 0..1. Global k = 3 + k'.
    # U[i, j, r] = A[i, j]
    # V[j, k, r] = B[j, 3+k']
    # W[k, i, r] = 1
    # This is schoolbook for 3x3 * 3x2.
    # Rank 18.
    # r in 24..41.
    
    # Part 4: A2*B22 (3x1 * 1x2)
    # i in 0..2, j=3, k in 3..4.
    # U[i, 3, 42] = A[i, 3]
    # V[3, k, 42] = B[3, k]
    # W[k, i, 42] = 1
    # r = 42.
    
    # Total rank 43.
    
    # Now we need the Laderman matrices for 3x3.
    # I will hardcode them.
    
    # Laderman 3x3 decomposition (23 terms).
    # This is a standard result.
    # I will look up the coefficients.
    # If I can't find them, I will use a simpler decomposition.
    # But 43 is a good improvement.
    
    # Let's assume I can find the Laderman coefficients.
    # If not, I will use schoolbook for the 3x3 part (rank 27).
    # Then total rank = 27 + 1 + 18 + 1 = 47.
    # Still better than 60.
    
    # Let's implement the 47-rank decomposition first (simpler).
    # Then try to find Laderman.
    
    # 47-rank decomposition:
    # Part 1: A1*B11 (3x3 * 3x3). Schoolbook rank 27.
    # Part 2: A2*B12 (3x1 * 1x3). Rank 1.
    # Part 3: A1*B21 (3x3 * 3x2). Schoolbook rank 18.
    # Part 4: A2*B22 (3x1 * 1x2). Rank 1.
    # Total 47.
    
    # Let's implement this.
    
    rank = 47
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1*B11 (i:0..2, j:0..2, k:0..2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: A2*B12 (i:0..2, j:3, k:0..2)
    # U[i, 3, col] = A[i, 3] -> but A is not known. We use symbolic variables?
    # No, the decomposition must be exact for ANY A, B.
    # So U, V, W must be constants (0, 1, -1, etc).
    # The entries of A and B are "multiplied" by the decomposition.
    # Wait, the tensor T[i, j, k] = 1.
    # The product sum_r U[i, j, r] V[j, k, r] W[k, i, r] should be 1.
    # This is independent of A and B.
    # My previous reasoning about A2_i * B12_k was wrong.
    # The decomposition is for the tensor, not for specific matrices.
    # The tensor T[i, j, k] = 1.
    # So U, V, W are constants.
    
    # So the "A2*B12" part:
    # We want sum_r U[i, 3, r] V[3, k, r] W[k, i, r] = 1 for i=0..2, k=0..2.
    # This is a 3x3 matrix of 1s.
    # Rank 1?
    # If U[i, 3, r] = 1, V[3, k, r] = 1, W[k, i, r] = 1, then product is 1.
    # So 1 term works!
    # U[i, 3, col] = 1 for all i.
    # V[3, k, col] = 1 for all k.
    # W[k, i, col] = 1 for all k, i.
    
    # So Part 2 is 1 term.
    
    # Part 3: A1*B21 (i:0..2, j:0..2, k:3..4).
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for i=0..2, j=0..2, k=3,4.
    # This is a set of 3x3x2 = 18 entries of 1.
    # Schoolbook: 18 terms.
    # Can we do better?
    # The tensor is T[i, j, k] = 1 for i=0..2, j=0..2, k=3,4.
    # This is a block of 1s.
    # Rank is at least max(3*2, 3*2, 2*3) = 6?
    # No, the rank is the number of terms.
    # If we use schoolbook, it's 18.
    # Can we merge terms?
    # Probably not easily.
    
    # Part 4: A2*B22 (i:0..2, j:3, k:3..4).
    # We want sum_r U[i, 3, r] V[3, k, r] W[k, i, r] = 1 for i=0..2, k=3,4.
    # This is a 3x2 block of 1s.
    # 1 term?
    # U[i, 3, col] = 1, V[3, k, col] = 1, W[k, i, col] = 1.
    # Product is 1.
    # So 1 term.
    
    # Total rank: 27 (Part 1) + 1 (Part 2) + 18 (Part 3) + 1 (Part 4) = 47.
    
    # Let's implement this.
    
    col = 0
    
    # Part 1: i:0..2, j:0..2, k:0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: i:0..2, j:3, k:0..2
    for i in range(3):
        U[i * m + 3, col] = 1
    for k in range(3):
        V[3 * p + k, col] = 1
    for k in range(3):
        for i in range(3):
            W[k * n + i, col] = 1
    col += 1
    
    # Part 3: i:0..2, j:0..2, k:3..4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 4: i:0..2, j:3, k:3..4
    for i in range(3):
        U[i * m + 3, col] = 1
    for k in range(3, 5):
        V[3 * p + k, col] = 1
    for k in range(3, 5):
        for i in range(3):
            W[k * n + i, col] = 1
    col += 1
    
    # Total col should be 47.
    
    # Convert to lists
    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

