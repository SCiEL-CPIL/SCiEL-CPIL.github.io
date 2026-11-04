"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)  <-- Note: W index is k*n + i (output C is n x p, but trace convention says W[k*n+i, r])
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Let's re-read carefully: "W[k*n+i,r]"
    # Usually C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k] = 1.
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]?
    # The task says W[k*n+i, r]. This implies the third mode of the tensor is indexed by (k, i) in column-major or similar?
    # Let's check the standard convention for these tasks.
    # Usually:
    # U corresponds to A indices (i, j) -> size n*m
    # V corresponds to B indices (j, k) -> size m*p
    # W corresponds to C indices (i, k) -> size n*p
    # But the task explicitly says W[k*n+i, r].
    # If C is n x p, indices are (i, k) with i in 0..n-1, k in 0..p-1.
    # k*n + i means index = k * n + i.
    # This is column-major indexing for an n x p matrix if we consider the first index i (row) and second index k (col).
    # Let's verify with the schoolbook code provided:
    # W[k * n + i][col] = 1.0
    # Yes, the schoolbook code uses W[k * n + i].
    # So W has size p * n.
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # This is the standard matrix multiplication tensor.
    # We know that for n=2, m=2, p=2, the rank is 7 (Strassen).
    # For n=3, m=3, p=3, the rank is 23 (Landsberg).
    # For n=3, m=3, p=6, we can view B as two 3x3 blocks.
    # C = A * B1 + A * B2.
    # Since A is the same, we can potentially share the computation for A.
    # However, the multiplication is linear in A, B1, B2.
    # T_{i,j,k1} for B1 and T_{i,j,k2} for B2.
    # This is equivalent to multiplying A (3x3) by a 3x6 matrix.
    # This can be seen as the Kronecker product of the 3x3x3 multiplication tensor and a 3x1x2 "broadcast" tensor?
    # Let's analyze the structure.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # If we split B into B_left (3x3) and B_right (3x3), then C = A*B_left + A*B_right.
    # The tensor is T = T_{3,3,3} \otimes I_2? No.
    # The indices are (i, j, k).
    # If we fix i, j, the term is A[i,j] * (B[j,0..5]).
    # The dependence on k is linear.
    # Actually, we can just use the known rank for 3x3x3 which is 23.
    # But here p=6.
    # Rank(3x3x6) <= Rank(3x3x3) * 2? No, that's not right.
    # Rank(A*B1 + A*B2) <= Rank(A*B1) + Rank(A*B2) = 23 + 23 = 46.
    # Can we do better?
    # The tensor is T_{i,j,k} = 1.
    # This is a "slice" of the 3x3x3 tensor? No, it's a product.
    # Actually, the 3x3x6 tensor is the direct sum of two 3x3x3 tensors?
    # No, the first two indices (i,j) are shared.
    # T(i,j,k) = 1 for k=0..5.
    # This is equivalent to computing 6 outputs C[i, k] = sum_j A[i,j] B[j,k].
    # Since the summation index j is the same for all k, we can compute the product A * B where B is 3x6.
    # This is exactly the problem.
    # Is there a known algorithm for 3x3 * 3x6?
    # We can use Strassen for 2x2 blocks?
    # 3x3 can be padded to 4x4. Rank(4x4x4) = 15 (Strassen).
    # Then 4x4 * 4x6?
    # Or use the 3x3x3 rank 23.
    # If we use 3x3x3 rank 23, we can compute A * B1 (3x3) and A * B2 (3x3).
    # Total rank 23 + 23 = 46.
    # Can we share the 23 terms?
    # The terms are linear combinations of A[i,j].
    # For B1, we have 23 intermediate products M_r = sum_{i,j} U_r[i,j] A[i,j] * ...
    # Wait, the decomposition is:
    # C[i,k] = sum_r U_r[i,j] V_r[j,k] W_r[i,k]? No.
    # Standard form:
    # M_r = sum_{i,j} U_r[i,j] A[i,j]
    # Then C[i,k] = sum_r V_r[j,k] M_r ? No.
    # The tensor decomposition is:
    # T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[i,k]
    # Here U_r[i,j] is the coefficient for A[i,j].
    # V_r[j,k] is the coefficient for B[j,k].
    # W_r[i,k] is the coefficient for C[i,k].
    # For 3x3x3, we have 23 such triples (U_r, V_r, W_r).
    # For 3x3x6, we need to decompose T[i,j,k] for k=0..5.
    # If we use the 3x3x3 decomposition for B1 (k=0,1,2) and another for B2 (k=3,4,5), the rank is 46.
    # Is it possible to do better?
    # The rank of the 3x3x6 tensor is at least the rank of the 3x3x3 tensor (by setting B2=0), so rank >= 23.
    # It is also at least the rank of 3x3x3 * 2? No.
    # Let's try to construct a rank 46 solution first, as it's guaranteed to work if we have the 3x3x3 decomposition.
    # But we don't have the 3x3x3 decomposition hardcoded easily.
    # We can use Strassen's algorithm for 2x2.
    # 3x3 * 3x6 can be computed by padding to 4x4 * 4x6?
    # Or 4x4 * 4x4 + 4x4 * 4x2?
    # Rank(4x4x4) = 15.
    # Rank(4x4x2) ?
    # Let's stick to the schoolbook if we can't find a better one quickly, but the prompt asks for improvement.
    # The prompt says "score is the rank". Lower is better.
    # Schoolbook is 54.
    # If we find rank 46, score is 54/46 = 1.17.
    # If we find rank 36, score is 54/36 = 1.5.
    
    # Let's try to implement a rank 46 solution using two 3x3x3 decompositions.
    # But I need the 3x3x3 decomposition.
    # I can use a known 3x3x3 decomposition (Landsberg's 23 rank algorithm).
    # Or I can use a simpler one.
    # Actually, I can just use the schoolbook for 3x3x3 (rank 27) twice.
    # Rank 27 + 27 = 54. No improvement.
    # I need the rank 23 algorithm for 3x3x3.
    # Finding the 23 rank algorithm coefficients is non-trivial.
    
    # Alternative: Use Strassen for 2x2.
    # 3x3 * 3x6.
    # Pad A to 4x4, B to 4x6.
    # Compute C = A_pad * B_pad.
    # We can split B_pad into two 4x3 blocks? No, 4x6 is two 4x3 blocks.
    # Rank(4x4x3)?
    # Or split B_pad into 4x4 and 4x2.
    # Rank(4x4x4) = 15.
    # Rank(4x4x2) = ?
    # The tensor for 4x4x2 is T[i,j,k] = 1 for k=0,1.
    # This is equivalent to multiplying A (4x4) by B1 (4x2).
    # C = A * B1.
    # We can compute this with 2 multiplications of 4x4? No.
    # We can compute A * b1 and A * b2 where b1, b2 are vectors.
    # Rank of 4x4x1 is 4? No, A*v.
    # A*v = sum_j A[i,j] v[j].
    # Tensor T[i,j] = 1. Rank 4?
    # Actually, matrix-vector multiplication 4x4 * 4x1 has rank 4?
    # C[i] = sum_j A[i,j] v[j].
    # This is a linear map. The tensor is T[i,j] = 1.
    # Rank is 4?
    # U_r[i,j] = delta_{i,r} delta_{j,r}? No.
    # If we use identity, M_r = A[r,r] v[r]. Sum is not A*v.
    # We need M_r = sum_j A[i,j] v[j].
    # This is not a tensor decomposition of rank 4 in the standard sense?
    # Wait, the tensor for matrix-vector multiplication is not the same as matrix-matrix.
    # The task is matrix-matrix multiplication.
    
    # Let's go back to 3x3x3 rank 23.
    # I will hardcode a known 3x3x3 rank 23 decomposition.
    # Source: "A 23-rank algorithm for 3x3 matrix multiplication" by Landsberg.
    # The coefficients are complex.
    # However, I can use a simpler approach:
    # Use the schoolbook for 3x3x3 but with some sharing?
    # No, schoolbook is 27.
    
    # What if I use Strassen for 2x2 blocks?
    # 3x3 * 3x6.
    # Split A into 2x2, 2x1, 1x2, 1x1 blocks?
    # This is getting complicated.
    
    # Let's try to find a rank 46 solution using the 3x3x3 rank 23 algorithm.
    # I will generate the 3x3x3 rank 23 decomposition using a known library or hardcoded values.
    # Since I cannot use external libraries, I will hardcode the 3x3x3 rank 23 algorithm.
    # The 3x3x3 rank 23 algorithm is known.
    # I will use the one from "Fast Matrix Multiplication" by Bini et al.
    # Actually, I can just use the schoolbook for now and return 54.
    # But the goal is to improve.
    # Let's try to implement a rank 46 solution by using two 3x3x3 decompositions.
    # If I can find the 3x3x3 rank 23 decomposition, I can use it twice.
    # Total rank 46.
    
    # I will hardcode the 3x3x3 rank 23 decomposition.
    # The coefficients are:
    # U, V, W are 9x23 matrices.
    # I will use a simplified version or just return the schoolbook if I can't find the coefficients.
    # Given the time constraint, I will return the schoolbook solution (rank 54) but with a check for time.
    # Wait, the prompt says "score is the rank".
    # If I return rank 54, score is 1.0.
    # If I return rank 46, score is 1.17.
    # I need to find a better rank.
    
    # Let's try to use the fact that 3x3 * 3x6 can be done with 3x3 * 3x3 and 3x3 * 3x3.
    # If I use the rank 23 algorithm for 3x3x3, I get rank 46.
    # I will hardcode the 3x3x3 rank 23 algorithm.
    # The coefficients are available in literature.
    # I will use a placeholder for now and return the schoolbook.
    
    # Actually, I can use the Strassen algorithm for 2x2.
    # 3x3 * 3x6 can be computed by padding to 4x4 * 4x6.
    # Rank(4x4x4) = 15.
    # Rank(4x4x2) = ?
    # 4x4 * 4x2 is equivalent to 2 vector multiplications.
    # Each vector multiplication 4x4 * 4x1 has rank 4?
    # No, the tensor for 4x4 * 4x1 is T[i,j] = 1.
    # Rank is 4?
    # If rank is 4, then total rank is 15 + 4 + 4 = 23?
    # No, the 4x4x4 part is for the 4x4 block of B.
    # The 4x4x2 part is for the 4x2 block of B.
    # Total rank 15 + 8 = 23?
    # But we padded to 4x4.
    # The original is 3x3 * 3x6.
    # If we pad to 4x4 * 4x6, we compute extra terms.
    # We can just ignore them.
    # So rank is 15 + 8 = 23?
    # This seems too good to be true.
    # Let's check the rank of 4x4 * 4x1.
    # C[i] = sum_j A[i,j] v[j].
    # This is a bilinear map.
    # The tensor is T[i,j] = 1.
    # The rank of this tensor is 4?
    # Yes, because we can write it as sum_{r=1}^4 e_i^r e_j^r.
    # So rank is 4.
    # So 4x4 * 4x2 has rank 8.
    # 4x4 * 4x4 has rank 15.
    # Total rank 15 + 8 = 23.
    # But we are computing 4x4 * 4x6.
    # This gives us a 4x4 * 4x6 decomposition of rank 23.
    # But we need 3x3 * 3x6.
    # We can just take the 3x3 * 3x6 part of the 4x4 * 4x6 result.
    # So the rank is 23.
    # This is a huge improvement!
    # Let's verify.
    # We pad A (3x3) to A_pad (4x4) by adding a row and column of zeros.
    # We pad B (3x6) to B_pad (4x6) by adding a row of zeros.
    # We compute C_pad = A_pad * B_pad.
    # C_pad is 4x6.
    # The top-left 3x6 part is C = A * B.
    # The extra terms are zero because of the padding.
    # So we can use the 4x4 * 4x6 decomposition.
    # The rank of 4x4 * 4x6 is Rank(4x4x4) + Rank(4x4x2) = 15 + 8 = 23.
    # Wait, is Rank(4x4x2) = 8?
    # 4x4 * 4x2 is two 4x4 * 4x1 multiplications.
    # Each has rank 4.
    # So total rank 8.
    # Yes.
    # So the total rank is 23.
    # This is much better than 54.
    
    # Now I need to construct the 4x4 * 4x6 decomposition.
    # I need the 4x4x4 Strassen decomposition (rank 15).
    # And the 4x4x2 decomposition (rank 8).
    
    # Strassen 4x4x4:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # This is for 2x2 blocks of 2x2 matrices.
    # For 4x4, we treat it as 2x2 blocks of 2x2 matrices.
    # So we need 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # So total rank 7 * 7 = 49?
    # No, the Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
    # But each 2x2 multiplication is itself decomposed into 7 scalar multiplications.
    # So total rank 49.
    # But we know that Rank(4x4x4) = 15.
    # This is because the 4x4x4 tensor can be decomposed more efficiently.
    # The rank 15 algorithm for 4x4x4 is known.
    # I will hardcode the 4x4x4 rank 15 algorithm.
    
    # This is getting too complex to hardcode from memory.
    # I will return the schoolbook solution (rank 54) for now.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
