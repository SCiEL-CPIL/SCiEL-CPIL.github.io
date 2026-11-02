import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    # We need a rank-r decomposition for 3x5 * 5x5 -> 3x5
    
    # Strategy: Use a known rank-19 decomposition for 5x5x5 and replicate for each of the 3 rows.
    # However, we can potentially do better by exploiting the structure.
    # The tensor is T[i, j, k] where i in [0,2], j in [0,4], k in [0,4].
    # C[i, k] = sum_j A[i, j] B[j, k]
    
    # A simple block approach:
    # Split the 5x5 multiplication into blocks.
    # But let's try to find a low rank decomposition directly.
    
    # Known bounds for 5x5x5:
    # Schoolbook: 25
    # Strassen-like: 19 is a known upper bound (e.g., from Winograd, etc.)
    # Actually, the best known upper bound for 5x5x5 is 19.
    # The best known lower bound is 18.
    
    # If we use rank 19 for each of the 3 rows independently, total rank is 3 * 19 = 57.
    # Can we do better?
    # The matrix A is 3x5. The matrix B is 5x5.
    # We can view this as computing 3 separate 1x5 * 5x5 products.
    # Each 1x5 * 5x5 product is a linear map from B to C_row.
    # C_row = A_row * B.
    # This is a tensor of size 5 x 5 x 5 (indices j, k, and the row index i is fixed).
    # Wait, for a fixed i, C[i, k] = sum_j A[i, j] B[j, k].
    # The bilinear forms are A[i, j] * B[j, k].
    # For a fixed i, there are 5*5 = 25 terms.
    # Since the A[i, j] are independent for different i, we cannot share factors between different i's
    # unless we combine them. But A[i, j] and A[i', j] are distinct variables.
    # So the total rank is at least the sum of the ranks for each i?
    # Not necessarily, because we can have factors that involve A[i, j] and A[i', j'] simultaneously.
    # However, the tensor is a direct sum of 3 tensors T_i, where T_i is the 5x5x5 tensor.
    # The rank of a direct sum is at most the sum of the ranks.
    # It is at least the maximum of the ranks.
    # Is it possible to have rank( T1 + T2 + T3 ) < rank(T1) + rank(T2) + rank(T3)?
    # Generally, rank(T1 + T2) <= rank(T1) + rank(T2).
    # So rank <= 3 * 19 = 57.
    
    # Can we do better than 57?
    # If we use the schoolbook method, rank = 3 * 25 = 75.
    # If we use a rank-19 decomposition for 5x5, rank = 3 * 19 = 57.
    
    # Let's try to construct a rank-19 decomposition for 5x5x5.
    # One known rank-19 decomposition is based on the "Larsen" or "Winograd" algorithms.
    # However, implementing a specific rank-19 decomposition from scratch is complex and error-prone.
    
    # Alternative: Use a rank-20 or rank-21 decomposition if 19 is too hard to code correctly.
    # Or, use a different strategy.
    
    # Let's try to use a rank-19 decomposition.
    # Reference: "The rank of the matrix multiplication tensor" by various authors.
    # A common rank-19 decomposition for 5x5 uses the following structure:
    # It can be derived by combining a 4x4 Strassen (rank 7) and handling the 5th dimension.
    # 5 = 4 + 1.
    # A = [[A44, A41], [A14, A11]] where A44 is 4x4, A41 is 4x1, A14 is 1x4, A11 is 1x1.
    # B = [[B44, B41], [B14, B11]]
    # C = A * B = [[A44*B44 + A41*B14, A44*B41 + A41*B11],
    #              [A14*B44 + A11*B14, A14*B41 + A11*B11]]
    
    # Let's compute the rank of this block decomposition.
    # C11 (4x4): A44*B44 + A41*B14
    #   A44*B44: rank 7 (Strassen for 4x4)
    #   A41*B14: rank 16 (4*4 terms)
    #   But they can be combined? No, they involve different variables.
    #   Wait, A41 is 4x1, B14 is 1x4. The product is 4x4.
    #   The terms are A41[r] * B14[c] for r in 0..3, c in 0..3.
    #   There are 16 such terms.
    #   Total for C11: 7 + 16 = 23? No, Strassen for A44*B44 uses 7 products.
    #   The other terms are simple products.
    #   So C11 requires 7 + 16 = 23 products?
    #   This doesn't seem to save much.
    
    # Let's look for a known rank-19 decomposition.
    # Actually, the best known upper bound for 5x5x5 is 19.
    # A specific decomposition is given in "Lower bounds for the complexity of bilinear algorithms" or similar.
    # Without a specific reference, it is risky to hardcode.
    
    # Let's try a different approach: Use a rank-21 decomposition which might be easier to construct.
    # Or, stick to the 57 rank (3 * 19) if I can find a valid 19-rank decomposition.
    
    # Let's try to use a rank-19 decomposition from a known source.
    # One such decomposition is based on the following:
    # It is not trivial to write down.
    
    # Instead, let's use a rank-20 decomposition for 5x5.
    # If we can get rank 20 for 5x5, total rank is 60.
    # If we can get rank 19, total rank is 57.
    
    # Let's try to implement a rank-19 decomposition.
    # I will use the decomposition from "The rank of the matrix multiplication tensor" by Cohn, Wigderson, and others?
    # No, let's use a simpler one.
    
    # Actually, there is a well-known rank-19 decomposition for 5x5.
    # It can be found in many papers.
    # Let's try to construct it using the "Larsen" method.
    # Larsen (1993) gave a rank-19 algorithm for 5x5.
    
    # Since I cannot recall the exact 19 terms, I will use a rank-20 decomposition.
    # A rank-20 decomposition can be constructed by taking a rank-7 decomposition for 4x4 and adding 13 terms for the 5th row/column.
    # 5 = 4 + 1.
    # A = [[A44, a], [b, c]]
    # B = [[B44, d], [e, f]]
    # C = [[A44*B44 + a*e, A44*d + a*f],
    #      [b*B44 + c*e, b*d + c*f]]
    
    # C11 = A44*B44 + a*e
    #   A44*B44: rank 7
    #   a*e: 16 terms (a is 4x1, e is 1x4)
    #   Total: 7 + 16 = 23? No, the terms a*e are distinct from A44*B44.
    #   So C11 requires 23 products.
    #   This is worse than 25.
    
    # Let's try a different split.
    # 5 = 3 + 2.
    # A = [[A33, A32], [A23, A22]]
    # B = [[B33, B32], [B23, B22]]
    # C = [[A33*B33 + A32*B23, A33*B32 + A32*B22],
    #      [A23*B33 + A22*B23, A23*B32 + A22*B22]]
    
    # C11 (3x3): A33*B33 + A32*B23
    #   A33*B33: rank 7 (Strassen)
    #   A32*B23: 3*2 = 6 terms
    #   Total: 13
    # C12 (3x2): A33*B32 + A32*B22
    #   A33*B32: 3*3*2 = 18 terms? No, A33 is 3x3, B32 is 3x2.
    #   This is a matrix multiplication of 3x3 by 3x2.
    #   Rank is at most 3*3*2 = 18.
    #   Can we do better?
    #   Using Strassen for 3x3 by 3x2?
    #   Not straightforward.
    
    # This block approach is getting complicated.
    
    # Let's go back to the 57 rank idea.
    # I will use a known rank-19 decomposition for 5x5.
    # I will hardcode the coefficients.
    # Since I don't have the coefficients memorized, I will use a rank-25 decomposition (schoolbook) for now,
    # but that scores 1.0.
    
    # To beat 1.0, I need rank < 75.
    # Any rank < 75 is an improvement.
    # Let's try rank 70.
    # Can I get rank 70?
    # 3 * 23 = 69.
    # If I can find a rank-23 decomposition for 5x5, total rank is 69.
    # Rank 23 is easier to construct than 19.
    
    # Let's try to construct a rank-23 decomposition for 5x5.
    # Use the 4+1 split.
    # A = [[A44, a], [b, c]]
    # B = [[B44, d], [e, f]]
    # C11 = A44*B44 + a*e
    #   A44*B44: rank 7
    #   a*e: 16 terms
    #   Total: 23
    # C12 = A44*d + a*f
    #   A44*d: 4*4 = 16 terms? No, A44 is 4x4, d is 4x1.
    #   This is a matrix-vector product.
    #   Rank is 16.
    #   a*f: 4 terms.
    #   Total: 20
    # C21 = b*B44 + c*e
    #   b*B44: 4*4 = 16 terms.
    #   c*e: 4 terms.
    #   Total: 20
    # C22 = b*d + c*f
    #   b*d: 4*1 = 4 terms? No, b is 1x4, d is 4x1.
    #   This is a dot product.
    #   Rank is 4.
    #   c*f: 1 term.
    #   Total: 5
    
    # Total rank: 23 + 20 + 20 + 5 = 68.
    # This is for the 5x5 multiplication.
    # Wait, the terms are not independent.
    # The products used in C11, C12, C21, C22 are distinct.
    # So the total rank is the sum of the ranks of the blocks.
    # 68 < 75.
    # So a rank-68 decomposition for 5x5 is possible.
    # Then for 3x5 * 5x5, the total rank is 3 * 68 = 204?
    # No, the 3x5 * 5x5 problem is not 3 independent 5x5 multiplications.
    # It is 3 rows of A multiplied by B.
    # Each row is a 1x5 vector.
    # So it is 3 independent 1x5 * 5x5 multiplications.
    # The rank of a 1x5 * 5x5 multiplication is 25 (schoolbook).
    # Can we do better?
    # C = v * B, where v is 1x5, B is 5x5.
    # C[k] = sum_j v[j] B[j, k].
    # This is a linear map.
    # The tensor is T[j, k, l] where l is the row index? No.
    # For a fixed row i, C[i, k] = sum_j A[i, j] B[j, k].
    # The bilinear forms are A[i, j] * B[j, k].
    # There are 25 such forms for each i.
    # Since A[i, j] are independent for different i, the total rank is 3 * 25 = 75.
    # Unless we can combine terms across different i's.
    # But A[i, j] and A[i', j] are distinct variables.
    # So the tensor is a direct sum of 3 tensors.
    # The rank of a direct sum is the sum of the ranks.
    # So the rank is 3 * rank(1x5 * 5x5).
    # The rank of 1x5 * 5x5 is 25.
    # So the total rank is 75.
    
    # Wait, is the rank of 1x5 * 5x5 really 25?
    # C[k] = sum_j v[j] B[j, k].
    # This is just a matrix-vector product.
    # The tensor is T[j, k] = v[j] B[j, k].
    # This is a rank-25 tensor?
    # No, the tensor is of size 5 x 5 x 5? No.
    # The input is v (5) and B (25).
    # The output is C (5).
    # The tensor is T[j, k, l]? No.
    # The bilinear forms are v[j] * B[j, k].
    # There are 25 such forms.
    # Are they linearly independent?
    # Yes, because each involves a unique pair (j, k).
    # So the rank is 25.
    
    # So the total rank is 3 * 25 = 75.
    # It seems impossible to beat 75 for this specific problem structure.
    # Unless... I am misunderstanding the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 5 matrix."
    # A is 3x5, B is 5x5, C is 3x5.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is T[i, j, k] where i in 0..2, j in 0..4, k in 0..4.
    # T[i, j, k] = A[i, j] * B[j, k].
    # This is a 3 x 5 x 5 tensor.
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is u[i] * v[j] * w[k].
    # But here, the bilinear form is A[i, j] * B[j, k].
    # This is not a simple product of three vectors.
    # It is a product of a variable from A and a variable from B.
    # So the tensor is a "matrix multiplication tensor" of size 3x5x5.
    # The rank of the matrix multiplication tensor [n, m, p] is the minimum number of multiplications.
    # For [3, 5, 5], the rank is at most 3 * 5 * 5 = 75.
    # Can it be less?
    # Yes, if there are dependencies.
    # But A[i, j] and B[j, k] are all independent variables.
    # So the tensor is a direct sum of 3 tensors T_i, where T_i is the [1, 5, 5] tensor.
    # The rank of [1, 5, 5] is 25.
    # So the rank of [3, 5, 5] is 3 * 25 = 75.
    
    # It seems the rank is exactly 75.
    # So the seed program is optimal.
    # But the task says "score is the rank" and "lower is better".
    # And "1.0 equals the seed program".
    # So if the rank is 75, the score is 75.
    # If I can't beat 75, I can't beat 1.0.
    
    # Wait, is the rank of [1, 5, 5] really 25?
    # C[k] = sum_j v[j] B[j, k].
    # This is a linear map from v to C for each B.
    # No, it's bilinear.
    # The tensor is T[j, k] = v[j] B[j, k].
    # This is a 5x5 matrix for each (v, B).
    # No, the tensor is a 3-way tensor T[i, j, k] with i=0 (fixed).
    # T[0, j, k] = v[j] B[j, k].
    # This is a 1x5x5 tensor.
    # The rank is the minimum r such that T = sum_{l=1}^r u_l[0] * v_l[j] * w_l[k].
    # But u_l[0] is a scalar.
    # So T[j, k] = sum_{l=1}^r c_l * v_l[j] * w_l[k].
    # This is a matrix factorization of T[j, k] into rank-1 matrices.
    # But T[j, k] is not a fixed matrix. It depends on v and B.
    # The tensor is the identity map on the space of matrices? No.
    # The tensor is the bilinear form (v, B) -> v * B.
    # This is the standard matrix multiplication tensor for [1, 5, 5].
    # The rank of [1, m, p] is m * p.
    # Because the bilinear forms v[j] * B[j, k] are linearly independent.
    # So the rank is 25.
    
    # So the total rank is 75.
    # I cannot beat 75.
    
    # Unless I am wrong about the independence.
    # Are v[j] * B[j, k] linearly independent?
    # Yes, because each term involves a unique pair of variables (v[j], B[j, k]).
    # No other term involves that specific pair.
    # So they are linearly independent.
    
    # So the rank is 75.
    # The seed program is optimal.
    # I will just return the seed program.
    
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
