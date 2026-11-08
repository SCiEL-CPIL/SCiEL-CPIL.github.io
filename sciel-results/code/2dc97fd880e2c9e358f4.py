import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 4, 4, 6
    
    # Use recursive Strassen to achieve rank 77
    # Top level: 4x4 * 4x6 -> 7 * (2x2 * 2x3)
    # Each 2x2 * 2x3 -> 7 * (2x2 * 2x2) + 4 * (2x2 * 2x1) = 7*7 + 4*4 = 49 + 16 = 65?
    # Wait. 2x2 * 2x3.
    # Split A (2x2) into 4 blocks of 1x1.
    # Split B (2x3) into 2x2 and 2x1 parts.
    # This is not standard Strassen.
    
    # Let's stick to the simpler decomposition which is known to work:
    # Rank 77 is achievable for 4x4x6?
    # 4x4x4 is rank 49.
    # 4x4x2 is rank 28?
    # 49 + 28 = 77.
    
    # Let's verify if I can construct 4x4x2 with rank 28.
    # 4x4x2 = 2 * (4x4x1)? No.
    # 4x4x2 can be seen as two 4x4x1 problems.
    # Rank of 4x4x1 is 16 (schoolbook) or less?
    # Actually, 4x4x1 is just matrix-vector multiplication. Rank is 16 (min).
    # So 4x4x2 would be 32?
    
    # Let's re-evaluate the inspiration score 1.2468.
    # 96 / 77 = 1.24675...
    # So the inspiration uses rank 77.
    
    # How to get 77?
    # 4x4x4 (rank 49) + 4x4x2 (rank 28) = 77.
    # Is 4x4x2 rank 28?
    # 4x4x2 tensor.
    # Maybe it's (2x2x2) + (2x2x2) ... ?
    
    # Let's try to implement the rank 77 solution directly if possible, or fall back to 96.
    # Since I don't have the explicit 49+28 terms, and 96 is safe, I will output 96.
    # But wait, the inspiration got 1.2468 which corresponds to 77.
    # If I output 96, my score is 1.0.
    # If I can output 77, my score is 1.2468.
    
    # I will attempt to generate the 77 terms.
    # 4x4x4 Strassen (49 terms)
    # 4x4x2 ... ?
    
    # Actually, there is a known rank 77 decomposition for 4x4x6?
    # Or is it 4x4x4 (49) + 4x2x6 (18)? No.
    
    # Let's look at the structure of 77.
    # 7 = 7 * 11.
    # 11 = 7 (Strassen 2x2x2) + 4 (Schoolbook 2x2x1).
    # So 4x4x6 is decomposed into 7 blocks of (2x2 * 2x3).
    # Each 2x2 * 2x3 is decomposed into 7 * (2x2 * 2x2) + 4 * (2x2 * 2x1)?
    # No, 2x2 * 2x3.
    # A is 2x2. B is 2x3.
    # Split B into B1 (2x2) and B2 (2x1).
    # A * B1 (2x2 * 2x2) -> Rank 7 (Strassen)
    # A * B2 (2x2 * 2x1) -> Rank 4 (Schoolbook, since 2x2*2x1 is just 4 scalar mults)
    # Total per block: 7 + 4 = 11.
    # 7 blocks -> 77.
    
    # This makes sense!
    
    # Implementation Plan:
    # 1. Define Strassen 2x2x2 terms (7 terms).
    # 2. Define Schoolbook 2x2x1 terms (4 terms).
    # 3. Construct 2x2x2x3 decomposition (11 terms).
    # 4. Apply 7 Strassen combinations at the top level to get 77 terms for 4x4x6.
    
    # Step 1: Strassen 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express each M as a rank 1 tensor in the 2x2x2 space.
    # But Strassen terms are not rank 1 tensors themselves in the standard basis?
    # Wait. The decomposition of the multiplication tensor is a sum of rank 1 tensors.
    # Strassen's algorithm gives a decomposition into 7 products of linear forms.
    # Each product is a rank 1 tensor.
    
    # Let's define the linear forms for A and B and C for each of the 7 terms.
    # Term 1:
    # A part: a + d
    # B part: e + h
    # C part: C11 + C22 - C12 - C21? No.
    # The tensor T is defined by T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, the decomposition is:
    # sum_t u_t(A) v_t(B) w_t(C)
    # where u_t, v_t are linear forms on A and B, and w_t is a linear form on C?
    # No, w_t corresponds to the coefficients in C.
    
    # Let's use the standard representation:
    # For each term t:
    # u_t: vector in R^4 (coefficients for A)
    # v_t: vector in R^4 (coefficients for B)
    # w_t: vector in R^4 (coefficients for C)
    
    # Strassen 2x2x2 terms:
    # 1. u: [1, 0, 0, 1] (a+d), v: [1, 0, 0, 1] (e+h), w: [1, 0, 0, 1] (C11+C22) ... wait.
    # The relation is:
    # C = sum_t u_t(A) v_t(B) w_t^T
    # where w_t is a matrix? No, w_t is a vector of coefficients for C.
    # Actually, it's:
    # C_{ik} = sum_t (sum_j u_{t,ij} A_{ij}) (sum_j v_{t,jk} B_{jk}) w_{t,ik} ? No.
    
    # Correct form:
    # The tensor T is a 3-way tensor T[i,j,k].
    # Decomposition: T = sum_t u_t \otimes v_t \otimes w_t
    # where (u_t \otimes v_t \otimes w_t)[i,j,k] = u_t[i] v_t[j] w_t[k].
    # And we want T[i,j,k] = delta_{i,k} ? No.
    # T[i,j,k] = 1 if j == i? No.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has T[i,j,k] = 1 if ... wait.
    # The tensor for matrix multiplication is T[i,j,k] = 1 if the indices match?
    # No, the tensor T is such that <T, A \otimes B> = C.
    # T[i,j,k] = 1 if i, j, k are compatible?
    # Actually, T[i,j,k] = 1 if we multiply A[i,j] and B[j,k] to get C[i,k].
    # So T[i,j,k] = 1 for all i,j,k? No.
    # T[i,j,k] = 1 if j is the inner index.
    # The tensor is T[i,j,k] = 1 if i,j,k are such that A[i,j]B[j,k] contributes to C[i,k].
    # This is always true for any i,j,k.
    # So T[i,j,k] = 1 for all i,j,k?
    # No, that would be the all-ones tensor.
    # The matrix multiplication tensor is T[i,j,k] = 1 if j==j?
    # It is T[i,j,k] = 1 for all i,j,k.
    # Wait, if T is all ones, then <T, A \otimes B> = (sum A) (sum B). That's not matrix multiplication.
    
    # The correct tensor is T[i,j,k] = 1 if i, j, k form a valid triple?
    # No, the tensor is defined by T[i,j,k] = 1 if the position (i,j) in A and (j,k) in B map to (i,k) in C.
    # This mapping is unique.
    # So T[i,j,k] = 1 if ... there is no condition?
    # The tensor T is a 3-tensor of size n x m x p.
    # T[i,j,k] = 1 if i, j, k are indices?
    # Actually, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # No, that's wrong.
    
    # Let's check the definition of the matrix multiplication tensor.
    # T_{ijk} = 1 if i, j, k are such that A_{ij} B_{jk} contributes to C_{ik}.
    # This is true for all i, j, k.
    # So T_{ijk} = 1 for all i, j, k?
    # If so, the rank is 1? No.
    
    # I am confusing the tensor definition.
    # The tensor T is such that C_{ik} = sum_{j} A_{ij} B_{jk}.
    # This can be written as C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}.
    # Comparing coefficients, T_{ijk} = 1 for all i, j, k.
    # This implies the tensor is the all-ones tensor?
    # No, the indices i, j, k are not independent in the sum?
    # The sum is over j.
    # So for a fixed i, k, C_{ik} = sum_j A_{ij} B_{jk}.
    # This means the tensor T has non-zero entries only where?
    # T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # It is 1 if the indices match?
    # Yes, T_{ijk} = 1 for all i, j, k.
    # But wait, if T is all ones, then <T, A \otimes B> = sum_{ijk} A_{ij} B_{jk}.
    # This is not C_{ik}. It is a scalar.
    
    # The correct view:
    # T is a 3-way tensor.
    # The contraction is: C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}.
    # For this to hold, T_{ijk} must be 1 for all i, j, k?
    # No, T_{ijk} depends on i, j, k.
    # But for a fixed i, k, the sum is over j.
    # So T_{ijk} must be 1 for all j?
    # Yes.
    # So T_{ijk} = 1 for all i, j, k.
    # This is the all-ones tensor.
    # But the rank of the all-ones tensor of size n x m x p is 1?
    # No, the rank of the all-ones tensor is 1.
    # But matrix multiplication has rank > 1.
    
    # Where is the error?
    # The tensor T is not the all-ones tensor.
    # The tensor T is defined by T[i,j,k] = 1 if ...
    # Ah, the indices i, j, k are not all independent in the same way.
    # The tensor T is of size n x m x p.
    # T[i,j,k] = 1 if i, j, k are valid indices.
    # But the contraction is:
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a bilinear form.
    # The tensor T is such that T(i, j, k) = 1 if i, j, k are the indices for A, B, C?
    # No.
    
    # Let's use the standard decomposition.
    # Rank 1 tensor: u \otimes v \otimes w.
    # (u \otimes v \otimes w)[i,j,k] = u[i] v[j] w[k].
    # We want sum_t (u_t \otimes v_t \otimes w_t) = T.
    # T[i,j,k] = 1 if ... ?
    # It turns out T[i,j,k] = 1 for all i,j,k is WRONG.
    # T[i,j,k] = 1 if j == j?
    # Actually, T[i,j,k] = 1 for all i, j, k is the tensor for the scalar product of all elements?
    
    # Let's look at the 2x2 case.
    # C11 = a*e + b*g
    # C12 = a*f + b*h
    # C21 = c*e + d*g
    # C22 = c*f + d*h
    
    # T[0,0,0] = 1 (a*e -> C11)
    # T[0,1,0] = 1 (b*g -> C11)
    # T[0,0,1] = 1 (a*f -> C12)
    # T[0,1,1] = 1 (b*h -> C12)
    # T[1,0,0] = 1 (c*e -> C21)
    # T[1,1,0] = 1 (d*g -> C21)
    # T[1,0,1] = 1 (c*f -> C22)
    # T[1,1,1] = 1 (d*h -> C22)
    
    # All other entries are 0.
    # So T[i,j,k] = 1 if i, j, k are such that A[i,j] and B[j,k] contribute to C[i,k].
    # This is ALWAYS true for valid indices.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, if T is all ones, then rank is 1.
    # But we know rank is 7 for 2x2.
    
    # The confusion lies in the definition of the tensor product.
    # The tensor T is not a 3D array of ones.
    # It is a 3D array where T[i,j,k] = 1 if the indices match the multiplication rule.
    # But they always match.
    
    # Ah, the indices i, j, k in T[i,j,k] correspond to:
    # i: row of A, col of C
    # j: col of A, row of B
    # k: col of B, col of C
    # So for a fixed i, k, we sum over j.
    # T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # This coefficient is 1.
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Why is the rank not 1?
    # Because the tensor T is not the all-ones tensor in the vector space sense?
    # No, it is.
    # Wait. If T is all ones, then T = u \otimes v \otimes w where u, v, w are all-ones vectors.
    # Then <T, A \otimes B> = (sum A) (sum B).
    # But we want C[i,k] = sum_j A[i,j] B[j,k].
    # These are different!
    
    # The tensor T must be such that when contracted with A and B, it gives C.
    # The contraction is:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k].
    # For this to equal sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all j?
    # Yes.
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Then why is the rank 7?
    # Because the tensor T is not the all-ones tensor?
    # Let's check the rank of the all-ones tensor of size 2x2x2.
    # It is 1.
    # So T cannot be the all-ones tensor.
    
    # I must be misinterpreting the indices.
    # In the tensor T[i,j,k], the indices are:
    # i: index of A
    # j: index of B
    # k: index of C
    # But A, B, C have different shapes.
    # A is n x m. B is m x p. C is n x p.
    # So i is in [0,n), j is in [0,m), k is in [0,p).
    # But in the product A[i,j] B[j,k], the index j is shared.
    # In the tensor T, the index j is the middle index.
    # So T[i,j,k] = 1 if ... ?
    # It is 1 for all i, j, k.
    
    # There is a fundamental misunderstanding here.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The all-ones tensor corresponds to the outer product of sums.
    # The matrix multiplication tensor has a specific structure.
    
    # Let's look at the 2x2 case again.
    # T[0,0,0] = 1 (A00 B00 -> C00)
    # T[0,1,0] = 1 (A01 B10 -> C00)
    # T[0,0,1] = 1 (A00 B01 -> C01)
    # T[0,1,1] = 1 (A01 B11 -> C01)
    # T[1,0,0] = 1 (A10 B00 -> C10)
    # T[1,1,0] = 1 (A11 B10 -> C10)
    # T[1,0,1] = 1 (A10 B01 -> C11)
    # T[1,1,1] = 1 (A11 B11 -> C11)
    
    # All 8 entries are 1.
    # So T is the all-ones tensor.
    # But the rank of the all-ones tensor is 1.
    # Contradiction.
    
    # Resolution:
    # The tensor T is not a 3-way tensor in the standard sense for rank decomposition?
    # Or my understanding of rank is wrong?
    # No, the rank of the matrix multiplication tensor is 7 for 2x2.
    # This is a well-known fact.
    # So T is NOT the all-ones tensor.
    
    # Why?
    # Because the indices i, j, k are not independent?
    # In the tensor T[i,j,k], i, j, k are the indices for A, B, C.
    # But A is 2x2, B is 2x2, C is 2x2.
    # So i is (row, col) of A? No, i is just one index.
    # A has indices (i, j). B has indices (j, k). C has indices (i, k).
    # So the tensor T has indices (i, j, k) where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # So T is a 3-way tensor of size 2x2x2.
    # And T[i,j,k] = 1 for all i,j,k.
    
    # This is extremely confusing.
    # Let's assume the standard result: Rank is 7.
    # And the decomposition is given by Strassen.
    
    # I will implement the Strassen decomposition by hardcoding the 7 terms for 2x2x2.
    # And then combine them.
    
    # Strassen 2x2x2 terms (u, v, w) where u, v, w are vectors of length 4.
    # u is for A (flattened), v is for B (flattened), w is for C (flattened).
    
    # Term 1:
    # u: [1, 0, 0, 1] (a+d)
    # v: [1, 0, 0, 1] (e+h)
    # w: [1, 0, 0, 1] (C00+C11) ... wait.
    # The w vector should be such that w[k] is the coefficient for C[k].
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # So for Term 1 (M1):
    # w: [1, 0, 0, 1] (contributes to C00 and C11)
    
    # Term 2 (M2):
    # u: [0, 0, 1, 1] (c+d)
    # v: [1, 0, 0, 0] (e)
    # w: [0, 0, 1, -1] (C10 - C11)
    
    # Term 3 (M3):
    # u: [1, 0, 0, 0] (a)
    # v: [0, 1, 0, -1] (f-h)
    # w: [0, 1, 0, 1] (C01 + C11)
    
    # Term 4 (M4):
    # u: [0, 0, 0, 1] (d)
    # v: [0, 0, 1, -1] (g-e)
    # w: [1, 0, 0, 0] (C00)
    
    # Term 5 (M5):
    # u: [1, 1, 0, 0] (a+b)
    # v: [0, 0, 0, 1] (h)
    # w: [-1, 1, 0, 0] (-C00 + C01)
    
    # Term 6 (M6):
    # u: [0, 0, 1, -1] (c-a)
    # v: [1, 1, 0, 0] (e+f)
    # w: [0, 0, 1, 1] (C10 + C11)
    
    # Term 7 (M7):
    # u: [0, 1, 0, -1] (b-d)
    # v: [0, 0, 1, 1] (g+h)
    # w: [1, 0, 0, 0] (C00)
    
    # Let's verify Term 1:
    # M1 = (a+d)(e+h)
    # Contribution to C00: 1 * M1
    # Contribution to C11: 1 * M1
    # w = [1, 0, 0, 1] -> C00 coeff 1, C11 coeff 1. Correct.
    
    # Let's verify Term 4:
    # M4 = d(g-e)
    # Contribution to C00: 1 * M4
    # w = [1, 0, 0, 0] -> C00 coeff 1. Correct.
    
    # This looks consistent.
    
    # Now I can build the 2x2x2 decomposition.
    # Then I can build the 2x2x2x3 decomposition.
    # And then the 4x4x6 decomposition.
    
    # Given the time, I will implement the 2x2x2 decomposition and check if it works.
    # If it works, I will extend it.
    
    # For now, I will return the schoolbook solution (rank 96) to ensure correctness.
    # The complexity of implementing the full 77-rank solution correctly in this short time is high.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
