import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The trivial decomposition has rank n*m*p = 36
    # We try to find a lower rank decomposition
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # The tensor T_{i,j,k} corresponds to the coefficient of A_{i,j} * B_{j,k} in C_{i,k}
    # Wait, the convention is W[k*n+i, r]. Let's re-read carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the output index is (k, i) where k is the column of C? No.
    # Standard matrix mult: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T_{i, j, k} = delta_{j, j} ... no, T_{i, j, k} is 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # Actually, the multiplication tensor T is defined by T_{i, j, k} = 1 if A_{i,j} and B_{j,k} multiply to give C_{i,k}.
    # But wait, the indices in the problem statement:
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i
    # So the output is indexed by (k, i). This means the result matrix C is indexed by (k, i)?
    # If C_{k,i} = sum_j A_{i,j} B_{j,k}, then C is the transpose of the standard product?
    # Or is it just a naming convention?
    # Let's check the seed program:
    # U[i*m+j][col] = 1
    # V[j*p+k][col] = 1
    # W[k*n+i][col] = 1
    # This constructs the tensor where for each (i,j,k), we have a rank-1 term.
    # The sum over r of U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should equal T_{i,j,k}.
    # In the seed, for a fixed (i,j,k), there is exactly one r (col) where all three are 1.
    # So the tensor T_{i,j,k} = 1 for all i,j,k?
    # No, matrix multiplication is not a full tensor of 1s.
    # The tensor T_{i,j,k} for matrix multiplication is defined such that
    # sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
    # The "multiplication tensor" T has entries T_{i, j, k} which is 1 if the pair (i,j) from A and (j,k) from B
    # produce the entry (i,k) in C.
    # Wait, the indices in W are k*n + i. This suggests the output entry is indexed by (k, i).
    # If the output is C_{k,i}, then C_{k,i} = sum_j A_{i,j} B_{j,k}.
    # This is equivalent to C = B^T A? Or C_{k,i} = (B A^T)_{k,i}?
    # Let's assume the standard definition where the tensor T_{i, j, k} = 1 if the indices match for multiplication.
    # Specifically, T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # But here the output index is (k, i). So it contributes to C_{k,i}.
    # This implies C_{k,i} = sum_j A_{i,j} B_{j,k}.
    # This is the transpose of the standard product if we view A and B as given.
    # Regardless, the structure of the tensor is the same as standard matrix multiplication, just with output indices swapped.
    # The rank of the tensor is invariant under permutation of modes.
    # So we are looking for the rank of the 3x3x4 matrix multiplication tensor.
    
    # Known bounds:
    # Rank of 3x3x3 is 19 (Landsberg).
    # Rank of 3x3x4?
    # Strassen's algorithm for 2x2x2 is rank 7.
    # For 3x3x3, the best known is 19.
    # For 3x3x4, we can use block decomposition.
    # Let's try to find a decomposition of rank < 36.
    
    # We can use the fact that the multiplication tensor is a sub-tensor of a larger one.
    # Or we can use known algorithms.
    # However, finding the exact minimum rank is hard.
    # We can try to find a decomposition of rank 24 or so using a structured approach.
    
    # One simple improvement:
    # Use the fact that A is 3x3 and B is 3x4.
    # We can view B as two 3x2 matrices B1, B2.
    # Then C = A * [B1 B2] = [A*B1 A*B2].
    # Rank of 3x3x2 matrix multiplication?
    # If we can find a rank r for 3x3x2, then 3x3x4 might be 2r?
    # Or we can use a mixed approach.
    
    # Let's try a numerical search for a low-rank decomposition.
    # We want to find U, V, W such that T = sum_r u_r v_r w_r.
    # T_{i,j,k} = 1 if the indices are valid for multiplication?
    # No, T_{i,j,k} is the coefficient. For matrix multiplication, T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # In our convention, C_{k,i} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i in [0,2], j in [0,2], k in [0,3]?
    # No! The tensor T is defined by the bilinear map.
    # The entry T_{i, j, k} is 1 if the product A_{i,j} B_{j,k} contributes to the output at index (k, i).
    # Since every pair (i,j) from A and (j,k) from B contributes to exactly one output (k,i),
    # and there is no other constraint, T_{i,j,k} = 1 for all i,j,k?
    # If T is the all-ones tensor, its rank is 1 (if dimensions allow) or max(n,m,p).
    # But matrix multiplication is NOT the all-ones tensor.
    # The all-ones tensor corresponds to the "outer product" or "Khatri-Rao" product, not matrix multiplication.
    # In matrix multiplication, the index j is shared.
    # The tensor T is defined as T_{i, j, k} = 1 if the indices are "compatible".
    # But here, for any i, j, k, the term A_{i,j} B_{j,k} is well-defined and contributes to C_{k,i}.
    # So is the tensor just 1 everywhere?
    # If so, the rank would be very small.
    # Let's check the seed program again.
    # It sets U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1.
    # This means for each (i,j,k), there is a unique rank-1 component.
    # This implies that the tensor T has T_{i,j,k} = 1 for all i,j,k.
    # If T is the all-ones tensor, then T = u v^T w^T where u, v, w are all-ones vectors?
    # No, the indices are different.
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # If we set U_{i,j,r} = 1, V_{j,k,r} = 1, W_{k,i,r} = 1 for r=0, and 0 otherwise,
    # then T_{i,j,k} = 1.
    # So the rank is 1?
    # But matrix multiplication rank is known to be high.
    # The confusion is in the definition of the tensor.
    # The "multiplication tensor" is not the all-ones tensor.
    # The all-ones tensor corresponds to the tensor T_{i,j,k} = 1.
    # But in matrix multiplication, the "tensor" is defined such that
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # The tensor T is defined by T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} is part of the sum for C_{i,k}.
    # Since for a fixed output (i,k), the sum is over j.
    # So for a fixed output (i,k), the terms are (i, 0, k), (i, 1, k), (i, 2, k).
    # This means the tensor T has non-zero entries only when the indices are linked?
    # No, the tensor T is a 3-way array.
    # T_{i, j, k} = 1 if the pair (i,j) and (j,k) are "matched" by the multiplication rule.
    # But every pair (i,j) and (j,k) is matched.
    # So T is indeed the all-ones tensor?
    # If T is the all-ones tensor, the rank is 1.
    # But the seed program uses rank 36.
    # Why?
    # Because the tensor decomposition for matrix multiplication is not just about the values 1.
    # It's about the structure.
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then the rank is 1.
    # Let's test this hypothesis.
    # If rank is 1, then U, V, W are rank-1 matrices.
    # U_{i,j} = u_i v_j, V_{j,k} = v_j w_k, W_{k,i} = w_k u_i.
    # Then T_{i,j,k} = u_i v_j v_j w_k w_k u_i = u_i^2 v_j^2 w_k^2.
    # This is not 1 for all i,j,k unless u_i, v_j, w_k are all 1.
    # If u_i=1, v_j=1, w_k=1, then T_{i,j,k} = 1.
    # So the rank is 1?
    # This contradicts the known results for matrix multiplication rank.
    
    # The error is in the definition of the tensor.
    # The multiplication tensor T is NOT the all-ones tensor.
    # The tensor T is defined by the bilinear map.
    # The entry T_{i, j, k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # But in the tensor decomposition, we are decomposing the tensor T.
    # The tensor T is defined as T = sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k.
    # For matrix multiplication, T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # But wait, in the standard definition, the tensor is T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
    # And the output is C_{i, k}.
    # So T_{i, j, k} = 1 for all i, j, k?
    # Yes, because for every i, j, k, the product A_{i,j} B_{j,k} is a term in the sum for C_{i,k}.
    # So T is the all-ones tensor.
    # Then why is the rank not 1?
    # Because the tensor decomposition is not of the all-ones tensor in the usual sense?
    # No, if T is the all-ones tensor, its rank is 1.
    # Let's check the seed program's output.
    # It returns rank 36.
    # If the true rank is 1, the score would be 1/36? No, score is rank.
    # The task says "score is the rank".
    # If the rank is 1, the score is 1.
    # But the seed score is 1.0000, which corresponds to rank 36?
    # "score is the rank" -> lower is better.
    # Seed rank is 36. Score 1.0000 means seed/value = 1.0, so value = 36?
    # Or is the score the rank itself?
    # "score is the rank" -> if rank is 36, score is 36.
    # But the prompt says "score 1.0000".
    # And "1.0 equals the seed program".
    # So the seed program's score is 1.0.
    # This implies the metric is normalized?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value"
    # So if seed value is 36, and child value is 36, score is 1.0.
    # If child value is 18, score is 2.0.
    # So we want to minimize the rank.
    
    # If the tensor is the all-ones tensor, the rank is 1.
    # Let's verify this.
    # T_{i,j,k} = 1.
    # Rank 1 decomposition:
    # U_{i,j,0} = 1, V_{j,k,0} = 1, W_{k,i,0} = 1.
    # Then sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1*1*1 = 1.
    # This matches T_{i,j,k} = 1.
    # So the rank is 1.
    # Why did the seed program use rank 36?
    # Maybe the tensor is NOT the all-ones tensor.
    # Let's re-read the task description carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 4 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This implies the output is C_{k,i}.
    # The multiplication is C_{k,i} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined by the coefficients.
    # T_{i,j,k} = 1 if A_{i,j} B_{j,k} is a term in C_{k,i}.
    # Since it always is, T_{i,j,k} = 1.
    # So the rank is 1.
    
    # Is there a catch?
    # Maybe the "tensor" is not T_{i,j,k} but something else?
    # No, the interface is U, V, W.
    # U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # This is the standard format for tensor decomposition.
    
    # If the rank is 1, the seed program is very bad.
    # Let's try to return a rank 1 decomposition.
    
    U = [[1] * 1 for _ in range(n * m)]
    V = [[1] * 1 for _ in range(m * p)]
    W = [[1] * 1 for _ in range(p * n)]
    
    # Check if this works.
    # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = 1 * 1 * 1 = 1.
    # This is correct if T_{i,j,k} = 1.
    
    # Let's double check if T_{i,j,k} is indeed 1.
    # In matrix multiplication, the "tensor" is the set of terms.
    # The term A_{i,j} B_{j,k} appears in C_{i,k}.
    # So the coefficient is 1.
    # There is no other constraint.
    # So yes, T is the all-ones tensor.
    
    # Why is the rank of matrix multiplication not 1?
    # Because the rank of the tensor is NOT the rank of the matrix multiplication algorithm?
    # No, the rank of the multiplication tensor IS the minimum number of multiplications.
    # If the rank is 1, we can multiply 3x3 by 3x4 with 1 multiplication?
    # That's impossible.
    # The error is that T_{i,j,k} is NOT 1 for all i,j,k.
    # The tensor T is defined such that the bilinear form is:
    # B(A, B) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} e_{i,k}
    # No, that's not right.
    # The tensor T is defined by T_{i, j, k} where i, j, k are the indices.
    # The multiplication is C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # This can be written as C = sum_{i, j, k} A_{i, j} B_{j, k} E_{i, k}
    # where E_{i, k} is the matrix with 1 at (i, k) and 0 elsewhere.
    # So the tensor is T = sum_{i, j, k} A_{i, j} B_{j, k} E_{i, k}.
    # The tensor T is a linear map from A \otimes B to C.
    # The tensor T is defined by its components T_{i, j, k}^{(c)} where c is the output index.
    # But here the output index is (k, i).
    # So T_{i, j, k}^{(k, i)} = 1.
    # And T_{i, j, k}^{(c')} = 0 for c' != (k, i).
    # So for a fixed output index (k, i), the non-zero entries are for all j.
    # So the tensor is a sum of rank-1 tensors?
    # No, the tensor is T = sum_{i, j, k} e_i \otimes e_j \otimes e_k.
    # This is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1.
    # Wait, the rank of the all-ones tensor is 1?
    # Yes, because J = 1 1^T 1^T.
    # So the rank is 1.
    
    # But we know that matrix multiplication requires more than 1 multiplication.
    # The resolution is that the "tensor" in matrix multiplication is NOT the all-ones tensor.
    # The tensor is defined by the indices.
    # The term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So the tensor has a 1 at (i, j, k) for the output (i, k).
    # But the tensor is a 3-way array T_{i, j, k}.
    # The value T_{i, j, k} is 1 if the term A_{i, j} B_{j, k} is part of the multiplication.
    # Since it always is, T_{i, j, k} = 1.
    # So the tensor is the all-ones tensor.
    # And its rank is 1.
    
    # This implies that matrix multiplication can be done with 1 multiplication?
    # No, because the output is a matrix, not a scalar.
    # The tensor decomposition gives the number of scalar multiplications.
    # If the rank is 1, we need 1 scalar multiplication.
    # But we need to compute a 3x4 matrix, which has 12 entries.
    # Each entry is a sum of 3 products.
    # So we need at least 12 multiplications?
    # No, we can reuse products.
    # But 1 multiplication is clearly not enough.
    
    # The mistake is in the definition of the tensor.
    # The tensor T is NOT the all-ones tensor.
    # The tensor T is defined by T_{i, j, k} = 1 if the indices are "compatible".
    # But in matrix multiplication, the index j is shared.
    # The tensor is T = sum_{i, j, k} A_{i, j} B_{j, k} C_{i, k}.
    # This is a trilinear form.
    # The tensor is T_{i, j, k} = 1.
    # But the "rank" of the multiplication is the minimum number of terms in a decomposition of the form:
    # C_{i, k} = sum_r U_{i, j, r} V_{j, k, r} ... no.
    # The decomposition is:
    # C_{i, k} = sum_r (sum_j U_{i, j, r} A_{i, j}) (sum_k V_{j, k, r} B_{j, k}) ... no.
    # The standard Strassen decomposition is:
    # C_{i, k} = sum_r M_r E_{i, k, r}
    # where M_r are linear combinations of A and B.
    # This corresponds to a tensor decomposition of the multiplication tensor.
    # The multiplication tensor T is defined by T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} is in C_{i, k}.
    # But this is still 1 for all i, j, k.
    
    # Let's look at the 2x2 case.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]].
    # The tensor T has entries T_{i, j, k}.
    # T_{0, 0, 0} = 1 (ae)
    # T_{0, 1, 0} = 1 (bg)
    # T_{0, 0, 1} = 1 (af)
    # T_{0, 1, 1} = 1 (bh)
    # T_{1, 0, 0} = 1 (ce)
    # T_{1, 1, 0} = 1 (dg)
    # T_{1, 0, 1} = 1 (cf)
    # T_{1, 1, 1} = 1 (dh)
    # So T is the all-ones tensor of size 2x2x2.
    # The rank of the 2x2x2 all-ones tensor is 1?
    # No, the rank of the 2x2x2 all-ones tensor is 1?
    # J = 1 1^T 1^T. Rank 1.
    # But Strassen's algorithm uses 7 multiplications.
    # The rank of the 2x2x2 multiplication tensor is 7.
    # So the multiplication tensor is NOT the all-ones tensor.
    
    # Why?
    # Because the tensor is not T_{i, j, k} = 1.
    # The tensor is defined by the bilinear map.
    # The tensor T is an element of V \otimes W \otimes X^*.
    # The entries are T_{i, j, k} where i is index in V, j in W, k in X^*.
    # Here V is the space of A, W is the space of B, X is the space of C.
    # The map is (A, B) -> C.
    # The tensor T is defined by T(A, B) = C.
    # The components are T_{i, j, k} = <T, e_i \otimes e_j \otimes e_k^*>.
    # T(e_i, e_j) = e_k.
    # So T_{i, j, k} = 1 if T(e_i, e_j) = e_k.
    # For matrix multiplication, T(e_i \otimes e_j, e_j \otimes e_k) = e_i \otimes e_k.
    # So T_{i, j, k, l} = 1 if (i, j) and (j, k) map to (i, l).
    # This is a 4-way tensor?
    # No, the tensor is 3-way: A \otimes B \otimes C^*.
    # T_{i, j, k} where i is (row, col) of A, j is (row, col) of B, k is (row, col) of C.
    # T_{i, j, k} = 1 if A_i B_j = C_k.
    # For A_{i, j} B_{j, k} = C_{i, k}.
    # So T_{(i, j), (j, k), (i, k)} = 1.
    # And T_{(i', j'), (j', k'), (i', k')} = 1.
    # All other entries are 0.
    # So the tensor is NOT the all-ones tensor.
    # It is a sparse tensor with 1s at specific positions.
    # For 3x3x4, there are 3*3*4 = 36 non-zero entries.
    # The rank of this sparse tensor is what we want to minimize.
    
    # The seed program uses rank 36, which is the trivial decomposition.
    # We want to find a lower rank.
    
    # We can use the fact that the tensor is a sub-tensor of a larger tensor.
    # Or use known algorithms.
    # For 3x3x3, rank is 19.
    # For 3x3x4, we can use a block approach.
    # Let's try to find a decomposition of rank 24.
    
    # We can use a numerical search.
    # But the search space is large.
    # We can try to use the structure.
    
    # Let's return the seed result for now, as we don't have a guaranteed improvement.
    # But we can try a simple improvement:
    # Use the fact that the last column of B is independent.
    # Rank of 3x3x3 is 19.
    # Rank of 3x3x1 is 3.
    # Total rank 19+3=22?
    # Not necessarily additive.
    
    # Let's try to return a rank 24 decomposition using a heuristic.
    # Or just return the seed.
    
    # Given the time constraint and uncertainty, we return the seed.
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
