"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # Output is 4x6.
    # We decompose the multiplication into 6 independent matrix-vector products:
    # C[:, k] = A * v_k, where v_k is the k-th column of B (4x1).
    # Each product is a 4x4 * 4x1 multiplication.
    
    # Let's find a rank decomposition for 4x4 * 4x1.
    # A is 4x4, v is 4x1. Result is 4x1.
    # Tensor indices: U[i, r], V[j, r], W[k, r] such that sum_r U[i,r]V[j,r]W[k,r] = delta_{i,k} * (A_{i,j} * v_j)?
    # No, the tensor for matrix-vector multiplication M_{i,j} * v_j -> y_i.
    # The tensor T has indices (i, j, k) where i is row of A, j is col of A (row of v), k is row of result?
    # Wait, the standard matrix multiplication tensor T_{i,j,k} corresponds to A_{i,j} * B_{j,k} -> C_{i,k}.
    # For matrix-vector, B is 4x1, so k is fixed to 0 (or 1).
    # So we need T_{i,j,0} = A_{i,j} v_j -> y_i.
    # This is a rank decomposition of the identity-like tensor?
    # Actually, the operation is y_i = sum_j A_{i,j} v_j.
    # This can be viewed as a tensor with indices (i, j, k) where k is the output index?
    # No, the output is a vector. The "tensor" here is just the multiplication map.
    # However, the problem asks for a decomposition of the *matrix multiplication* tensor.
    # The matrix multiplication tensor for 4x4 * 4x6 has indices (i, j, k) for A_{i,j}, B_{j,k}, C_{i,k}.
    # We can decompose the set of columns.
    # For each column k of B, we compute y^{(k)} = A v^{(k)}.
    # The tensor for this specific column is T^{(k)}_{i,j} = A_{i,j} v^{(k)}_j -> y^{(k)}_i.
    # This is not a standard rank-1 tensor in the 3-index sense unless we fix k.
    # But we can sum over k.
    # The total tensor is T_{i,j,k} = sum_k T^{(k)}_{i,j} * delta_{k, k}? No.
    # T_{i,j,k} is 1 if the operation A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In schoolbook, T_{i,j,k} = 1 for all i,j,k.
    # Rank of T is the minimum r such that T_{i,j,k} = sum_{r=1}^R U_{i,r} V_{j,r} W_{k,r}.
    
    # Strategy:
    # Decompose the 4x4 matrix A into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # v = [v1; v2] where v1, v2 are 2x1 vectors.
    # y = [y1; y2]
    # y1 = A11 v1 + A12 v2
    # y2 = A21 v1 + A22 v2
    
    # We can use Strassen-like decomposition for 2x2 * 2x1?
    # 2x2 * 2x1 requires 4 multiplications in schoolbook.
    # Is there a rank-3 decomposition for 2x2 * 2x1?
    # T_{i,j} for 2x2*2x1:
    # y1 = A11 v1 + A12 v2
    # y2 = A21 v1 + A22 v2
    # This is a bilinear map from (A, v) to y.
    # The tensor has indices (i, j) for A, (j) for v, (i) for y.
    # Wait, the standard tensor decomposition is for 3-mode tensors.
    # Here the "modes" are:
    # Mode 1: Row of A (i)
    # Mode 2: Col of A / Row of v (j)
    # Mode 3: Col of B (k) -> but B is 4x1, so k is fixed?
    # No, the problem is 4x4 * 4x6.
    # So Mode 3 is the column index of B, which ranges 0..5.
    # So T_{i,j,k} = 1 for all i,j,k.
    
    # If we decompose by columns of B:
    # For each k in 0..5, we have a sub-tensor T^{(k)}_{i,j} = 1 (since B_{j,k} is the variable).
    # Actually, T_{i,j,k} = 1.
    # We can write T_{i,j,k} = sum_{r} U_{i,r} V_{j,r} W_{k,r}.
    # If we assume the decomposition is a direct sum over k, i.e., W_{k,r} is non-zero only for one k,
    # then the rank is sum over k of rank(T^{(k)}).
    # T^{(k)} is the tensor for 4x4 * 4x1.
    # What is the rank of the 4x4 * 4x1 tensor?
    # This is the tensor T_{i,j} where the operation is y_i = sum_j A_{i,j} v_j.
    # This is a 2-mode tensor? No, it's a bilinear map.
    # The "rank" of a bilinear map is the tensor rank of the associated 3-tensor if we add a dummy index?
    # No, for matrix-vector multiplication, the tensor is effectively 2-indexed (i,j) for the coefficients,
    # but the output is a vector.
    # Let's look at the standard definition.
    # Matrix multiplication C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T_{i,j,k} = 1.
    # Rank of T_{i,j,k}.
    
    # For 4x4 * 4x1, B is 4x1. Let's call it b.
    # C_{i} = sum_j A_{i,j} b_j.
    # This can be seen as T_{i,j,k} where k is always 0?
    # If we embed it in the 4x4 * 4x4 tensor, the rank of the full tensor is at least the rank of the sub-tensor.
    # But we can decompose the 4x4 * 4x6 tensor as a sum of 6 tensors, each corresponding to one column of B.
    # T_{i,j,k} = sum_{k'} T^{(k')}_{i,j} * delta_{k,k'}.
    # This requires rank(T) <= sum_{k'} rank(T^{(k')}).
    # What is rank(T^{(k')})?
    # T^{(k')}_{i,j} is the tensor for y_i = sum_j A_{i,j} b_j.
    # This is a bilinear form. The tensor rank of a bilinear map A x b -> y is the minimum r such that
    # y_i = sum_{r=1}^R U_{i,r} (sum_j V_{j,r} b_j) ? No.
    # y_i = sum_r U_{i,r} V_{j,r} b_j.
    # This is a decomposition of the identity matrix? No.
    # The coefficients are 1.
    # T_{i,j} = 1.
    # We want T_{i,j} = sum_r U_{i,r} V_{j,r}.
    # This is just the matrix rank of the all-ones matrix!
    # The all-ones matrix 4x4 has rank 1.
    # So T_{i,j} = 1 * 1.
    # U_{i,1} = 1, V_{j,1} = 1.
    # Then y_i = 1 * (sum_j 1 * b_j) = sum_j b_j.
    # This is NOT y_i = sum_j A_{i,j} b_j.
    # The A_{i,j} are variables!
    # The tensor T_{i,j,k} has entries 1, but the variables are A_{i,j} and B_{j,k}.
    # The decomposition is:
    # C_{i,k} = sum_r U_{i,r} V_{j,r} W_{k,r} ... wait, the indices must match.
    # C_{i,k} = sum_{j, r} A_{i,j} B_{j,k} U_{i,r} V_{j,r} W_{k,r}? No.
    # The standard form is:
    # C_{i,k} = sum_{r=1}^R (sum_j A_{i,j} U_{j,r}) (sum_k B_{j,k} V_{k,r})? No.
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) ...
    
    # Correct Strassen form:
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{j,r}) (sum_k B_{j,k} V_{k,r})?
    # No, the indices in U and V must align with A and B.
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{j,r}) (sum_l B_{l,k} V_{l,r})?
    # If U_{j,r} and V_{l,r} are scalars, then:
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{j,r}) (sum_l B_{l,k} V_{l,r}).
    # This equals sum_j,l A_{i,j} B_{l,k} (sum_r U_{j,r} V_{l,r}).
    # We need sum_r U_{j,r} V_{l,r} = delta_{j,l}.
    # This is impossible for r < min(n,m,p) if we need to reconstruct the identity.
    # Wait, the schoolbook decomposition is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This corresponds to r = n*m*p terms? No.
    # Each term is U_{i,j,r} A_{i,j} * V_{j,k,r} B_{j,k}.
    # For schoolbook, for each (i,j,k), we have one term.
    # U_{i,j, (i,j,k)} = 1, V_{j,k, (i,j,k)} = 1, W_{i,k, (i,j,k)} = 1.
    # Then sum_r ... = A_{i,j} B_{j,k}.
    
    # So the question is: can we represent delta_{j,l} with fewer terms?
    # No, the tensor is T_{i,j,k} = 1.
    # We need T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # This is the tensor rank of the all-ones tensor of size 4x4x6.
    # The tensor rank of the all-ones tensor J_{n,m,p} is n*m*p?
    # No, J_{n,m,p} has rank 1?
    # J_{i,j,k} = 1.
    # 1 = 1 * 1 * 1.
    # So U_{i,1}=1, V_{j,1}=1, W_{k,1}=1.
    # Then sum_r U_{i,r} V_{j,r} W_{k,r} = 1.
    # This matches T_{i,j,k} = 1.
    # So the rank of the all-ones tensor is 1.
    # BUT, the matrix multiplication tensor is NOT the all-ones tensor in this context.
    # The tensor T_{i,j,k} defines the structure.
    # The actual computation is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The "tensor" we are decomposing is the one that maps (A, B) to C.
    # The entries of the tensor are the coefficients of the monomials A_{i,j} B_{j,k} in C_{i,k}.
    # For C_{i,k}, the monomial A_{i,j} B_{j,k} appears with coefficient 1.
    # So T_{i,j,k} = 1.
    # If T_{i,j,k} = 1 for all i,j,k, then the rank is 1?
    # No, because the decomposition must be of the form:
    # C_{i,k} = sum_r (linear form in A) * (linear form in B).
    # Specifically:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_l V_{j,l,r} B_{j,l})?
    # No, the indices in U and V must be such that the product gives the correct terms.
    # Standard form:
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{j,r}) (sum_l B_{l,k} V_{l,r}) is WRONG.
    # It must be:
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) (sum_k B_{j,k} V_{j,k,r})? No.
    
    # Let's look at the Strassen algorithm.
    # M1 = (A11 + A22)(B11 + B22)
    # This corresponds to a term where:
    # U_{i,j,r} is non-zero for specific i,j.
    # V_{j,k,r} is non-zero for specific j,k.
    # W_{i,k,r} is non-zero for specific i,k.
    
    # The tensor T_{i,j,k} is defined such that:
    # C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} T_{i,j,k}^{(r)}? No.
    
    # The correct definition:
    # A matrix multiplication algorithm with rank r consists of r triples (U_r, V_r, W_r)
    # where U_r is an n x m matrix, V_r is an m x p matrix, W_r is a p x n matrix.
    # The algorithm computes:
    # C = sum_{r=1}^R (A U_r) (V_r B)^T ? No.
    # C = sum_{r=1}^R (A U_r) * (B V_r^T) ? No.
    
    # The formula is:
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{j,r}) (sum_l B_{l,k} V_{l,r})?
    # Let's check dimensions.
    # A is n x m. U_r is m x 1? No, U_r is a matrix?
    # In Strassen, the "matrices" U, V, W are often 0/1 matrices.
    # The term is (A U_r) (B V_r^T) ?
    # If U_r is m x m and V_r is p x m, then:
    # (A U_r) is n x m.
    # (B V_r^T) is m x p? No, B is m x p. V_r^T is m x p.
    # So (A U_r) (B V_r^T) is n x p.
    # This matches C.
    
    # So the decomposition is:
    # C = sum_{r=1}^R (A U_r) (B V_r^T)
    # where U_r is m x m and V_r is p x m?
    # Wait, the standard Strassen uses 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # M1 = (A11 + A22)(B11 + B22)
    # This is (A U1) (B V1^T) where U1 and V1 are 2x2 matrices?
    # A U1 = [[A11+A22, 0], [0, A11+A22]]? No.
    # A U1 = [[A11, A12], [A21, A22]] * [[1, 0], [1, 0]] = [[A11+A22, 0], [A21+A22, 0]]? No.
    
    # Let's use the index notation.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want C_{i,k} = sum_r U_{i,r} V_{j,r} W_{k,r} ... no.
    # C_{i,k} = sum_r (sum_j A_{i,j} U_{j,r}) (sum_l B_{l,k} V_{l,r})?
    # If this is the form, then:
    # C_{i,k} = sum_r (A U_r)_i (V_r B^T)_k?
    # Let's assume the form is:
    # C = sum_r (A U_r) (B V_r^T)
    # where U_r is m x m, V_r is p x m.
    # Then (A U_r) is n x m. (B V_r^T) is m x p.
    # Product is n x p.
    # This works.
    
    # For 4x4 * 4x6:
    # n=4, m=4, p=6.
    # U_r is 4x4. V_r is 6x4.
    # Rank is R.
    
    # Can we decompose this into 6 independent 4x4 * 4x1 problems?
    # For each column k of B, we compute C[:, k] = A B[:, k].
    # This is a matrix-vector product.
    # C[:, k] = sum_r (A U_r) (B[:, k] V_r^T)?
    # B[:, k] is 4x1. V_r^T is 4x6.
    # B[:, k] V_r^T is 4x6? No.
    # The formula C = sum_r (A U_r) (B V_r^T) implies that the k-th column of C is:
    # C[:, k] = sum_r (A U_r) (B V_r^T)[:, k]
    # (B V_r^T)[:, k] is a column vector.
    # B V_r^T is 4x6. Its k-th column is B * (V_r^T)[:, k] = B * (V_r[k, :])^T.
    # So C[:, k] = sum_r (A U_r) (B v_{r,k}) where v_{r,k} is the k-th row of V_r^T, i.e., the k-th column of V_r?
    # V_r is 6x4. So V_r[k, :] is a row vector of length 4.
    # So C[:, k] = sum_r (A U_r) (B V_r[k, :])^T.
    
    # This couples the columns.
    # However, if we choose U_r and V_r such that the support of V_r is sparse,
    # we might decouple the columns.
    
    # Specifically, if we can find a decomposition where V_r[k, :] is non-zero only for one k,
    # then the rank is the sum of the ranks for each k.
    # For a fixed k, we need to decompose C[:, k] = A B[:, k].
    # This is a bilinear map.
    # What is the rank of the 4x4 * 4x1 tensor?
    # It is the minimum r such that A b = sum_{r=1}^R (A U_r) (b v_r^T) ?
    # No, the form is C = sum_r (A U_r) (B V_r^T).
    # For a single column, B is 4x1. Let's call it b.
    # C = sum_r (A U_r) (b v_r^T) where v_r is a scalar?
    # No, V_r is 6x4. If we restrict to one column, V_r is 1x4.
    # So V_r[k, :] is a 1x4 vector.
    # Let v_r be this 1x4 vector.
    # Then C[:, k] = sum_r (A U_r) (b v_r^T).
    # (b v_r^T) is 4x4.
    # So C[:, k] = sum_r (A U_r) (b v_r^T).
    # This is a sum of matrix products.
    # The rank of the tensor for 4x4 * 4x1 is the minimum r such that
    # A b = sum_{r=1}^R (A U_r) (b v_r^T).
    # This is equivalent to finding the tensor rank of the 3-tensor T_{i,j,l} = delta_{i,l} ?
    # No, the tensor is defined by the coefficients.
    # The operation is linear in A and b.
    # The tensor rank is the minimum r such that
    # T_{i,j,l} = sum_r U_{i,j,r} V_{j,l,r}.
    # Here T_{i,j,l} = 1 if i=l? No.
    # C_i = sum_j A_{i,j} b_j.
    # So the coefficient of A_{i,j} b_j in C_i is 1.
    # The coefficient of A_{i,j} b_l in C_k is 0 if j != l or i != k.
    # So T_{i,j,k} = delta_{i,k} delta_{j,l}? No, l is the index of b.
    # Let's say b has index j.
    # C_i = sum_j A_{i,j} b_j.
    # The tensor is T_{i,j,k} where i is row of A, j is col of A (row of b), k is row of C.
    # T_{i,j,k} = 1 if i=k, else 0? No, j is summed over.
    # The tensor is 2-indexed? No, it's a bilinear map.
    # The rank of the bilinear map A x b -> C is the tensor rank of the 3-tensor T_{i,j,k}
    # where T_{i,j,k} = 1 if the term A_{i,j} b_j contributes to C_k.
    # A_{i,j} b_j contributes to C_i.
    # So T_{i,j,k} = 1 if k=i, else 0.
    # This is the identity tensor!
    # The rank of the identity tensor I_{n,n,n} is n^2?
    # No, the rank of the identity tensor (3-way) is n^2?
    # For n=2, rank is 4? No, Strassen says 7 for 2x2*2x2.
    # The identity tensor is not the matrix multiplication tensor.
    
    # I am confusing myself.
    # Let's stick to the known result:
    # Rank of 4x4 * 4x4 is at most 49 (Strassen).
    # Rank of 4x4 * 4x6 can be bounded by 6 * rank(4x4 * 4x1).
    # What is rank(4x4 * 4x1)?
    # It is the rank of the tensor T_{i,j} = 1 (for the coefficients)?
    # No, it is the rank of the bilinear map.
    # For n x m * m x 1, the rank is n*m?
    # No, it is at most n*m.
    # Can it be lower?
    # For 2x2 * 2x1, rank is 4?
    # y1 = A11 v1 + A12 v2
    # y2 = A21 v1 + A22 v2
    # This requires 4 multiplications.
    # Is there a rank-3 decomposition?
    # Probably not, as it's a full rank system.
    
    # So for 4x4 * 4x1, rank is likely 16.
    # Then for 4x4 * 4x6, rank is 6 * 16 = 96.
    # This is the same as schoolbook.
    
    # So the "direct sum" strategy does not improve on schoolbook.
    # We need to use the full 4x4 * 4x6 structure.
    # We can use Strassen on 2x2 blocks.
    # A is 4x4, B is 4x6.
    # Split A into four 2x2 blocks.
    # Split B into two 2x6 blocks? Or six 4x1 blocks?
    # If we split B into three 4x2 blocks, then we have 4x4 * 4x2 multiplications.
    # Rank of 4x4 * 4x2?
    # This is 2 * rank(4x4 * 4x1) = 32?
    # Or can we do better?
    
    # Let's use the known Strassen decomposition for 4x4 * 4x4 (rank 49) and adapt it.
    # Or use the 2x2 * 2x2 Strassen (rank 7) on blocks.
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x3? No, B is 4x6)
    # Let B be 4x6. Split into top 2x6 and bottom 2x6.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # where B1, B2 are 2x6.
    # This is a 2x2 block multiplication where the "entries" are 2x6 matrices.
    # We can apply Strassen to the 2x2 block structure.
    # M1 = (A11 + A22)(B1 + B2)
    # M2 = (A21 + A22) B1
    # M3 = A11 (B1 - B2)
    # M4 = A22 (B2 - B1)
    # M5 = (A11 + A12) B2
    # M6 = (A21 - A11)(B1 + B2)
    # M7 = (A22 - A12)(B2 + B1)
    
    # Each Mi is a 2x6 matrix.
    # The multiplications are 2x2 * 2x6.
    # Rank of 2x2 * 2x6?
    # This is 3 * rank(2x2 * 2x2) = 3 * 7 = 21?
    # Or 2x2 * 2x6 can be done in 2x2 * 2x2 (rank 7) + 2x2 * 2x2 (rank 7) + 2x2 * 2x2 (rank 7) = 21.
    # So each Mi requires rank 21.
    # There are 7 Mi's.
    # Total rank = 7 * 21 = 147.
    # This is worse than 96.
    
    # Why? Because 2x2 * 2x6 is not 3 * 2x2 * 2x2.
    # 2x2 * 2x6 is equivalent to 3 independent 2x2 * 2x2 multiplications.
    # So rank is 3 * 7 = 21.
    # 7 * 21 = 147.
    
    # Schoolbook for 4x4 * 4x6 is 4*4*6 = 96.
    # So Strassen on 2x2 blocks is worse.
    
    # What if we use Strassen on 4x4 * 4x4 (rank 49) and then extend to 4x6?
    # 4x4 * 4x6 = 4x4 * 4x4 + 4x4 * 4x2.
    # Rank <= 49 + 2 * rank(4x4 * 4x2)?
    # rank(4x4 * 4x2) <= 2 * rank(4x4 * 4x1) = 32.
    # Total <= 49 + 64 = 113.
    # Still worse than 96.
    
    # It seems 96 is hard to beat with simple block decompositions.
    # However, the prompt suggests that a lower rank exists.
    # "Known results suggest rank around 60-80 for 4x4x4".
    # For 4x4x4, rank is 49 (Strassen).
    # For 4x4x6, maybe we can do better than 96.
    
    # Let's try to return the schoolbook decomposition as a safe baseline, as the current program does.
    # The current program returns rank 96.
    # Score 1.0.
    
    # If I can't find a better one, I should return the schoolbook.
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
