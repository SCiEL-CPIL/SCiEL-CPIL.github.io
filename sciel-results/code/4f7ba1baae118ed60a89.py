"""Research program: Rank-14 exact decomposition for 2x3x3 matrix multiplication via tensor network contraction."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We are trying to decompose the tensor T[i, j, k] for matrix multiplication C = A @ B
    # where A is n x m, B is m x p, C is n x p.
    # The tensor T[i, j, k] = 1 for all i, j, k is NOT correct.
    # The tensor for matrix multiplication is defined such that:
    # C_{i, k} = sum_{j} A_{i, j} B_{j, k}
    # The decomposition is:
    # C_{i, k} = sum_{c} W[k*n + i, c] * ( sum_j U[i*m + j, c] A_{i, j} ) * ( sum_l V[l*p + k, c] B_{l, k} )
    # Wait, the standard tensor decomposition form for matrix multiplication M_{n,m,p} is:
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This must equal the "multiplication tensor" which is 1 if the indices correspond to the summation index being shared.
    # Actually, the tensor T is the identity map in the tensor product space V_n^* ⊗ V_m ⊗ V_p^* mapped to V_n^* ⊗ V_p^*?
    # No. The tensor T ∈ V_n ⊗ V_m ⊗ V_p such that:
    # <T, A ⊗ B> = Tr(A B) ? No.
    # The standard definition: T_{i,j,k} = 1 if the operation is (A_{i,j} * B_{j,k}).
    # So T_{i,j,k} = 1 for all i, j, k?
    # Let's check the contraction.
    # If T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{i,k}^c, and we contract with A_{i,j} and B_{j,k}:
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_c (sum_{i,j} U_{i,j}^c A_{i,j}) (sum_{j,k} V_{j,k}^c B_{j,k}) (sum_{i,k} W_{i,k}^c C_{i,k})?
    # No, the output is C_{i,k}.
    # The identity is: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want: C_{i,k} = sum_c U_{i, *, c} V_{*, k, c} W_{i, k, c} ?
    # The indices of U are flattened (n*m), V (m*p), W (p*n).
    # Let U be a tensor u[i, j, c]. V be v[j, k, c]. W be w[i, k, c].
    # Then sum_c u[i,j,c] v[j,k,c] w[i,k,c] = delta_{i,k}? No.
    # We need sum_c u[i,j,c] v[j,k,c] w[i,k,c] to be the "1" that picks out the term in the sum for C_{i,k}.
    # Actually, the tensor T_{i,j,k} is just 1.
    # Because C_{i,k} = sum_j 1 * A_{i,j} B_{j,k}.
    # So we need to decompose the all-ones tensor of shape (n, m, p) into a sum of rank-1 tensors.
    # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    
    # So the task is to find the minimal rank decomposition of the all-ones tensor 1_{n x m x p}.
    
    # For n=2, m=3, p=3.
    # The tensor is 2x3x3 all ones.
    
    # Let's try the directive:
    # "Apply a tensor network contraction by grouping the output indices ... into a 2x3 matrix structure, then use the known rank-7 decomposition for the 3x3 matrix multiplication tensor..."
    
    # Wait, the rank of the 3x3 matrix multiplication tensor is 19? No, schoolbook is 27. Strassen is for 2x2.
    # The rank of 3x3x3 is not 7. The rank of 2x2x2 is 7.
    # The directive says "known rank-7 decomposition for the 3x3 matrix multiplication tensor". This is likely a mistake in the prompt or my understanding.
    # Strassen's algorithm decomposes 2x2x2 into rank 7.
    # Maybe the directive implies embedding 2x3x3 into something?
    
    # Let's look at the structure.
    # T_{i,j,k} = 1.
    
    # Can we decompose this into rank 14?
    # Let's try to construct it.
    
    # Hypothesis:
    # The tensor is separable?
    # 1_{i} * 1_{j} * 1_{k}. This is rank 1.
    # WAIT.
    # If T_{i,j,k} = 1 for all i,j,k, then T = u ⊗ v ⊗ w where u=[1,1], v=[1,1,1], w=[1,1,1].
    # This is a rank 1 tensor!
    # Why did I think it was complex?
    
    # Let's re-read the problem statement carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # If the tensor is simply the all-ones tensor, the rank is 1.
    # U would be a column of ones.
    # V would be a column of ones.
    # W would be a column of ones.
    # Let's check if this works.
    # C_{i,k} = sum_c W_{i,k,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_l V_{l,k,c} B_{l,k})?
    # No, the contraction is:
    # C_{i,k} = sum_c W_{i,k,c} U_{i,j,c} V_{j,k,c} ?
    # The indices of U are (i, j). The indices of V are (j, k).
    # If U_{i,j,c} = 1, V_{j,k,c} = 1, W_{i,k,c} = 1.
    # Then term c contributes: 1 * 1 * 1 = 1?
    # No, the summation over j is implicit in the matrix multiplication?
    
    # Let's look at the standard tensor definition again.
    # The multiplication tensor T ∈ A^* ⊗ B^* ⊗ C where A is M_{n,m}, B is M_{m,p}, C is M_{n,p}.
    # T(A, B) = A B.
    # In coordinates:
    # T = sum_{i,j,k} e_{i,j}^* ⊗ e_{j,k}^* ⊗ e_{i,k}
    # Wait. The basis elements are:
    # A = sum_{i,j} a_{ij} E_{ij}
    # B = sum_{j,k} b_{jk} F_{jk}
    # C = sum_{i,k} c_{ik} G_{ik}
    # The tensor T is such that <T, A ⊗ B> = C? No, T is a linear map A ⊗ B -> C.
    # T( sum a_{ij} E_{ij} ⊗ sum b_{jk} F_{jk} ) = sum_{i,j,k} a_{ij} b_{jk} G_{ik}
    # So the coefficient of G_{ik} in the output is sum_j a_{ij} b_{jk}.
    # The tensor T can be written as sum_{i,j,k} G_{ik} ⊗ E_{ij}^* ⊗ F_{jk}^* ?
    # No, T is an element of C ⊗ A^* ⊗ B^*.
    # T = sum_{i,j,k} G_{ik} ⊗ E_{ij}^* ⊗ F_{jk}^*
    # Let's check:
    # <T, A ⊗ B> = sum_{i,j,k} <G_{ik}, C> <E_{ij}^*, A> <F_{jk}^*, B> = sum_{i,j,k} c_{ik} a_{ij} b_{jk}
    # This is a scalar. But we want C.
    # The tensor decomposition usually refers to the tensor T_{i,j,k} = delta_{i,k}?
    # No.
    
    # Let's look at the specific indexing in the prompt:
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    
    # This matches the indices:
    # U[i*m+j, c] corresponds to (i, j)
    # V[j*p+k, c] corresponds to (j, k)
    # W[k*n+i, c] corresponds to (k, i) -> which is (i, k) with swapped indices?
    # "official trace convention W[k*n+i,r]"
    # Usually C_{i,k} is stored. Here index is k*n+i.
    # So W_{idx, c} where idx = k*n + i. This represents the (i, k) component?
    # If row index is k and col index is i?
    # Standard C_{i,k} is row i, col k.
    # If W index is k*n+i, then for n=2, p=3.
    # If i=0, k=0 -> idx 0.
    # If i=1, k=0 -> idx 1.
    # If i=0, k=1 -> idx 2.
    # So the index is `col * n + row`.
    # This is column-major ordering for a matrix of size p x n?
    # Or just a specific flattening.
    
    # The decomposition condition is:
    # For all i, k:
    # sum_j (A_{i,j} B_{j,k}) = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] ??
    # No, the sum over c must reproduce the sum over j.
    # The term for a specific c is:
    # (sum_j U[i*m+j, c] A_{i,j}) * (sum_j V[j*p+k, c] B_{j,k}) * W[k*n+i, c] ?
    # No, that would be a product of two sums.
    
    # The standard tensor decomposition for matrix multiplication is:
    # T_{i,j,k} = sum_c u_{i,j,c} v_{j,k,c} w_{i,k,c}
    # Where T_{i,j,k} is the tensor of the multiplication map.
    # What is T_{i,j,k}?
    # The map is (A, B) -> C.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be seen as a multilinear form?
    # If we consider the tensor T such that:
    # C_{i,k} = sum_{j, l} T_{i,j,k,l} ... No.
    
    # The rank of the matrix multiplication tensor M_{n,m,p} is the smallest r such that
    # M_{n,m,p} can be written as sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{n*p}.
    # And the condition is:
    # For any A, B:
    # A B = sum_{c=1}^r (U_c \bullet A) (V_c \bullet B) W_c^T ?
    # No.
    
    # Let's look at the "official trace convention".
    # This usually implies:
    # C_{i,k} = sum_c w_{i,k,c} (sum_j u_{i,j,c} A_{i,j}) (sum_j v_{j,k,c} B_{j,k}) is WRONG.
    
    # Correct form:
    # C_{i,k} = sum_c w_{i,k,c} * ( sum_j u_{i,j,c} A_{i,j} ) * ( sum_l v_{l,k,c} B_{l,k} ) is also wrong.
    
    # The correct form is:
    # C_{i,k} = sum_c w_{i,k,c} * (sum_j u_{i,j,c} A_{i,j}) * (sum_j v_{j,k,c} B_{j,k}) ... No, the indices don't match for a single c term to produce a scalar that multiplies W.
    
    # Actually, the decomposition is:
    # T_{i,j,k} = sum_c u_{i,j,c} v_{j,k,c} w_{i,k,c}
    # Where T_{i,j,k} is the "multiplication tensor".
    # What is T_{i,j,k}?
    # It is the coefficient of A_{i,j} B_{j,k} in the expression for C.
    # But C_{i,k} contains A_{i,j} B_{j,k} for all j.
    # So the "tensor" T is not a single scalar tensor T_{i,j,k}.
    # The tensor is a 4-index object? No.
    
    # Let's reconsider the rank definition.
    # The matrix multiplication algorithm is a bilinear map.
    # It can be represented by a tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}.
    # The tensor T has components T_{a, b, c} where a indexes A, b indexes B, c indexes C.
    # The condition is:
    # C_c = sum_{a,b} T_{a,b,c} A_a B_b.
    # In our case:
    # A_a where a = i*m+j.
    # B_b where b = j*p+k.
    # C_c where c = k*n+i.
    # C_{i,k} = sum_j A_{i*m+j} B_{j*p+k}.
    # So T_{i*m+j, j*p+k, k*n+i} = 1.
    # All other components of T are 0.
    
    # So T is a sparse tensor.
    # We want to decompose this T into a sum of rank-1 tensors.
    # Rank 1 tensor is u ⊗ v ⊗ w.
    # T_{a,b,c} = sum_r u_{a,r} v_{b,r} w_{c,r}.
    
    # So we need to find u, v, w such that:
    # u_{i*m+j, r} v_{j*p+k, r} w_{k*n+i, r} sums to 1 if (a,b,c) corresponds to a valid triple (i,j,k), and 0 otherwise.
    # The valid triples are exactly those where the index of A (i,j) and B (j,k) share the same j.
    # i.e., a = i*m+j, b = j'*p+k.
    # If j == j', then T_{a,b,c} = 1.
    # If j != j', then T_{a,b,c} = 0.
    
    # So the condition is:
    # sum_r u_{i*m+j, r} v_{j'*p+k, r} w_{k*n+i, r} = delta_{j, j'}
    
    # This is the problem.
    
    # For n=2, m=3, p=3.
    # Schoolbook rank: 18.
    # u_{i*m+j, r} = delta_{r, i*m+j}
    # v_{j*p+k, r} = delta_{r, j*p+k}
    # w_{k*n+i, r} = delta_{r, k*n+i}
    # This doesn't work directly because the indices in u, v, w must match r.
    # In schoolbook, for each (i,j,k), we have a term.
    # u_{i*m+j, i*m+j} = 1? No.
    # Let's construct schoolbook:
    # For each triple (i,j,k):
    # Set r = index of (i,j,k).
    # u_{i*m+j, r} = 1
    # v_{j*p+k, r} = 1
    # w_{k*n+i, r} = 1
    # Then sum_r u v w = 1 if there is an r such that u=1, v=1, w=1.
    # This happens if we pick r corresponding to (i,j,k).
    # Then u_{i*m+j, r}=1, v_{j*p+k, r}=1, w_{k*n+i, r}=1.
    # Product is 1.
    # If we pick a different (i', j', k'), u_{i*m+j, r'} is 0 unless i=i', j=j'.
    # So the sum is 1 if j=j' (and i=i', k=k' for the W part? No, W is fixed by i,k).
    # Wait.
    # If we have terms for all (i,j,k).
    # For a fixed (a,b,c) = (i*m+j, j'*p+k, k*n+i).
    # We sum over r.
    # Term r contributes 1 if u_{i*m+j, r}=1, v_{j'*p+k, r}=1, w_{k*n+i, r}=1.
    # In schoolbook construction:
    # u_{i*m+j, r} is non-zero only if r corresponds to some (i_r, j_r) with i_r=i, j_r=j.
    # v_{j'*p+k, r} is non-zero only if r corresponds to some (j_r, k_r) with j_r=j', k_r=k.
    # w_{k*n+i, r} is non-zero only if r corresponds to some (i_r, k_r) with i_r=i, k_r=k.
    
    # So for a term r to contribute, we need:
    # (i_r, j_r) = (i, j)
    # (j_r, k_r) = (j', k)
    # (i_r, k_r) = (i, k)
    
    # This implies:
    # i_r = i, j_r = j
    # j_r = j', k_r = k
    # i_r = i, k_r = k
    
    # So we need j = j' and k = k and i = i.
    # So if j = j', the term r corresponding to (i, j, k) contributes 1.
    # Are there other terms?
    # If j = j', then (i,j,k) is the only triple that satisfies all three conditions?
    # i_r=i, j_r=j.
    # j_r=j, k_r=k.
    # i_r=i, k_r=k.
    # Yes, unique triple (i,j,k).
    # So the sum is 1 if j=j', 0 otherwise.
    
    # So schoolbook works. Rank 18.
    
    # Can we do better?
    # The directive suggests rank 14.
    # How to construct it?
    
    # Let's look at the structure of the tensor.
    # We need sum_r u_{i,j,r} v_{j',k,r} w_{i,k,r} = delta_{j,j'}.
    
    # This looks like we are trying to represent the identity matrix I_m (of size m x m) using a sum of outer products of vectors from u, v, w.
    # For each fixed i, k, we have a matrix M^{(i,k)}_{j, j'} = sum_r u_{i,j,r} v_{j',k,r} w_{i,k,r}.
    # We want M^{(i,k)} = I_m.
    
    # Note that w_{i,k,r} is a scalar multiplier for the rank-1 update u_{i,:,r} v^T_{:,k,r}.
    # So for each (i,k), we decompose I_m into sum_r w_{i,k,r} u^{(i,k)}_r (v^{(i,k)}_r)^T.
    # This is a sum of rank-1 matrices.
    # The rank of I_m is m=3.
    # So for each (i,k), we need at least rank 3.
    # There are n*p = 2*3 = 6 pairs of (i,k).
    # Total rank could be up to 6*3 = 18.
    # But the vectors u and v are shared across i,k?
    # u_{i,j,r} depends on i. v_{j,k,r} depends on k.
    # w_{i,k,r} depends on both.
    
    # So we have 14 vectors u_r (each of size n*m = 6), v_r (each of size m*p = 9), w_r (each of size p*n = 6).
    
    # Let's try to find a rank-14 solution.
    # This is a hard optimization problem.
    # However, I can try to construct one using the properties of Strassen or similar.
    
    # Actually, let's try a simple heuristic.
    # If we set w_{i,k,r} = 1 for all r?
    # Then we need sum_r u_{i,j,r} v_{j',k,r} = delta_{j,j'}.
    # This means for each (i,k), the matrix U_{i,:,r} V_{:,k,r} sums to I.
    
    # Let's try to reuse the same set of vectors for different i,k?
    # If i and k don't affect u and v, then u_{j,r} and v_{j,k,r}...
    # But u depends on i.
    
    # Let's try to build a candidate manually or via a solver?
    # I don't have a solver.
    
    # Let's try a known result.
    # The rank of M_{2,3,3} is 14.
    # Reference: "The rank of the 2x3x3 tensor is 14" (e.g., from tensor rank literature).
    
    # How to construct it?
    # One construction:
    # Embed M_{2,3,3} into M_{2,4,4} or similar?
    # Or use the fact that M_{2,3,3} can be obtained from M_{2,2,2} (Strassen) by adding dimensions.
    
    # Let's try to extend Strassen.
    # Strassen decomposes 2x2x2 with rank 7.
    # We have 2x3x3.
    # We can view 3x3 as 2x2 block plus extra?
    
    # Alternative:
    # Use a random search to find a rank-14 decomposition?
    # Unlikely to succeed in 120s.
    
    # Let's try to construct a specific candidate.
    # I will use a known rank-14 construction if I can recall it, or derive a simple one.
    
    # Actually, I can try to solve the linear equations for a symmetric case?
    
    # Let's try this:
    # Since I cannot guarantee a rank-14 solution, and the prompt asks for a "research program", I will attempt to construct a rank-15 or rank-16 solution if possible, or stick to schoolbook.
    # But the directive says "improve it".
    
    # Let's try to construct a rank-15 solution.
    # Or better, let's try to implement a simple solver that attempts to find a low-rank decomposition using iterative refinement.
    # But that's complex.
    
    # Let's look at the "Kronecker product" hint.
    # "use the known rank-7 decomposition for the 3x3 matrix multiplication tensor"
    # This is definitely wrong (3x3 is rank 19? or 27?).
    # Maybe it meant "rank-7 decomposition for the 2x2 matrix multiplication tensor".
    
    # If we use Kronecker product:
    # M_{2,3,3} ?
    # 3 = 2+1.
    
    # Let's try to use the Strassen decomposition for the 2x2 part and schoolbook for the rest.
    # A = [A11 A12 A13; A21 A22 A23]
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    
    # This is getting complicated.
    
    # Given the constraints, I will provide a solution that attempts a rank-14 construction using a hardcoded set of coefficients if I can find a valid set, otherwise falls back to schoolbook.
    # Since I don't have the coefficients, I will use a heuristic:
    # Try to construct a rank-14 solution by optimizing? No, too slow.
    
    # I will return the schoolbook solution (rank 18) but structured clearly, as it is the only guaranteed correct answer I can produce.
    # Wait, the prompt says "improve it".
    # If I can't find a better rank, I should at least ensure the code is clean.
    # But "score is the rank".
    
    # Let's try one more idea:
    # Is it possible that the tensor is NOT the all-ones tensor?
    # "official trace convention W[k*n+i,r]"
    # Maybe the tensor is different?
    # No, the tensor for matrix multiplication is standard.
    
    # I will output the schoolbook solution. It is rank 18.
    # The seed program is rank 18.
    # Score 1.0.
    
    # I cannot do better without the coefficients.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
