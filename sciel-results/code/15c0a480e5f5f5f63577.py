"""Matrix multiplication 4x4x4: attempt rank-28 via 2x2x2 rank-7 over dual numbers."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We attempt a rank-28 decomposition.
    # The strategy: decompose 4x4x4 into four 2x2x2 blocks using the Klein four-group symmetry.
    # For 2x2x2, the minimum rank is 7 (Strassen).
    # We use the dual numbers algebra A = {a + b*eps | a,b in F, eps^2=0}.
    # The tensor T_4 can be viewed as a tensor over A of size 2x2x2 with rank 7 * (dim A) = 7*2 = 14?
    # No, that's not how it works.
    
    # Let's look at the structure.
    # The 4x4x4 multiplication tensor T has entries T[i, j, k] for i,j,k in {0,1,2,3}.
    # We want to find U, V, W of shape (16, r), (16, r), (16, r) such that
    # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    # Note: The problem statement says W[k*n+i, r]. This is the standard flattening.
    # So the tensor is T[i, j, k] where i is row of A, j is col of A (row of B), k is col of B.
    # Wait, standard matrix mult C = A * B. C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has indices (i, j, k) corresponding to A[i, j], B[j, k], C[i, k].
    # The interface specifies:
    # U: (n*m, rank) -> indices (i, j)
    # V: (m*p, rank) -> indices (j, k)
    # W: (p*n, rank) -> indices (k, i)  <-- Note: p*n means k in [0,p), i in [0,n).
    # So W is indexed by (k, i).
    # The decomposition is: C[i, k] = sum_j A[i, j] B[j, k] = sum_c (sum_j U[i*m+j, c] V[j*p+k, c]) ... wait.
    # Let's check the standard decomposition.
    # A[i, j] is represented by U. B[j, k] by V. C[i, k] by W.
    # Usually: T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c].
    # Here U has indices (i, j). So U[i*m+j, c].
    # V has indices (j, k). So V[j*p+k, c].
    # W has indices (k, i). So W[k*n+i, c].
    # The product for a fixed c is: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This doesn't look like a standard outer product of three vectors unless we sum over j.
    # Ah, the tensor T is defined by T[i, j, k] = delta_{i, i} delta_{j, j} delta_{k, k}? No.
    # The matrix multiplication tensor T_{ijk} is 1 if the indices match the contraction, 0 otherwise?
    # No, the tensor IS the multiplication map.
    # T(A, B) = C.
    # T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{ijk} is 1 if we are looking at C_{ik} and the term is A_{ij}B_{jk}.
    # So for a fixed (i, k), T_{ijk} is 1 only for the specific j? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has indices (i, j, k).
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # This is always true for any i, j, k.
    # So T_{ijk} = 1 for all i, j, k?
    # No, that would mean C_{ik} = sum_{i', j', k'} T_{i'j'k'} A_{i'j'} B_{j'k'} ... this is getting confusing.
    
    # Let's stick to the seed program's definition.
    # Seed: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # This implies that for a specific triple (i, j, k), there is a column c where U, V, W are 1.
    # And for that column, the product U*V*W contributes to the "tensor entry".
    # The evaluator likely checks if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == T_{ijk}.
    # What is T_{ijk}?
    # In matrix multiplication, the "tensor" is the map (A, B) -> C.
    # The entry C_{ik} is sum_j A_{ij} B_{jk}.
    # So the tensor T is a 3-tensor where T_{ijk} = 1 if the indices correspond to a valid term in the sum.
    # But every (i, j, k) is a valid term in SOME sum?
    # Actually, the tensor T is defined such that <T, A \otimes B> = C.
    # <T, A \otimes B> = sum_{i, j, k} T_{ijk} A_{ij} B_{jk}.
    # We want this to equal sum_{i, k} C_{ik} E_{ik} ... but C is a matrix.
    # So we need sum_{i, j, k} T_{ijk} A_{ij} B_{jk} = sum_{i, k} (sum_j A_{ij} B_{jk}) * 1.
    # This implies T_{ijk} must be such that when contracted with A and B, it gives C.
    # This means T_{ijk} is 1 if the indices (i, j, k) form a "path" i->j->k.
    # But in matrix multiplication, every i, j, k is a path.
    # So T_{ijk} = 1 for all i, j, k?
    # If T_{ijk} = 1, then sum_c U_{ij, c} V_{jk, c} W_{ki, c} must be 1 for all i, j, k.
    # Wait, W is indexed by (k, i). So W_{ki, c}.
    # The seed program sets U_{ij, c}=1, V_{jk, c}=1, W_{ki, c}=1 for a specific c corresponding to (i, j, k).
    # So for that c, the product is 1. For other c, it's 0.
    # So the sum is 1.
    # This confirms T_{ijk} = 1 for all i, j, k.
    # Wait, is that right?
    # If T_{ijk} = 1, then the tensor is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1? No.
    # The all-ones tensor J_{ijk} = 1.
    # J = 1 \otimes 1 \otimes 1. Rank 1.
    # But matrix multiplication is NOT the all-ones tensor.
    # Let's re-read the seed.
    # U[i*m+j, c] = 1.
    # V[j*p+k, c] = 1.
    # W[k*n+i, c] = 1.
    # This creates a tensor S_{ijk} = sum_c U_{ij, c} V_{jk, c} W_{ki, c}.
    # For the seed, S_{ijk} = 1 for all i, j, k.
    # So the seed program computes the rank of the all-ones tensor?
    # No, that can't be right. Matrix multiplication is not the all-ones tensor.
    
    # Let's look at the indices again.
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    # The product is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # This sum is over c.
    # For a fixed (i, j, k), this sum gives a value.
    # The evaluator checks if this value equals the "true" tensor entry.
    # What is the true tensor entry for matrix multiplication?
    # The multiplication map is mu: M_n x M_m -> M_p (if n=m=p).
    # mu(A, B) = A B.
    # The tensor T is defined by T(A, B) = mu(A, B).
    # In coordinates: (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has components T_{(i, k), (i', j), (j', k')}? No.
    # The tensor is a trilinear form T(A, B, C) = <C, AB> = sum_{i, k} C_{ik} (AB)_{ik} = sum_{i, j, k} C_{ik} A_{ij} B_{jk}.
    # So T_{i, j, k; i, k} ... this is getting messy with indices.
    # Let's use the standard definition: T is a tensor in U^* \otimes V^* \otimes W^*.
    # Here U = space of A (dim 16), V = space of B (dim 16), W = space of C (dim 16).
    # T = sum_{i, j, k} e_{ij} \otimes e_{jk} \otimes e_{ik} ?
    # Let's check: <T, A \otimes B \otimes C^*> = sum_{i, j, k} A_{ij} B_{jk} C_{ki}?
    # No, <T, A \otimes B> should give C.
    # So T is in U^* \otimes V^* \otimes W.
    # T = sum_{i, j, k} e^*_{ij} \otimes e^*_{jk} \otimes e_{ik}.
    # Then <T, A \otimes B> = sum_{i, j, k} A_{ij} B_{jk} e_{ik} = (AB).
    # So the components of T are:
    # T_{(i, j), (j, k), (i, k)} = 1.
    # All other components are 0.
    # Specifically, T_{u, v, w} = 1 if u=(i, j), v=(j, k), w=(i, k) for some i, j, k.
    # Note that the middle index of u (j) must match the first index of v (j).
    # And the first index of u (i) must match the first index of w (i).
    # And the second index of v (k) must match the second index of w (k).
    
    # So the condition is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if the indices are consistent?
    # No, the sum is over c. The result is a scalar for each (i, j, k).
    # Wait, the sum is over c, but i, j, k are fixed.
    # So for each triple (i, j, k), we compute S_{ijk} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # The evaluator checks if S_{ijk} == T_{ijk}.
    # What is T_{ijk}?
    # From T = sum_{i, j, k} e^*_{ij} \otimes e^*_{jk} \otimes e_{ik},
    # The component T_{(i, j), (j, k), (i, k)} is 1.
    # But the indices of S are (i, j, k).
    # The indices of U are (i, j).
    # The indices of V are (j, k).
    # The indices of W are (k, i).
    # So S_{ijk} corresponds to the term where A has index (i, j), B has index (j, k), and C has index (i, k) (flattened as k*n+i).
    # So S_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # In matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the coefficient of A_{ij} B_{jk} in C_{ik} is 1.
    # So T_{ijk} = 1 for ALL i, j, k.
    # This means the tensor is indeed the all-ones tensor!
    # Wait, if T_{ijk} = 1 for all i, j, k, then the rank is 1.
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then S_{ijk} = 1*1*1 = 1.
    # So the rank is 1?
    # That can't be right. Matrix multiplication has rank at least n^2.
    
    # I must be misunderstanding the tensor definition.
    # Let's look at the seed again.
    # The seed uses rank 64.
    # If the rank were 1, the seed would be very inefficient.
    # Let's re-read the task description.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 4 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # Maybe the tensor is not T_{ijk} = 1.
    # Let's consider the standard tensor decomposition for matrix multiplication.
    # T = sum_c u_c \otimes v_c \otimes w_c.
    # u_c is a vector in R^{n*m}.
    # v_c is a vector in R^{m*p}.
    # w_c is a vector in R^{p*n}.
    # The condition is:
    # For all A, B: AB = sum_c (u_c^T A) (v_c^T B) w_c.
    # Here u_c^T A is a scalar (dot product of u_c and A).
    # v_c^T B is a scalar.
    # w_c is a matrix (flattened).
    # So AB = sum_c (sum_{i, j} u_{c, ij} A_{ij}) (sum_{j, k} v_{c, jk} B_{jk}) w_c.
    # The coefficient of A_{i0, j0} B_{j1, k0} in the RHS is:
    # sum_c u_{c, i0, j0} v_{c, j1, k0} w_{c, i0, k0} (if j0 == j1? No, the sums are independent).
    # Wait, (u_c^T A) = sum_{i, j} u_{c, ij} A_{ij}.
    # (v_c^T B) = sum_{j, k} v_{c, jk} B_{jk}.
    # So the term is (sum_{i, j} u_{c, ij} A_{ij}) (sum_{j', k} v_{c, j'k} B_{j'k}) w_{c, ik}.
    # The coefficient of A_{i0, j0} B_{j1, k0} is:
    # sum_c u_{c, i0, j0} v_{c, j1, k0} w_{c, i0, k0} * [j0 == j1? No, the indices j and j' are summed independently].
    # So the coefficient is sum_c u_{c, i0, j0} v_{c, j1, k0} w_{c, i0, k0}.
    # This must equal the coefficient of A_{i0, j0} B_{j1, k0} in AB.
    # In AB, the term A_{i0, j0} B_{j1, k0} appears only if j0 == j1.
    # If j0 == j1, the coefficient is 1 (if i0, k0 are valid).
    # If j0 != j1, the coefficient is 0.
    # So we require:
    # sum_c U[i0*m+j0, c] V[j1*p+k0, c] W[k0*n+i0, c] = 1 if j0 == j1, else 0.
    # This is the correct condition!
    # The seed program sets U, V, W to 1 for a specific (i, j, k).
    # For that c, U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # If j0=j, j1=j, k0=k, i0=i, then the sum is 1.
    # If j0!=j1, then for any c, either U or V is 0?
    # In the seed, for a given c, U is 1 only for (i, j), V is 1 only for (j, k), W is 1 only for (k, i).
    # So if we pick c corresponding to (i, j, k), then U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # If we evaluate the sum for (i0, j0, j1, k0) with j0=j1=j, k0=k, i0=i, the sum is 1.
    # If j0 != j1, say j0=j, j1=j', then for the c corresponding to (i, j, k), V[j'*p+k, c] is 0 (unless j'=j).
    # So the sum is 0.
    # So the seed program correctly implements the condition.
    
    # So the problem is to find U, V, W of rank r such that:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = delta_{j, j'}.
    # (Assuming i, k are fixed in the output, but the sum is over c).
    # Actually, the condition is for all i, j, j', k:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = 1 if j == j', else 0.
    
    # This is equivalent to:
    # Let A be the matrix with entries A_{ij} = U[i*m+j, c].
    # Let B be the matrix with entries B_{jk} = V[j*p+k, c].
    # Let C be the matrix with entries C_{ki} = W[k*n+i, c].
    # Then the condition is:
    # sum_c (A_c)_{ij} (B_c)_{j'k} (C_c)_{ki} = delta_{j, j'}.
    # This can be rewritten as:
    # sum_c (A_c \otimes B_c \otimes C_c)_{ij, j'k, ki} = delta_{j, j'}.
    
    # This is the tensor decomposition of the identity tensor?
    # No, it's the tensor for matrix multiplication.
    
    # Now, how to minimize the rank?
    # The rank of 4x4x4 matrix multiplication is known to be between 49 and 64.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 4x4x4, we can use Strassen recursively.
    # 4x4 can be split into four 2x2 blocks.
    # This requires 7 * 7 = 49 multiplications.
    # So rank 49 is achievable.
    # Can we do better?
    # The lower bound is 49. The upper bound is 64.
    # It is an open problem whether the rank is 49 or higher.
    # However, for the purpose of this task, we should aim for the best known upper bound, which is 49.
    # The seed uses 64.
    # So we can improve to 49.
    
    # Let's implement Strassen's algorithm for 4x4x4.
    # We split A, B into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A B = [[C11, C12], [C21, C22]]
    # Strassen computes 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_i is a product of two 2x2 matrices.
    # We can use Strassen again for each 2x2 product, using 7 multiplications.
    # So total multiplications = 7 * 7 = 49.
    
    # We need to construct U, V, W for this decomposition.
    # This is a bit complex.
    # Let's define the basis for the 2x2 matrices.
    # The 2x2 matrices are represented as vectors in R^4.
    # Let e1, e2, e3, e4 be the basis vectors for [[1,0],[0,0]], [[0,1],[0,0]], [[0,0],[1,0]], [[0,0],[0,1]].
    
    # For the top-level Strassen, we have 7 "super-multiplications".
    # Each super-multiplication is a product of two linear combinations of 2x2 blocks.
    # Let's denote the 2x2 blocks of A as A11, A12, A21, A22.
    # And B as B11, B12, B21, B22.
    
    # The 7 products are:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    
    # Each Pi is a 2x2 matrix.
    # We can write Pi = sum_{c in 1..7} (u_{i,c}^T A) (v_{i,c}^T B) w_{i,c}.
    # Where u_{i,c}, v_{i,c} are vectors in R^4 (representing 2x2 matrices), and w_{i,c} is a 2x2 matrix.
    
    # To combine these, we need to express the full 4x4 multiplication in terms of these.
    # This requires careful bookkeeping of indices.
    
    # Let's try a different approach.
    # We can use the known rank-49 decomposition for 4x4x4.
    # I will hardcode the U, V, W matrices for the rank-49 decomposition.
    # This is a known result.
    
    # However, generating the matrices programmatically is safer.
    # Let's implement the recursive Strassen.
    
    # Step 1: Define the rank-7 decomposition for 2x2x2.
    # U2, V2, W2 are 4x7 matrices.
    # U2[i, c], V2[j, c], W2[k, c] such that sum_c U2[i, c] V2[j, c] W2[k, c] = delta_{i, j} (for 2x2).
    # Wait, the condition for 2x2 is sum_c U2[i, c] V2[j, c] W2[k, c] = 1 if i==j? No.
    # The condition is sum_c U2[i0, j0, c] V2[j1, k0, c] W2[k0, i0, c] = delta_{j0, j1}.
    # For 2x2, i, j, k are in {0, 1}.
    
    # Let's find the rank-7 decomposition for 2x2.
    # This is the standard Strassen decomposition.
    # I will use the coefficients from Strassen's algorithm.
    
    # For 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We can extract U, V, W from this.
    # Each M_i is a product of two linear forms.
    # M1 = (A11 + A22) * (B11 + B22)
    # This corresponds to u1 = [1, 0, 0, 1] (A11 + A22), v1 = [1, 0, 0, 1] (B11 + B22), w1 = [[1, 1], [1, 1]]? No.
    # M1 contributes to C11, C12, C21, C22.
    # C11 gets M1, so w1 has 1 at (0, 0).
    # C12 gets 0 from M1.
    # C21 gets 0 from M1.
    # C22 gets M1, so w1 has 1 at (1, 1).
    # So w1 = [[1, 0], [0, 1]]? No, M1 is a 2x2 matrix.
    # M1 = (A11 + A22) * (B11 + B22).
    # If A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # M1 = (a+d)(e+h) = ae + ah + de + dh.
    # C11 = ae + bg.
    # So M1 contributes ae + ah + de + dh to C11? No.
    # C11 = M1 + M4 - M5 + M7.
    # This is a linear combination of the M_i.
    # Each M_i is a scalar (product of two linear forms).
    # So C11 is a 2x2 matrix whose entries are linear combinations of the M_i.
    # This means w_i are 2x2 matrices.
    
    # Let's construct U, V, W for 2x2.
    # U is 4x7, V is 4x7, W is 4x7.
    # Indexing: U[i, c] where i is (row, col) of A.
    # V[j, c] where j is (row, col) of B.
    # W[k, c] where k is (row, col) of C.
    
    # M1: u1 = [1, 0, 0, 1] (A11, A12, A21, A22 -> 1, 0, 0, 1)
    # v1 = [1, 0, 0, 1] (B11, B12, B21, B22 -> 1, 0, 0, 1)
    # w1: M1 contributes to C11 and C22?
    # C11 = M1 + M4 - M5 + M7.
    # C22 = M1 - M2 + M3 + M6.
    # So w1 has 1 at (0, 0) and (1, 1).
    # w1 = [1, 0, 0, 1] (C11, C12, C21, C22).
    
    # M2: u2 = [0, 0, 1, 1] (A21 + A22)
    # v2 = [1, 0, 0, 0] (B11)
    # w2: C21 = M2 + M4.
    # So w2 has 1 at (1, 0) (C21).
    # w2 = [0, 0, 1, 0].
    
    # M3: u3 = [1, 0, 0, 0] (A11)
    # v3 = [0, 1, 0, -1] (B12 - B22)
    # w3: C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
    # So w3 has 1 at (0, 1) (C12) and (1, 1) (C22).
    # w3 = [0, 1, 0, 1].
    
    # M4: u4 = [0, 0, 0, 1] (A22)
    # v4 = [0, 0, 1, -1] (B21 - B11)
    # w4: C11 = M1 + M4 - M5 + M7. C21 = M2 + M4.
    # So w4 has 1 at (0, 0) (C11) and (1, 0) (C21).
    # w4 = [1, 0, 1, 0].
    
    # M5: u5 = [1, 1, 0, 0] (A11 + A12)
    # v5 = [0, 0, 0, 1] (B22)
    # w5: C11 = M1 + M4 - M5 + M7. C12 = M3 + M5.
    # So w5 has -1 at (0, 0) (C11) and 1 at (0, 1) (C12).
    # w5 = [-1, 1, 0, 0].
    
    # M6: u6 = [-1, 0, 1, 0] (A21 - A11)
    # v6 = [1, 1, 0, 0] (B11 + B12)
    # w6: C22 = M1 - M2 + M3 + M6.
    # So w6 has 1 at (1, 1) (C22).
    # w6 = [0, 0, 0, 1].
    
    # M7: u7 = [0, 1, 0, -1] (A12 - A22)
    # v7 = [0, 0, 1, 1] (B21 + B22)
    # w7: C11 = M1 + M4 - M5 + M7.
    # So w7 has 1 at (0, 0) (C11).
    # w7 = [1, 0, 0, 0].
    
    # Let's verify this.
    # U = [[1, 0, 1, 0, 1, -1, 0],
    #      [0, 0, 0, 0, 1, 0, 1],
    #      [0, 1, 0, 0, 0, 1, 0],
    #      [1, 1, 0, 1, 0, 0, -1]]
    # V = [[1, 1, 0, 0, 0, 1, 0],
    #      [0, 0, 1, 0, 0, 1, 0],
    #      [0, 0, 0, 1, 0, 0, 1],
    #      [1, 0, -1, -1, 1, 0, 1]]
    # W = [[1, 0, 0, 1, -1, 0, 1],
    #      [0, 0, 1, 0, 1, 0, 0],
    #      [0, 1, 0, 1, 0, 0, 0],
    #      [1, 0, 1, 0, 0, 1, 0]]
    
    # Now, for 4x4, we use this 2x2 decomposition recursively.
    # We split 4x4 into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2.
    # B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # C = [[C11, C12], [C21, C22]] where Cij are 2x2.
    
    # We use the 7 Strassen products for the 2x2 block multiplication.
    # P1 = (A11 + A22) * (B11 + B22)
    # ...
    # P7 = (A12 - A22) * (B21 + B22)
    
    # Each Pi is a 2x2 matrix.
    # We can compute Pi using the 2x2 Strassen decomposition.
    # So for each Pi, we need 7 multiplications of 2x2 matrices.
    # Total 49 multiplications.
    
    # We need to construct U, V, W for the 4x4 case.
    # U is 16x49.
    # V is 16x49.
    # W is 16x49.
    
    # The indices for U are (i, j) where i, j in {0, 1, 2, 3}.
    # We can map (i, j) to (block_row, block_col, local_row, local_col).
    # block_row = i // 2, block_col = j // 2.
    # local_row = i % 2, local_col = j % 2.
    
    # Similarly for V and W.
    
    # For each of the 7 Strassen products Pk (k=1..7), we have 7 sub-products.
    # Let's index the columns of U, V, W by (k, c) where k is the Strassen product and c is the sub-product.
    # Total columns = 7 * 7 = 49.
    
    # For a given column (k, c):
    # U[i, j, c] = U2[local_i, local_j, c] * coefficient from Pk's A part.
    # V[i, j, c] = V2[local_i, local_j, c] * coefficient from Pk's B part.
    # W[i, j, c] = W2[local_i, local_j, c] * coefficient from Pk's C part.
    
    # This is getting complicated. Let's use a simpler approach.
    # We can just use the rank-49 decomposition directly.
    # I will generate the U, V, W matrices using the recursive Strassen structure.
    
    # Let's define the 2x2 Strassen matrices U2, V2, W2 as above.
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ])
    
    # Now, for 4x4, we have 7 block products.
    # Let's define the coefficients for the block products.
    # P1: A11+A22, B11+B22
    # P2: A21+A22, B11
    # P3: A11, B12-B22
    # P4: A22, B21-B11
    # P5: A11+A12, B22
    # P6: A21-A11, B11+B12
    # P7: A12-A22, B21+B22
    
    # We can represent the block coefficients as 2x2 matrices of coefficients.
    # For A, the coefficients for Pk are a 2x2 matrix CAk.
    # For B, the coefficients for Pk are a 2x2 matrix CBk.
    # For C, the coefficients for Pk are a 2x2 matrix CCk.
    
    # P1: CA1 = [[1, 0], [0, 1]], CB1 = [[1, 0], [0, 1]], CC1 = [[1, 0], [0, 1]]
    # P2: CA2 = [[0, 0], [1, 1]], CB2 = [[1, 0], [0, 0]], CC2 = [[0, 0], [1, 0]]
    # P3: CA3 = [[1, 0], [0, 0]], CB3 = [[0, 1], [0, -1]], CC3 = [[0, 1], [0, 1]]
    # P4: CA4 = [[0, 0], [0, 1]], CB4 = [[0, 0], [1, -1]], CC4 = [[1, 0], [1, 0]]
    # P5: CA5 = [[1, 1], [0, 0]], CB5 = [[0, 0], [0, 1]], CC5 = [[-1, 1], [0, 0]]
    # P6: CA6 = [[-1, 0], [1, 0]], CB6 = [[1, 1], [0, 0]], CC6 = [[0, 0], [0, 1]]
    # P7: CA7 = [[0, 1], [0, -1]], CB7 = [[0, 0], [1, 1]], CC7 = [[1, 0], [0, 0]]
    
    # Now, we can construct U, V, W.
    # U[i, j, (k, c)] = U2[local_i, local_j, c] * CAk[block_i, block_j]
    # V[i, j, (k, c)] = V2[local_i, local_j, c] * CBk[block_i, block_j]
    # W[i, j, (k, c)] = W2[local_i, local_j, c] * CCk[block_i, block_j]
    
    # Let's implement this.
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    CA = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    CB = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, -1]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    CC = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for k in range(7):
        for c in range(7):
            col = k * 7 + c
            for i in range(4):
                for j in range(4):
                    block_i = i // 2
                    block_j = j // 2
                    local_i = i % 2
                    local_j = j % 2
                    U[i * 4 + j, col] = U2[local_i * 2 + local_j, c] * CA[k][block_i][block_j]
                    V[i * 4 + j, col] = V2[local_i * 2 + local_j, c] * CB[k][block_i][block_j]
                    W[i * 4 + j, col] = W2[local_i * 2 + local_j, c] * CC[k][block_i][block_j]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
