"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x2 matrix by a 2x11 matrix.
    We use a structured approach that exploits the small size of the first dimension.
    
    The multiplication is: C = A * B where A is 2x2, B is 2x11, C is 2x11.
    
    We can view this as two independent 2x11 multiplications (one for each row of A).
    For each row i of A, C[i, :] = A[i, 0] * B[0, :] + A[i, 1] * B[1, :]
    
    This is a linear combination of the two rows of B with coefficients from row i of A.
    
    For a fixed i, C[i, k] = A[i, 0]*B[0, k] + A[i, 1]*B[1, k]
    
    This is a rank-2 bilinear form in the variables A[i,0], A[i,1], B[0,k], B[1,k].
    
    Actually, let's think of it as a tensor T with indices (i, j, k) where:
    - i: output row (0,1) - corresponds to which row of C
    - j: input row from A (0,1) - which column of A / which row of B is being multiplied
    - k: column index (0..10)
    
    T[i,j,k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    
    Wait, the standard tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] where:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for all i,j,k (in the sense that each triple contributes).
    
    For n=2, m=2, p=11:
    T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,10}
    
    The schoolbook rank is n*m*p = 2*2*11 = 44.
    
    Can we do better? The tensor is T[i,j,k] = 1 for all i,j,k.
    This is actually a sum of rank-1 tensors: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    
    But we can factor this differently. Note that T[i,j,k] = 1 = 1*1*1.
    
    We can write T = sum_{i,j} (e_i ⊗ e_j) ⊗ (sum_k e_k) ... no, that's not rank-1.
    
    A rank-1 tensor is a ⊗ b ⊗ c where T[i,j,k] = a[i]*b[j]*c[k].
    
    For our T, T[i,j,k] = 1 for all i,j,k. This is exactly 1⊗1⊗1 where 1 is the all-ones vector.
    So T = 1_n ⊗ 1_m ⊗ 1_p is a rank-1 tensor!
    
    Wait, that's not right. The matrix multiplication tensor is not the all-ones tensor.
    
    Let me reconsider. The tensor T for matrix multiplication has T[i,j,k] = 1 if and only if
    the product A[i,j] * B[j,k] contributes to C[i,k]. In the standard formulation, every triple
    (i,j,k) contributes, so T[i,j,k] = 1 for all valid i,j,k.
    
    But that would make it rank 1, which is clearly wrong. The issue is that the tensor is not
    simply T[i,j,k] = 1; rather, the multiplication is C[i,k] = sum_j A[i,j] B[j,k], and the
    tensor encodes this bilinear map.
    
    The correct tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]. This is because
    each term A[i,j]*B[j,k] appears in the sum for C[i,k]. So the tensor is indeed the all-ones
    tensor of shape (n, m, p).
    
    But the all-ones tensor of shape (2,2,11) is 1_2 ⊗ 1_2 ⊗ 1_11, which is rank 1!
    
    That can't be right either, because we know that matrix multiplication requires more than
    rank 1.
    
    I think I'm confusing the tensor with the multiplication pattern. Let me look at this more carefully.
    
    The matrix multiplication tensor T_{n,m,p} is defined such that for matrices A (n×m) and B (m×p),
    the product C = AB satisfies C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1.
    
    So T is indeed the all-ones tensor. But the rank of the all-ones tensor 1_n ⊗ 1_m ⊗ 1_p is 1.
    
    This suggests that the tensor decomposition of the all-ones tensor has rank 1, which would mean
    we can compute the entire matrix product with just 1 multiplication. That's clearly not possible.
    
    I must be misunderstanding the tensor decomposition in this context. Let me re-read the task.
    
    The task says: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    So U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] is the tensor entry.
    
    For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since every product term appears).
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is a different problem! We're not decomposing the all-ones tensor as a sum of rank-1 tensors
    in the usual sense. Instead, we're looking for a decomposition where the product of three vectors
    (indexed by the combined indices) sums to 1.
    
    Actually, I think the tensor T[i,j,k] for matrix multiplication is indeed 1 for all i,j,k, and
    we're looking for a rank-r decomposition T = sum_c u_c ⊗ v_c ⊗ w_c where u_c is a vector of
    length n*m, v_c is a vector of length m*p, w_c is a vector of length p*n.
    
    But that's not the standard tensor rank. In the standard formulation, the tensor for matrix
    multiplication has entries T[i,j,k] and we decompose it as sum_c a_c[i] * b_c[j] * d_c[k]
    where a_c has length n, b_c has length m, d_c has length p.
    
    The task uses a different indexing: U[i*m+j, c] combines i and j into a single index, etc.
    
    So the tensor is actually of shape (n*m, m*p, p*n) and we're decomposing it. But that doesn't
    match the standard matrix multiplication tensor.
    
    Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    I think the idea is that the tensor T has indices (i,j,k) and we're flattening them:
    - First index: i*m + j (range 0 to n*m-1)
    - Second index: j*p + k (range 0 to m*p-1)  
    - Third index: k*n + i (range 0 to p*n-1)
    
    And T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent (i.e., the j in the first and second
    match, and the k in the second and third match, and the i in the first and third match), and 0 otherwise.
    
    Wait, that's still not quite right. Let me think about this differently.
    
    The standard matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    The rank of this tensor is the minimum r such that T[i,j,k] = sum_{c=1}^r a_c[i] b_c[j] d_c[k].
    
    In the task's formulation, we have U of shape (n*m, r), V of shape (m*p, r), W of shape (p*n, r).
    The decomposition is: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k].
    
    But T[i,j,k] = 1 for all i,j,k in the matrix multiplication tensor.
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is a constraint on the flattened tensors. It's not the same as the standard tensor rank.
    
    Actually, I think I've been overthinking this. Let me just try a simple approach: use the schoolbook
    decomposition and see if we can improve it.
    
    For now, let me just return the schoolbook decomposition with rank 44.
    """
    n, m, p = DIMENSIONS
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
