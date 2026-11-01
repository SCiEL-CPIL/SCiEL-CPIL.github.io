"""Tensor decomposition for 2x3 times 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for multiplying an n x m matrix by an m x p matrix is the direct sum
    # of p identical tensors for multiplying an n x m matrix by an m x 1 vector.
    # Let r = min_rank(n, m, 1). Then the total rank is p * r.
    # We need to find a rank-r decomposition for the n x m x 1 case (where p=1).
    # For n=2, m=3, p=1, the tensor T has entries T[i*3+j, j, i] = 1 (using 0-indexed i in [0,1], j in [0,2]).
    # Wait, the index conventions are:
    # U has shape (n*m, rank)
    # V has shape (m*p, rank) -> here m*1 = 3
    # W has shape (p*n, rank) -> here 1*n = 2
    # The equation is sum_k U[alpha, k] V[beta, k] W[gamma, k] = T[alpha, beta, gamma]
    # For the sub-problem (p=1), beta is just j, gamma is just i.
    # alpha = i*m + j.
    # T[alpha, beta, gamma] = 1 if alpha == i*m + beta AND gamma == i, else 0.
    # So we need: sum_k U[i*m+j, k] V[j, k] W[i, k] = 1 for all i, j.
    # And implicitly 0 for all other combinations?
    # The tensor is defined on the full space of dimensions (n*m) x (m*p) x (p*n).
    # The condition is that the bilinear map defined by U, V, W computes the matrix product.
    # This is equivalent to:
    # For any matrices A (n x m) and B (m x p), C = A B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The decomposition computes C[i, k] = sum_r U[i*m+?, r] * V[?*p+k, r] * W[k*n+i, r]
    # Actually, the standard Kronecker product structure implies that the tensor for (n, m, p)
    # is isomorphic to the direct sum of p copies of the tensor for (n, m, 1).
    # Specifically, fix the output column k. The sub-tensor involving input indices (i, j) and output index k
    # is determined by A[i, j] and B[j, k].
    # The decomposition rank is additive over the output columns if the decomposition respects the block structure.
    # The "schoolbook" algorithm has rank n*m*p.
    # Can we do better?
    # For (2, 3, 1), the tensor is:
    # T[i*3+j, j, i] = 1.
    # This is a tensor in R^6 x R^3 x R^2.
    # This is equivalent to the tensor for matrix multiplication of size (2, 3, 1).
    # This is also the transpose of the tensor for (1, 3, 2).
    # Let's try to find a low-rank decomposition for (2, 3, 1).
    # If we can find rank r, then for (2, 3, 9), the rank will be 9*r.
    #
    # Let's analyze the (2, 3, 1) tensor.
    # Indices: alpha in 0..5, beta in 0..2, gamma in 0..1.
    # Non-zero entries:
    # gamma=0 (i=0):
    #   j=0: alpha=0, beta=0 -> T[0, 0, 0] = 1
    #   j=1: alpha=1, beta=1 -> T[1, 1, 0] = 1
    #   j=2: alpha=2, beta=2 -> T[2, 2, 0] = 1
    # gamma=1 (i=1):
    #   j=0: alpha=3, beta=0 -> T[3, 0, 1] = 1
    #   j=1: alpha=4, beta=1 -> T[4, 1, 1] = 1
    #   j=2: alpha=5, beta=2 -> T[5, 2, 1] = 1
    #
    # We want to decompose this tensor into a sum of r rank-1 tensors.
    # A rank-1 tensor is u * v * w.
    # Let's check if rank 3 is possible.
    # If rank is 3, we have 3 terms.
    # Consider the slice gamma=0. It is a 6x3 matrix with 1s at (0,0), (1,1), (2,2).
    # This matrix has rank 3.
    # The rank of a slice of a tensor is a lower bound on the rank of the tensor?
    # No, the rank of a tensor is >= max rank of any slice.
    # So rank >= 3.
    # Can we achieve rank 3?
    # If rank is 3, then U, V, W are 6x3, 3x3, 2x3 matrices.
    # Let's try to construct it.
    # We need sum_k U[alpha, k] V[beta, k] W[gamma, k] = T[alpha, beta, gamma].
    #
    # Let's guess a structure.
    # Maybe we can use the fact that it's a "diagonal" tensor.
    #
    # Actually, for (n, m, 1), the tensor is essentially the identity map from R^m to R^n x R^m? No.
    # It maps A (n x m) to C (n x 1).
    # C = A * b.
    # This is linear in A and b.
    #
    # Let's look at known results for small matrix multiplication.
    # (2, 2, 1): Rank is 4 (schoolbook). Can we do better?
    # T[i*2+j, j, i] = 1.
    # Slices:
    # i=0: (0,0), (1,1) -> rank 2.
    # i=1: (2,0), (3,1) -> rank 2.
    # Lower bound 2? No, lower bound is max slice rank.
    # For (2, 2, 1), slice i=0 is [[1, 0], [0, 1]] (rows alpha, cols beta). Rank 2.
    # Is rank 3 possible for (2, 2, 1)?
    # Strassen's algorithm is for (2, 2, 2) with rank 7.
    # For (2, 2, 1), it is known that the rank is 4.
    # Proof: The tensor is the sum of 4 basis tensors.
    #
    # So for (2, 3, 1), the slice i=0 has rank 3. The slice i=1 has rank 3.
    # Is the rank 6?
    # If the rank of (n, m, 1) is always n*m, then for (2, 3, 1) it is 6.
    # Then for (2, 3, 9), the rank would be 9*6 = 54.
    # This matches the schoolbook algorithm.
    #
    # Is it possible to do better than n*m for (n, m, 1)?
    # The tensor for (n, m, 1) is equivalent to the tensor for the bilinear map
    # (A, b) -> A b.
    # This is the "vector-matrix multiplication" tensor.
    # It is known that the rank of the tensor for (n, m, 1) is n*m.
    # Reference: "On the rank of the tensor of matrix multiplication" or similar.
    # Specifically, the tensor for (n, m, 1) is the transpose of the tensor for (1, m, n).
    # The rank of the tensor for (1, m, n) is the rank of the map R^m x R^n -> R^n? No.
    # The map is (v, M) -> v M.
    # This is a bilinear map.
    # It is known that the rank of this tensor is m*n.
    #
    # Therefore, the minimal rank for (2, 3, 9) is 9 * (rank of (2, 3, 1)) = 9 * 6 = 54.
    #
    # So the schoolbook algorithm is already optimal!
    #
    # Let's double check this logic.
    # The tensor T_{n,m,p} is the direct sum of p copies of T_{n,m,1}.
    # rank(T_{n,m,p}) >= p * rank(T_{n,m,1}).
    # Also rank(T_{n,m,p}) <= p * rank(T_{n,m,1}) because we can just repeat the decomposition p times.
    # So rank(T_{n,m,p}) = p * rank(T_{n,m,1}).
    #
    # Now, what is rank(T_{n,m,1})?
    # T_{n,m,1} has entries T[i*m+j, j, i] = 1.
    # This is the tensor for the bilinear form (A, b) -> A b.
    # A is n x m, b is m x 1.
    # The rank of this tensor is indeed n*m.
    # Why?
    # The flattening of the tensor along the first mode (A) is a matrix of size (n*m) x (m*n).
    # The rank of this flattening is n*m.
    # Since the rank of the tensor is at least the rank of any flattening,
    # rank(T) >= n*m.
    # Since we have a decomposition of rank n*m (schoolbook), rank(T) <= n*m.
    # Thus rank(T) = n*m.
    #
    # So the minimal rank is n*m*p = 2*3*9 = 54.
    # The current program returns rank 54.
    # It is already optimal.
    #
    # The task is to "Find a rank-minimal exact tensor decomposition".
    # The score is the rank.
    # The current score is 1.0, which corresponds to rank 54.
    # Since 54 is the minimum possible rank, we cannot improve the score.
    #
    # Wait, is there any trick?
    # "integer/half-integer entries"
    # The schoolbook decomposition uses 0s and 1s, which are integers.
    #
    # Is it possible that the tensor is NOT the direct sum?
    # The definition of the matrix multiplication tensor:
    # T_{ijk} = 1 if the indices correspond to a valid term in the product, 0 otherwise.
    # For C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has indices alpha (for A), beta (for B), gamma (for C).
    # alpha = i*m + j, beta = j*p + k, gamma = k*n + i.
    # T[alpha, beta, gamma] = 1 if alpha, beta, gamma are consistent, i.e.,
    # j_alpha = j_beta and i_alpha = i_gamma and k_beta = k_gamma.
    #
    # Let's check the direct sum property again.
    # Fix k.
    # The terms in the sum for C_{ik} involve A_{ij} and B_{jk}.
    # The indices in the tensor are:
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    #
    # If we fix k, then beta is in {k, p+k, 2p+k, ...} ? No, beta = j*p + k.
    # Since j ranges from 0 to m-1, beta ranges from k to (m-1)p + k.
    # And alpha ranges over all i*m+j.
    # gamma = k*n + i. Since i ranges 0..n-1, gamma ranges k*n .. k*n+n-1.
    #
    # The sub-tensor for a fixed k has non-zero entries only when:
    # beta ≡ k (mod p)
    # gamma ≡ i (mod n) where i is determined by alpha?
    #
    # Let's verify the orthogonality of the sub-tensors for different k.
    # Suppose k1 != k2.
    # Can there be a term in the decomposition that contributes to both C_{i k1} and C_{i k2}?
    # A rank-1 term is u * v * w.
    # It contributes to T[alpha, beta, gamma] if u[alpha] v[beta] w[gamma] != 0.
    # If this term is part of the decomposition, it must satisfy the tensor equation.
    #
    # If we use the decomposition of the (n, m, 1) tensor, we have matrices U0, V0, W0.
    # U0: n*m x r
    # V0: m x r
    # W0: n x r
    #
    # We construct U, V, W for the (n, m, p) tensor as follows:
    # Rank = p * r.
    # For each k in 0..p-1:
    #   For each r' in 0..r-1:
    #     col = k*r + r'
    #
    #     U[alpha, col] = U0[alpha, r']
    #
    #     V[beta, col] = V0[j, r'] if beta = j*p + k, else 0
    #     (This is a permutation/selection of columns in V0)
    #
    #     W[gamma, col] = W0[i, r'] if gamma = k*n + i, else 0
    #     (This is a permutation/selection of rows in W0? No, W0 is n x r. W is p*n x rank.
    #      W[gamma, col] = W0[gamma - k*n, r'] if 0 <= gamma - k*n < n)
    #
    # Let's check if this works.
    # Sum_col U[alpha, col] V[beta, col] W[gamma, col]
    # = sum_k sum_{r'} U0[alpha, r'] * (V0[j, r'] if beta=j*p+k else 0) * (W0[i, r'] if gamma=k*n+i else 0)
    #
    # The term is non-zero only if:
    # 1. beta = j*p + k for some j. This implies k = beta mod p.
    # 2. gamma = k*n + i for some i. This implies i = gamma mod n? No, gamma // n = k.
    #    So k = gamma // n.
    #
    # For the term to be non-zero, we must have k = beta mod p AND k = gamma // n.
    # This fixes k uniquely (if it exists).
    # So the sum over k collapses to at most one term.
    #
    # If such a k exists, let j = (beta - k) / p and i = (gamma - k*n) / n.
    # Then the sum is:
    # sum_{r'} U0[alpha, r'] V0[j, r'] W0[i, r']
    #
    # This must equal T[alpha, beta, gamma].
    # T[alpha, beta, gamma] is 1 if alpha = i*m + j (where i, j are the indices from the product)
    # and the indices are consistent.
    #
    # In our construction, i and j are derived from beta and gamma.
    # alpha is free.
    # The sum is U0[alpha, :] . V0[j, :]^T * W0[i, :] ... wait, it's a tensor contraction.
    # sum_{r'} U0[alpha, r'] V0[j, r'] W0[i, r'].
    #
    # We know that U0, V0, W0 decompose the (n, m, 1) tensor.
    # The (n, m, 1) tensor T0 has entries T0[alpha, j, i] = 1 if alpha = i*m + j, else 0.
    # So sum_{r'} U0[alpha, r'] V0[j, r'] W0[i, r'] = T0[alpha, j, i].
    #
    # So the full sum is T0[alpha, j, i].
    #
    # Now, does T0[alpha, j, i] equal T[alpha, beta, gamma]?
    # T[alpha, beta, gamma] is 1 if:
    # - beta = j*p + k
    # - gamma = k*n + i
    # - alpha = i*m + j
    #
    # In our case, we have fixed k, j, i from beta and gamma.
    # So we need to check if alpha = i*m + j.
    # T0[alpha, j, i] is 1 if and only if alpha = i*m + j.
    #
    # So yes, the constructed decomposition is valid.
    # And the rank is p * rank(T_{n,m,1}).
    #
    # Since rank(T_{n,m,1}) = n*m, the total rank is n*m*p.
    #
    # Thus, the schoolbook algorithm is optimal.
    #
    # The current program implements the schoolbook algorithm.
    # The score is 1.0 (rank 54).
    #
    # Can we do anything?
    # The task says "score is the rank".
    # If the rank is minimal, the score is maximal?
    # "score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # "The evaluator measures the raw value (lower is better)."
    #
    # Wait.
    # "score is seed/value"
    # Seed value is 54.
    # If I find a decomposition with rank 54, value = 54.
    # score = 54/54 = 1.0.
    #
    # If I find a decomposition with rank 50, value = 50.
    # score = 54/50 = 1.08.
    #
    # So lower rank is better.
    #
    # But I just proved that the rank cannot be lower than 54.
    #
    # Is my proof correct?
    # rank(T_{n,m,1}) = n*m.
    # Let's re-verify for (2, 2, 1).
    # n=2, m=2.
    # T0[alpha, j, i].
    # alpha in 0..3, j in 0..1, i in 0..1.
    # Non-zeros:
    # i=0: (0,0,0), (1,1,0)
    # i=1: (2,0,1), (3,1,1)
    #
    # Flattening along alpha (size 4 x 2*2=4):
    # Matrix M where M[alpha, (j,i)] = T0[alpha, j, i].
    # M[0, (0,0)] = 1
    # M[1, (1,0)] = 1
    # M[2, (0,1)] = 1
    # M[3, (1,1)] = 1
    # All other entries 0.
    # This is a permutation matrix (or part of one).
    # The rows are distinct basis vectors.
    # Rank is 4.
    # So rank(T) >= 4.
    # Schoolbook rank is 4.
    # So rank is 4.
    #
    # For (2, 3, 1):
    # n=2, m=3.
    # Flattening along alpha (size 6 x 3*2=6).
    # Non-zeros:
    # i=0: (0,0,0), (1,1,0), (2,2,0)
    # i=1: (3,0,1), (4,1,1), (5,2,1)
    # Matrix is 6x6.
    # Rows:
    # 0: 1 at (0,0)
    # 1: 1 at (1,1)
    # 2: 1 at (2,2)
    # 3: 1 at (0,3)  (j=0, i=1 -> index 0*2+1=3)
    # 4: 1 at (1,4)
    # 5: 1 at (2,5)
    #
    # The columns are:
    # 0: 1 at row 0, 1 at row 3
    # 1: 1 at row 1, 1 at row 4
    # 2: 1 at row 2, 1 at row 5
    # 3: 0
    # 4: 0
    # 5: 0
    #
    # Wait, the flattening index is (j, i).
    # j in 0..2, i in 0..1.
    # Index = j*n + i? Or j + i*m?
    # Usually row-major or col-major.
    # Let's say index = j*2 + i.
    # (0,0) -> 0
    # (1,0) -> 2
    # (2,0) -> 4
    # (0,1) -> 1
    # (1,1) -> 3
    # (2,1) -> 5
    #
    # Non-zeros:
    # (0,0,0) -> alpha=0, col=0
    # (1,1,0) -> alpha=1, col=2
    # (2,2,0) -> alpha=2, col=4
    # (3,0,1) -> alpha=3, col=1
    # (4,1,1) -> alpha=4, col=3
    # (5,2,1) -> alpha=5, col=5
    #
    # The matrix is:
    # 1 0 0 0 0 0
    # 0 0 1 0 0 0
    # 0 0 0 0 1 0
    # 0 1 0 0 0 0
    # 0 0 0 1 0 0
    # 0 0 0 0 0 1
    #
    # This matrix has 6 non-zero entries, one in each row and one in each column.
    # It is a permutation matrix.
    # Rank is 6.
    #
    # So rank(T_{2,3,1}) >= 6.
    # And schoolbook gives 6.
    # So rank is 6.
    #
    # Total rank for (2, 3, 9) is 9 * 6 = 54.
    #
    # It seems the schoolbook algorithm is indeed optimal.
    #
    # Is there any other interpretation?
    # "integer/half-integer entries"
    # Maybe a different decomposition with the same rank but "better" in some other way?
    # No, the score is based on rank.
    #
    # Could the rank be lower if we allow fractional entries?
    # The rank of a tensor is an integer.
    # The minimal rank is the minimal number of rank-1 tensors needed.
    # The lower bound from flattening is rigorous.
    #
    # So the score cannot exceed 1.0.
    #
    # I will just return the schoolbook decomposition.
    #
    # Wait, the previous attempt failed source validation.
    # "empty" reason.
    # Maybe the code was malformed?
    # I will write clean, simple code.

    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                
                U[alpha][col] = 1
                V[beta][col] = 1
                W[gamma][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
