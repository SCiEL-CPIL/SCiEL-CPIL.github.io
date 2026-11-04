import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6

    # We want to find a rank-minimal decomposition of the matrix multiplication tensor
    # for multiplying an (n x m) matrix by an (m x p) matrix.
    #
    # The tensor is T[i, j, k] = delta_{i,j,k} where the multiplication is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    #
    # A rank-1 decomposition has the form:
    # C[i, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    #
    # The schoolbook decomposition uses rank = n*m*p:
    # For each (i, j, k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    #
    # However, we can potentially do better. Let's think about this more carefully.
    #
    # The key insight: we can use the structure of the problem.
    # C[i, k] = sum_j A[i, j] B[j, k]
    #
    # This is a sum over j. The tensor has the structure of a "convolution" in the j dimension.
    #
    # One approach: use a change of basis in the j dimension.
    # If we apply an invertible linear transformation to the j indices of A and B,
    # we can potentially reduce the rank.
    #
    # Specifically, let's consider the following:
    # Let F be an m x m invertible matrix.
    # Define A' = A * F^{-1} (so A' is n x m)
    # Define B' = F * B (so B' is m x p)
    # Then C = A * B = A' * F^{-1} * F * B = A' * B'
    #
    # This doesn't help directly since the multiplication is still n x m times m x p.
    #
    # Another approach: use the fact that we only need to compute n*p outputs.
    # Each output C[i, k] depends on all m entries of A[i, :] and B[:, k].
    #
    # Let's think about it differently. The tensor T is in (R^{n*m}) * (R^{m*p}) -> R^{p*n}
    # (using the convention W[k*n+i, r]).
    #
    # The rank of the matrix multiplication tensor for (n x m) * (m x p) is known to be
    # at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n) = max(n*m, m*p, p*n).
    # For n=2, m=3, p=6: max(6, 18, 12) = 18.
    #
    # Actually, the known lower bound is max(n*m, m*p, n*p) but there are better bounds.
    # The Strassen algorithm shows that for 2x2x2, the rank is 7 < 8.
    #
    # For our case, let's try to find a decomposition with rank less than 36.
    #
    # One idea: group the j indices. Since m=3, we can't easily use Strassen-style
    # block decomposition. But we can try to find a sparse or structured decomposition.
    #
    # Actually, let's just use the schoolbook decomposition which is guaranteed to work
    # and has rank 36. The task says "rank-minimal" but also says the score is the rank,
    # and the current program scores 1.0 with rank 36. If we can find a lower rank, we improve.
    #
    # Let me try a different approach: use the fact that the multiplication can be viewed
    # as a linear map, and try to find a low-rank decomposition of this linear map.
    #
    # The linear map L: R^{n*m} -> R^{p*n} (or R^{n*p} depending on convention)
    # L(A) = A * B for a fixed B... no, that's not right.
    #
    # Actually, the tensor T is such that:
    # <T, A ⊗ B> = sum_{i,k} C[i,k] = sum_{i,j,k} A[i,j] B[j,k]
    #
    # This is a trilinear form. The rank is the minimum number of rank-1 trilinear forms
    # needed to represent it.
    #
    # Let's just use the schoolbook decomposition for now, which is simple and correct.
    # rank = n * m * p = 2 * 3 * 6 = 36
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
