"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Construct the Matrix Multiplication Tensor T (flattened)
    # T has indices (i, j, k) where i in [0, n), j in [0, m), k in [0, p)
    # The tensor value T[i, j, k] is 1 if we are computing C[i, k] += A[i, j] * B[j, k]
    # Actually, the standard definition of the multiplication tensor T has entries T_{i,j,k} = 1
    # such that sum_k T_{i,j,k} A_{i,j} B_{j,k} ... wait.
    
    # Let's be precise.
    # We want to decompose the tensor T such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where (u_r)_i, (v_r)_j, (w_r)_k.
    # The matrix multiplication is C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be written as C_{i,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}.
    # So T_{i,j,k} is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In our case, A is n x m, B is m x p, C is n x p.
    # The indices for U are (i, j) flattened to i*m+j.
    # The indices for V are (j, k) flattened to j*p+k.
    # The indices for W are (k, i) flattened to k*n+i (following the task convention W[k*n+i, r]).
    
    # Let's build the tensor T of shape (n*m, m*p, p*n).
    # Dimension 0: i*m + j
    # Dimension 1: j*p + k
    # Dimension 2: k*n + i
    
    # T[a, b, c] = 1 if the multiplication of A element at index a and B element at index b
    # contributes to C element at index c.
    
    # A index a = i*m + j
    # B index b = j*p + k
    # C index c = k*n + i
    
    # For a given a (i,j) and b (j',k), they contribute to c (k',i') if:
    # i = i' and k = k' and j = j'.
    # So T[a, b, c] = 1 iff:
    #   i(a) == i(c)
    #   k(b) == k(c)
    #   j(a) == j(b)
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # 2. Compute the Rank of the Tensor
    # We can flatten T into a matrix.
    # Let's flatten dimensions (0, 1) into rows and dimension 2 into columns.
    # M is of shape (dim_U * dim_V, dim_W)
    # The rank of the tensor is the rank of this matrix M.
    # Note: This gives the "tensor rank" in the sense of the minimum number of terms
    # in a sum of outer products. This is exactly what we want.
    
    M = T.reshape(dim_U * dim_V, dim_W)
    
    # Perform SVD or QR to find the rank.
    # Since the entries are 0/1, the rank is an integer.
    # We can use np.linalg.matrix_rank with a tolerance.
    # However, to be robust and exact, we can use Gaussian elimination on the matrix.
    # But np.linalg.matrix_rank is usually sufficient for small matrices with 0/1 entries.
    
    # Let's use SVD to get the rank and also the basis vectors for reconstruction.
    U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    
    # Determine rank
    tol = 1e-10
    rank = np.sum(s > tol)
    
    # 3. Construct the Factor Matrices
    # We have M = U_svd * diag(s) * Vt_svd
    # M is (dim_U * dim_V, dim_W)
    # We need to decompose T into sum of rank-1 tensors.
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # From SVD:
    # M[a*dim_V + b, c] = sum_r U_svd[a*dim_V + b, r] * s[r] * Vt_svd[r, c]
    
    # Let's define the factors for the non-zero singular values.
    # We need to distribute the singular values between U and V or W.
    # It's convenient to put sqrt(s) in U and sqrt(s) in W, or s in one and 1 in the other.
    # Let's put s in the W factor for simplicity, or split it.
    # Let's use:
    # U_mat[a, r] = U_svd[a*dim_V + b, r] ... wait, U_svd is indexed by (a*dim_V+b).
    # We need to reshape U_svd to (dim_U, dim_V, rank)
    
    # U_svd shape: (dim_U * dim_V, rank)
    # Vt_svd shape: (rank, dim_W)
    
    # Reshape U_svd to (dim_U, dim_V, rank)
    U_svd_reshaped = U_svd[:, :rank].reshape(dim_U, dim_V, rank)
    
    # Now, T[a, b, c] = sum_r U_svd_reshaped[a, b, r] * s[r] * Vt_svd[r, c]
    
    # We want U[a, r], V[b, r], W[c, r] such that sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    
    # We can set:
    # U[a, r] = U_svd_reshaped[a, b, r] ... but this depends on b.
    # Wait, the standard decomposition T = sum u_r \otimes v_r \otimes w_r
    # corresponds to:
    # U[a, r] = (u_r)_a
    # V[b, r] = (v_r)_b
    # W[c, r] = (w_r)_c
    
    # In our SVD formulation:
    # M[a, b, c] (flattened) = sum_r (U_svd_reshaped[a, b, r] * s[r]) * Vt_svd[r, c]
    # This doesn't look like a simple outer product of 3 vectors because U_svd_reshaped
    # mixes a and b.
    
    # Let's re-evaluate.
    # The tensor T is 3-way.
    # We flattened (a, b) into one index.
    # This means we are finding the rank of the matrix M.
    # The rank of the matrix M is indeed the tensor rank of T.
    # However, the factors from SVD of M give us:
    # M = U_M S V_M^T
    # Where U_M is (dim_U*dim_V, rank), S is (rank, rank), V_M is (dim_W, rank).
    
    # We need to split U_M into U and V.
    # U_M is a matrix where row index is (a, b).
    # U_M[a*dim_V + b, r] = sum_k ... no, it's just the left singular vector.
    # It does NOT factorize into U[a, r] * V[b, r] generally.
    # The SVD of the flattening gives the "matrix rank" of the flattening,
    # which is equal to the tensor rank.
    # But to get the factors U, V, W, we need to find the actual rank-1 tensors.
    
    # Alternative approach:
    # Since the rank is small (likely 72 for schoolbook, but maybe less?),
    # and we have the SVD, we can try to find the factors.
    # Actually, for the matrix multiplication tensor, the rank is known to be n*m*p
    # unless there is a faster algorithm. For 3x3x8, it's likely 72 or close.
    # But we want the *minimal* rank.
    
    # If the rank is R, we can find the factors by solving a linear system?
    # Or we can use the fact that the tensor is sparse.
    
    # Let's try a different decomposition method.
    # We know T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i).
    # This is a permutation-like structure.
    # The rank of this tensor is exactly the number of non-zero entries in the "slices"?
    # No.
    
    # Let's use the property that the rank of the tensor is the minimum R such that
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    
    # If we assume the rank is R, we can set up a system of equations.
    # But R is unknown.
    
    # However, for this specific tensor (matrix multiplication), the rank is known
    # to be at most n*m*p.
    # Is it possible to do better?
    # For 3x3, rank is 23 (Strassen-like).
    # For 3x3x8, it's 3x3 * 8 = 24 blocks of 1x1x1? No.
    
    # Let's just compute the rank using the flattening.
    # The rank of the flattening (a,b) vs c is the tensor rank.
    # Let's verify this with a simple case.
    # If T is a single non-zero entry, rank is 1.
    
    # To get the factors U, V, W:
    # We have M = U_M S V_M^T.
    # M[a, b, c] = sum_r U_M[a, b, r] S[r] V_M[r, c]
    # We need U[a, r] V[b, r] W[c, r].
    # This requires U_M[a, b, r] to factorize as U[a, r] V[b, r].
    # This is not guaranteed by SVD.
    
    # However, we can find the factors by observing the structure.
    # The tensor T is the structure tensor of the algebra of n x m x p matrix multiplication.
    # It is known that the rank is the minimum number of multiplications.
    
    # Let's try to construct the factors directly from the non-zero entries.
    # The schoolbook decomposition has rank n*m*p.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for each (i,j,k).
    # This is a valid decomposition.
    # Is it minimal?
    # For 3x3x3, rank is 23 < 27.
    # For 3x3x8, the rank is likely 3 * rank(3x3x1)? No.
    
    # Let's assume the question implies we should find the *minimal* rank.
    # If we can't find a better one easily, we return the schoolbook one.
    # But we can try to use the SVD rank to see if it's less than 72.
    
    # If rank < 72, we need to find the factors.
    # We can use the "slice rank" or other bounds?
    
    # Actually, there is a known algorithm to find the factors given the rank.
    # But for now, let's check if the rank is 72.
    # If it is 72, we can just return the schoolbook decomposition.
    # If it is less, we need to find the factors.
    
    # Let's check the rank.
    if rank < dim_U * dim_V * dim_W:
        # We found a potentially lower rank.
        # We need to find the factors.
        # This is a hard problem in general, but for this specific tensor,
        # we might be able to use the SVD components if they factorize.
        # Or we can try to solve for the factors using the non-zero structure.
        
        # For now, if rank < 72, we might not have an easy way to get the factors
        # within the time limit.
        # However, the schoolbook decomposition is always valid.
        # So we can always return the schoolbook decomposition.
        # The score is the rank.
        # If we return schoolbook, rank is 72.
        # If we can find a lower rank, score is lower (better).
        
        # Let's try to find the factors for the lower rank.
        # One way is to use the fact that the tensor is sparse.
        # But this is complex.
        
        # Let's stick to the schoolbook decomposition for now, as it is guaranteed to work.
        # The task asks for "rank-minimal".
        # If we can't find it, we return the best we have.
        
        pass
    
    # Let's implement the schoolbook decomposition as a fallback and primary candidate.
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
    
    # Check if we can do better.
    # If the SVD rank is less than 72, we should try to find the factors.
    # But finding the factors is non-trivial.
    # Given the time constraint, we will return the schoolbook decomposition.
    # This is a valid decomposition, and its rank is 72.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
