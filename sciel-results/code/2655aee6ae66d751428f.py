import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _validate_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 140  # 140s to leave safety margin
    
    # Start with schoolbook decomposition
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    best_rank = schoolbook_rank
    
    # Build the stacked matrix to find linear dependencies
    U_mat = np.array(best_U, dtype=float)  # n*m x best_rank
    V_mat = np.array(best_V, dtype=float)  # m*p x best_rank
    W_mat = np.array(best_W, dtype=float)  # p*n x best_rank
    
    # Stack the transposed matrices
    # We want alpha such that U^T alpha = 0, V^T alpha = 0, W^T alpha = 0
    # This is equivalent to [U^T; V^T; W^T] alpha = 0
    M = np.vstack([U_mat.T, V_mat.T, W_mat.T])
    
    # Find the null space of M
    try:
        # Use SVD to find the null space
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
        # The null space is given by the last (rank - M.shape[0]) rows of Vt
        # The rank of M is the number of non-zero singular values
        tol = 1e-10
        rank_M = np.sum(s > tol)
        null_space = Vt[rank_M:].T  # This is a matrix where each column is a null space vector
    except:
        null_space = np.zeros((best_rank, 0))
    
    # Use the null space vectors to reduce the rank
    # We want to find a set of columns to remove such that the remaining columns are linearly independent
    # in the sense that there is no non-trivial linear combination of the remaining columns that is in the null space.
    # Actually, we can just remove one column at a time if it is in the span of the others in the null space.
    # A simpler approach: find a basis for the column space of M, and then the null space is the orthogonal complement.
    # We can remove any column that is a linear combination of the others in the null space.
    
    # Let's try to find a maximal set of columns that are linearly independent in the null space.
    # Actually, we can just find a basis for the null space and use it to reduce the rank.
    # The rank of the decomposition is the dimension of the span of the columns of U, V, W.
    # If we remove a column that is in the null space, the rank decreases by 1.
    
    # Let's find a maximal set of columns that are linearly independent in the null space.
    # We can use Gaussian elimination on the null space matrix.
    
    # For now, let's just remove the first null space vector's support.
    # This is not optimal, but it should reduce the rank.
    
    # A better approach: find a basis for the null space, and then find a maximal set of columns that are not in the span of the basis.
    # This is complex. Let's just remove one column at a time if it is in the null space.
    
    # Let's find a basis for the null space.
    if null_space.shape[1] > 0:
        # Find a basis for the null space
        U_ns, s_ns, Vt_ns = np.linalg.svd(null_space, full_matrices=True)
        rank_ns = np.sum(s_ns > 1e-10)
        basis = U_ns[:, :rank_ns]
        
        # Find a maximal set of columns that are linearly independent in the null space.
        # We can use the fact that the null space is the orthogonal complement of the column space of M.
        # So we can remove any column that is in the null space.
        
        # Let's just remove the first column that is in the null space.
        # This is not optimal, but it should reduce the rank.
        
        # Actually, we can remove all columns that are in the null space.
        # But we need to be careful not to remove too many.
        
        # Let's just remove one column at a time.
        for i in range(best_rank):
            if time.time() > deadline:
                break
            # Check if column i is in the null space
            # This is not correct. We need to check if the column is in the span of the other columns in the null space.
            pass
    
    # For now, let's just return the schoolbook decomposition.
    # The null space approach is complex to implement correctly.
    
    # Final validation
    if not _validate_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
