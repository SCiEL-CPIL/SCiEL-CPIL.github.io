import numpy as np
import time

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r]
    # and check T[i,j,k] == W[k*n+i, r] summed over r... wait, W is a factor matrix.
    # Actually: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] contributes)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j,k
    # And W is used as: output[i,k] = sum_r W[k*n+i, r] * (intermediate_r)
    # where intermediate_r = sum_j A[i,j] * V[j*p+k, r]... 
    
    # Let me re-read the convention: W[k*n+i, r]
    # The bilinear algorithm: C = A*B
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # No wait, that's not right either. Standard tensor decomposition:
    # C[i,k] = sum_r (sum_j A[i,j] * U[r, i*m+j]) * (sum_j B[j,k] * V[r, j*p+k]) * W[r, k*n+i]
    # Hmm, let me think more carefully.
    
    # Standard: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Bilinear: C[i,k] = sum_r alpha_r * beta_r * gamma_r
    # where alpha_r = sum_j A[i,j] * U[r, i*m+j]  (but this depends on i which is in C)
    
    # Actually the standard form is:
    # C[i,k] = sum_r (sum_j A[i,j] * U[r, j]) * (sum_j B[j,k] * V[r, j]) * W[r]
    # No... let me use the convention from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    
    # So we check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :])
                # But we also need W... 
                # Wait, if it's just U*V then where does W come in?
                # Let me reconsider: maybe the convention is that the tensor is
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # And T[i,j,k] should equal 1 for all i,j,k (since A[i,j]*B[j,k] = 1*1)
                pass
    
    # Actually for the trivial decomposition, U[i*m+j, r]=1 only when r corresponds to (i,j,k),
    # V[j*p+k, r]=1 only when r corresponds to (i,j,k), W[k*n+i, r]=1 only when r corresponds to (i,j,k)
    # So the product is 1 only for the matching r, and 0 otherwise. Sum = 1. Correct.
    
    # General check:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                if abs(np.sum(prod) - 1.0) > 1e-9:
                    return False
    return True


def _try_reduce(U, V, W, n, m, p, max_time=100.0):
    """Try to reduce rank by finding linear dependencies."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Strategy: try to merge pairs of terms
    # For terms r1, r2, check if we can replace them with a single term
    # This is complex; instead, try to find if any column is a linear combination of others
    # in a way that preserves the tensor identity
    
    # Simpler approach: use the fact that if we have a decomposition of rank R,
    # we can check if it's actually rank R-1 by looking for linear dependencies
    # in the "slice" tensors
    
    # For now, return the initial decomposition
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    u_cols = (i_idx * m + j_idx).flatten()
    v_cols = (j_idx * p + k_idx).flatten()
    w_cols = (k_idx * n + i_idx).flatten()
    
    r_cols = np.arange(rank)
    
    U[u_cols, r_cols] = 1
    V[v_cols, r_cols] = 1
    W[w_cols, r_cols] = 1
    
    # Try to reduce rank (with time limit)
    U, V, W, final_rank = _try_reduce(U, V, W, n, m, p, max_time=100.0)
    
    # Convert to integers if possible
    U = np.round(U).astype(int)
    V = np.round(V).astype(int)
    W = np.round(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
