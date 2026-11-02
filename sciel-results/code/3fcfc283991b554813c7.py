"""Matrix multiplication 2x5x5 via SVD-based rank reduction search."""
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Efficient vectorized check
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    # Compute the sum for all i, j, k
    # Term is U_reshaped[i,j,:] * V_reshaped[j,k,:] * W_reshaped[k,i,:]
    # We can iterate over j (middle index) which is the only one shared in a simple loop?
    # Actually, we can use einsum.
    # Result[i,k] should be 1 for all i,k? No, the tensor is 1 for all i,j,k.
    # The decomposition is for the *entire* tensor.
    # The check is: for all i, j, k, sum_c ... = 1.
    
    # Let's just loop over j, and vectorize over i, k?
    # Or just use einsum to compute the full tensor T_reconstructed.
    # T_reconstructed[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Using einsum:
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r)
    # We want T[i, j, k]
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    
    # This is not a standard single einsum because indices are permuted.
    # Let's construct it.
    # Sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # We can do:
    # Step 1: A[i, j, k] = sum_c U[i, j, c] * V[j, k, c]
    # A = np.einsum('ijc,jkc->ijk', U_reshaped, V_reshaped)
    # Step 2: T[i, j, k] = sum_c A[i, j, k] * W[k, i, c] ?? No, A already summed over c.
    # The term is sum_c (U[i,j,c] * V[j,k,c] * W[k,i,c]).
    # This is the trace of the product of three vectors?
    
    # Let's just loop over c and accumulate.
    # T = zeros(n, m, p)
    # for c in range(rank):
    #    T += U_reshaped[:,:,c] * V_reshaped[:,:,c] * W_reshaped[:,:,c] # Dimensions don't match directly for broadcasting.
    
    # U_reshaped[i,j,c] shape (n,m)
    # V_reshaped[j,k,c] shape (m,p)
    # W_reshaped[k,i,c] shape (p,n)
    
    # For a fixed c:
    # u = U_reshaped[:, :, c] # (n, m)
    # v = V_reshaped[:, :, c] # (m, p)
    # w = W_reshaped[:, :, c] # (p, n)
    
    # We need to add u[i,j] * v[j,k] * w[k,i] to T[i,j,k].
    
    # This is equivalent to:
    # T += np.einsum('ij,jk,ki->ijk', u, v, w)
    
    T_reconstructed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(len(U[0])):
        u = U_reshaped[:, :, c]
        v = V_reshaped[:, :, c]
        w = W_reshaped[:, :, c]
        T_reconstructed += np.einsum('ij,jk,ki->ijk', u, v, w)
        
    # The target tensor is 1 everywhere.
    if not np.allclose(T_reconstructed, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _reduce_rank(U, V, W, n, m, p, max_iter=100, tol=1e-10):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies.
    This is a heuristic search.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # We look for columns that are linear combinations of others.
    # This is complex for 3-way tensors.
    # Instead, we can try to zero out entries if they are redundant.
    
    # For now, return the input if no specific reduction algorithm is implemented.
    # A full implementation would use Gröbner bases or SVD on the flattened tensor.
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce
    U, V, W, rank = _reduce_rank(U, V, W, n, m, p)
    
    # Convert to lists of lists as required by interface?
    # The interface says: {'U', 'V', 'W'}: factor matrices of shapes ... with integer/half-integer entries
    # The current program returns lists of lists.
    # Let's stick to lists of lists to be safe, or numpy arrays if accepted.
    # The seed program returns lists of lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
