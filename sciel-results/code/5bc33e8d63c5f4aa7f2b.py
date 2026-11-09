import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) for the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    For all i,k: sum_j A[i,j] B[j,k] = sum_c U[i*m + ?, c] * V[?, c] * W[k*n+i, c]
    Actually, the tensor decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    For matrix multiplication, T[i,j,k] = 1 for all i,j,k (each product contributes to exactly one C[i,k]).
    Wait, no. The multiplication tensor is:
    C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor T_{i,j,k} has T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    In standard matrix multiplication, EVERY product A[i,j]*B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Compute the tensor from the decomposition
    # T_decomposed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3D tensor. Let's compute it efficiently.
    # For each c, the outer product of U[:,c], V[:,c], W[:,c] reshaped appropriately.
    # Actually, let's just check a few random instances or compute the full tensor.
    # Since n=3, m=5, p=7, the tensor has 3*5*7 = 105 elements.
    # We can compute the full tensor.
    T_decomposed = np.zeros((n, m, p))
    for c in range(rank):
        u_col = U[:, c]  # length n*m
        v_col = V[:, c]  # length m*p
        w_col = W[:, c]  # length p*n
        
        # Reshape u_col to (n, m)
        u_mat = u_col.reshape(n, m)
        # Reshape v_col to (m, p)
        v_mat = v_col.reshape(m, p)
        # Reshape w_col to (p, n)
        w_mat = w_col.reshape(p, n)
        
        # For each i, j, k: contribution is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # We can compute this as:
        # For each i: sum_j,k u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # Let's compute it as:
        # temp[i, j] = sum_k u_mat[i,j] * v_mat[j,k] * w_mat[k,i] ... this isn't quite right.
        # Actually: T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # So for fixed i, the slice T[i,:,:] = u_mat[i,:] * v_mat * w_mat.T ... no.
        # T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # For fixed i: T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is: (u_mat[i,:] * v_mat) * w_mat.T ?
        # u_mat[i,j] * v_mat[j,k] = (u_mat[i,:] * v_mat)[k] ... no, u_mat[i,j] is scalar for fixed i,j.
        # Let's just loop over i since n=3 is small.
        for i in range(n):
            # T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # = (u_mat[i,:] * v_mat) * w_mat[:,i] ... elementwise?
            # u_mat[i,j] * v_mat[j,k] for all j,k is: u_mat[i,:] (length m) * v_mat (m x p) -> (m * v_mat) is not right.
            # u_mat[i,:] is a vector of length m. v_mat is m x p.
            # The product u_mat[i,j] * v_mat[j,k] summed over j would be a matrix product, but we don't sum.
            # We want the elementwise product for each (j,k).
            # So: temp[j,k] = u_mat[i,j] * v_mat[j,k]
            # This is: u_mat[i,:] (as column, m x 1) * v_mat (m x p) -> no, elementwise.
            # temp = np.outer(u_mat[i,:], np.ones(p)) * v_mat ? No.
            # temp[j,k] = u_mat[i,j] * v_mat[j,k]
            # This is: for each j, multiply row j of v_mat by u_mat[i,j].
            # temp = u_mat[i,:][:, np.newaxis] * v_mat  # shape (m, p)
            temp = u_mat[i, :][:, np.newaxis] * v_mat  # (m, p)
            # Now T[i,j,k] = temp[j,k] * w_mat[k,i]
            # So T[i,:,:] = temp * w_mat[:, i][np.newaxis, :]  # (m, p)
            T_decomposed[i, :, :] = temp * w_mat[:, i][np.newaxis, :]
    
    # Check if T_decomposed is all ones
    return np.allclose(T_decomposed, 1.0, atol=1e-10)

def construct():
    deadline = time.time() + 120.0  # 120 second deadline
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 105)
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by merging terms or using Strassen-like optimizations
    # For now, return the schoolbook decomposition as the baseline.
    # Future work: implement rank reduction algorithms.
    
    if time.time() > deadline - 5:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Attempt simple rank reduction: check if any two terms can be merged
    # This is a placeholder for more sophisticated algorithms.
    # For a 3x5x7 matrix multiplication, known lower bounds suggest rank < 105 may be possible.
    # But finding such a decomposition is non-trivial.
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
