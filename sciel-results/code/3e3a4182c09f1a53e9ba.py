<<<<<<< SEARCH
def _verify(U, V, W):
    """Check that U * V^T * W^T reconstructs the multiplication tensor.
    Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor T[i,j,k] = 1 if we compute C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # U index: i*m + j ; V index: j*p + k ; W index: k*n + i
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build T from decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    # Expected: T[i,j,k] = 1
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)
=======
def _verify(U, V, W):
    """Check that the decomposition exactly reconstructs the multiplication tensor.
    Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # We need to check if for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient check using einsum on the rank dimension
    # U_flat: (nm, rank)
    # V_flat: (mp, rank)
    # W_flat: (pn, rank)
    
    # We can reshape U, V, W to 3D arrays?
    # U is indexed by i, j. Reshape to (n, m, rank).
    U_3d = U.reshape(n, m, rank)
    # V is indexed by j, k. Reshape to (m, p, rank).
    V_3d = V.reshape(m, p, rank)
    # W is indexed by k, i. Reshape to (p, n, rank).
    W_3d = W.reshape(p, n, rank)
    
    # We want T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
    
    # This is a contraction over c, and specific index alignments.
    # Let's just loop over i, j, k for simplicity as dimensions are small (2,4,8)
    # 2*4*8 = 64 checks. Each check is a dot product of length rank (56).
    # 64 * 56 operations is trivial.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                val = np.dot(np.dot(u_val, v_val), w_val)
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True
>>>>>>> REPLACE

