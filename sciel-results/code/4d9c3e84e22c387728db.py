def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, -1)
    V_3d = V_np.reshape(m, p, -1)
    W_3d = W_np.reshape(p, n, -1)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # einsum: i j r, j k r, k i r -> i j k
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True
