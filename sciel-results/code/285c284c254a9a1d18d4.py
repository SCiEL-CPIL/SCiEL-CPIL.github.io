<<<<<<< SEARCH
def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        if vc[j, k] != 0 and wc[k, i] != 0:
                            result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.allclose(result, 1.0, atol=1e-9)
=======
def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Vectorized: result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum for efficiency
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(result, 1.0, atol=1e-9)
>>>>>>> REPLACE

