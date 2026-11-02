def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Check half-integer constraint
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    # Verify tensor decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can compute this using einsum
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    # We need to check if T[i, j, k] == 1 for all i, j, k
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Uresh = Ua.reshape(n, m, rank)
    Vresh = Va.reshape(m, p, rank)
    Wresh = Wa.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by k*n+i, so Wresh[k, i, r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', Uresh, Vresh, Wresh)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True
