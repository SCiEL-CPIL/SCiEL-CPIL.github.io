<<<<<<< SEARCH
def _check_decomposition(n, m, p, U, V, W):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for l in range(p):
                    # Sum over c: U[i*m+j, c] * V[jp*p+l, c] * W[l*n+i, c]
                    s = np.sum(U[i * m + j, :] * V[jp * p + l, :] * W[l * n + i, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True
=======
def _check_decomposition(n, m, p, U, V, W):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                target[i, j, j, l] = 1.0
    
    return _check_decomposition_fast(n, m, p, U, V, W, target)
>>>>>>> REPLACE

