def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    n, m, p = 3, 3, 5
    
    # Build the target tensor
    # T[a, b, c] = 1 if a%3 == b//5 and b%5 == c//3, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that sum_t U[:,t]*V[:,t]*W[:,t] == T"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        result = np.zeros((n*m, m*p, p*n))
        for t in range(rank):
            result += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        return np.allclose(result, T, atol=tol)
    
    # Try rank-37 decomposition: 19 for 3x3x3 + 9 for col3 + 9 for col4
    # First, let me just use schoolbook for the 3x3x3 part (27 terms) + 9 + 9 = 45
    # That's the same as full schoolbook.
    
    # Let me try to use a rank-19 decomposition for 3x3x3.
    # I'll use a known decomposition and verify it.
    
    # For now, let me just use schoolbook rank 45 as a fallback
    # and try to find a better one.
    
    # Schoolbook rank 45
    rank = 45
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    # Verify schoolbook
    assert verify(U, V, W), "Schoolbook verification failed!"
    
    # Now try to find a better decomposition
    # For the 3x3x3 part, try rank-19
    # I'll use a specific known decomposition
    
    # Actually, let me try a different approach:
    # Use rank 27 for 3x3x3 (schoolbook) but try to merge with the extra columns
    # This won't help since they're independent.
    
    # Let me try to find a rank-19 decomposition for 3x3x3 using a known formula.
    # Here's one approach: use the "3x3 Strassen" from the literature.
    
    # For now, return schoolbook rank 45.
    # TODO: implement rank-19 for 3x3x3
    
    return {"U": U, "V": V, "W": W}
