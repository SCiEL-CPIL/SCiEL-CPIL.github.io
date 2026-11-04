def verify_fast(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and T[a,b,c] = 0 otherwise
    
    # Compute T using einsum or broadcasting
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is like a 3-way outer product sum
    
    # Method: T = np.einsum('ar,br,cr->abc', U, V, W)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if abs(T[a,b,c] - 1) > 1e-9:
                    return False, f"Expected 1 at ({a},{b},{c}), got {T[a,b,c]}"
    
    # Check all other entries are 0
    # Create a mask of valid entries
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid[a,b,c] = True
    
    # Check invalid entries
    invalid = ~valid
    if np.any(np.abs(T[invalid]) > 1e-9):
        max_val = np.max(np.abs(T[invalid]))
        return False, f"Invalid entry has max value {max_val}"
    
    return True, "OK"
