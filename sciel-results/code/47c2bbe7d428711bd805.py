<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build base decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to simple schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing redundant columns
    time_remaining = 120 - (time.time() - start_time) - 10  # 10s safety margin
    if time_remaining > 5:
        U, V, W = _try_remove_columns(U, V, W, n, m, p, max_time=time_remaining)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Rebuild and don't try to reduce
        U, V, W = _build_base_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline (10s safety margin before 150s kill)
    
    # Build base decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to simple schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing redundant columns
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        U, V, W = _try_remove_columns(U, V, W, n, m, p, max_time=time_remaining)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Rebuild and don't try to reduce
        U, V, W = _build_base_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
