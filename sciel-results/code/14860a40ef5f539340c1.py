<<<<<<< SEARCH
def _alternating_minimization(U, V, W, n, m, p, max_iter=2000, tol=1e-10, time_budget=None):
    """Alternating minimization to refine U, V, W with time budget."""
    rank = U.shape[1]
    start = time.time()
    
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
            
        # Fix V, W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(C.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    return None, None, None

        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return None, None, None

        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return None, None, None

        U, V, W = U_new, V_new, W_new
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
            
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None, None, None
=======
def _alternating_minimization(U, V, W, n, m, p, max_iter=2000, tol=1e-10, time_budget=None):
    """Alternating minimization to refine U, V, W with time budget."""
    rank = U.shape[1]
    start = time.time()
    
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
            
        # Fix V, W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    # Solve C * u^T = ones(p) for u (shape rank)
                    u, _, _, _ = np.linalg.lstsq(C, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    return None, None, None

        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    # Solve B * v^T = ones(n) for v (shape rank)
                    v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return None, None, None

        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    # Solve A * w^T = ones(m) for w (shape rank)
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return None, None, None

        U, V, W = U_new, V_new, W_new
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
            
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None, None, None
>>>>>>> REPLACE

