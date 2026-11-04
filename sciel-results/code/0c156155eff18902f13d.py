# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 100  # leave safety margin before 120s kill

    def verify_exact(U, V, W, n, m, p, rank):
        """
        Check if the decomposition is exact.
        Assumes U, V, W contain integers or half-integers (multiples of 0.5).
        We multiply by 2 to work with integers to avoid float issues.
        """
        # Convert to integers by multiplying by 2
        U_int = np.array(U) * 2
        V_int = np.array(V) * 2
        W_int = np.array(W) * 2
        
        # Check if they are close to integers
        if not np.allclose(U_int, np.round(U_int), atol=1e-6): return False
        if not np.allclose(V_int, np.round(V_int), atol=1e-6): return False
        if not np.allclose(W_int, np.round(W_int), atol=1e-6): return False
        
        U_int = np.round(U_int).astype(int)
        V_int = np.round(V_int).astype(int)
        W_int = np.round(W_int).astype(int)
        
        # The sum should be 1.0.
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # In integer domain (multiplied by 2):
        # sum_t (2U) * (2V) * (2W) = 8
        target = 8
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U_int[i*m+j, t] * V_int[j*p+k, t] * W_int[k*n+i, t]
                    if s != target:
                        return False
        return True

    def round_and_verify(U, V, W, n, m, p, rank):
        """
        Try rounding U, V, W to nearest half-integers and verify.
        """
        # Round to nearest 0.5
        U_r = np.round(np.array(U) * 2) / 2
        V_r = np.round(np.array(V) * 2) / 2
        W_r = np.round(np.array(W) * 2) / 2
        
        if verify_exact(U_r, V_r, W_r, n, m, p, rank):
            return U_r, V_r, W_r
        return None

    def try_als(rank, max_iter=300, tol=1e-9, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            # Check convergence
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                # Try to round and verify
                res = round_and_verify(U, V, W, n, m, p, rank)
                if res:
                    return res
        
        # Final check even if not fully converged
        res = round_and_verify(U, V, W, n, m, p, rank)
        if res:
            return res
        
        return None

    # Try ranks from 14 up to 24
    # Lower ranks are harder to find, so we give them more seeds.
    for rank in range(14, 25):
        if time.time() > deadline:
            break
        
        # More seeds for lower ranks
        num_seeds = 10 if rank <= 16 else 5
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            result = try_als(rank, max_iter=300, tol=1e-9, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook rank 24
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
