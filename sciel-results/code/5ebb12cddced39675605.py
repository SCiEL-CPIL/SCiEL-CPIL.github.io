import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin

    n, m, p = 3, 4, 5
    N1 = n * m  # 12
    N2 = m * p  # 20
    N3 = p * n  # 15

    def remaining():
        return deadline - time.time()

    def schoolbook():
        rank = n * m * p  # 60
        U = np.zeros((N1, rank))
        V = np.zeros((N2, rank))
        W = np.zeros((N3, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-6):
        """Check if sum_c U[i*m+a,c] V[b*p+k,c] W[k*n+i,c] = delta_{a,b} for all i,a,b,k."""
        r = U.shape[1]
        # For each (i,k), check the m x m matrix: sum_c U[i*m+a,c] V[b*p+k,c] W[k*n+i,c] = delta_{a,b}
        for i in range(n):
            for k in range(p):
                # Build m x m matrix M[a,b] = sum_c U[i*m+a,c] * V[b*p+k,c] * W[k*n+i,c]
                M = np.zeros((m, m))
                for c in range(r):
                    u_col = U[i * m:(i + 1) * m, c]  # m values
                    v_col = V[:, c][np.arange(m) * p + k]  # m values: V[b*p+k, c] for b in 0..m-1
                    w_val = W[k * n + i, c]
                    M += np.outer(u_col, v_col) * w_val
                if not np.allclose(M, np.eye(m), atol=tol):
                    return False
        return True

    def compute_error(U, V, W):
        """Compute max |M - I| over all (i,k) slices."""
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for c in range(r):
                    u_col = U[i * m:(i + 1) * m, c]
                    v_col = V[np.arange(m) * p + k, c]
                    w_val = W[k * n + i, c]
                    M += np.outer(u_col, v_col) * w_val
                max_err = max(max_err, np.max(np.abs(M - np.eye(m))))
        return max_err

    def als_decompose(rank_R, n_trials=3, max_iter=500, lr_init=0.1):
        """Alternating least squares to find rank-R decomposition."""
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')

        for trial in range(n_trials):
            if remaining() < 5.0:
                break

            rng = np.random.RandomState(42 + trial * 7)
            # Initialize with structured random values
            U = rng.randn(N1, rank_R) * 0.5
            V = rng.randn(N2, rank_R) * 0.5
            W = rng.randn(N3, rank_R) * 0.5

            lr = lr_init
            for it in range(max_iter):
                if remaining() < 3.0:
                    break

                # Fix V and W, solve for U
                # For each (i,k): sum_c U[i*m+a,c] * V[b*p+k,c] * W[k*n+i,c] = delta_{a,b}
                # => For fixed (i,k): sum_c U[i*m+a,c] * (V[b*p+k,c] * W[k*n+i,c]) = delta_{a,b}
                # Let D_{b,c} = V[b*p+k,c] * W[k*n+i,c]
                # Then sum_c U[i*m+a,c] * D_{b,c} = delta_{a,b} for all b
                # This is: U[i*m:(i+1)*m, :] @ D.T = I_m
                # So U[i*m:(i+1)*m, :] = (D.T)^{-1} if D is square, or least squares

                for i in range(n):
                    for k in range(p):
                        # D is (m, rank_R): D[b,c] = V[b*p+k,c] * W[k*n+i,c]
                        D = V[np.arange(m) * p + k, :] * W[k * n + i, :]  # (m, rank_R)
                        # We want U_block @ D.T = I_m, where U_block is (m, rank_R)
                        # U_block = (D.T)^{-1} if possible, else lstsq
                        try:
                            U_block = np.linalg.solve(D.T, np.eye(m))
                        except np.linalg.LinAlgError:
                            U_block, _, _, _ = np.linalg.lstsq(D.T, np.eye(m), rcond=None)
                        U[i * m:(i + 1) * m, :] = U_block

                # Fix U and W, solve for V
                # For each (i,k): sum_c U[i*m+a,c] * V[b*p+k,c] * W[k*n+i,c] = delta_{a,b}
                # => sum_c V[b*p+k,c] * (U[i*m+a,c] * W[k*n+i,c]) = delta_{a,b}
                # Let E_{a,c} = U[i*m+a,c] * W[k*n+i,c]
                # Then E @ V[b*p+k, :].T = delta_{a,b} => V[b*p+k,:] = (E.T)^{-1} ... 
                # More precisely: for fixed b: sum_c E[a,c] * V[b*p+k,c] = delta_{a,b} for all a
                # => E @ V[b*p+k, :].T = e_b (b-th standard basis vector)
                # => V[b*p+k, :] = (E.T)^{-1} e_b = column b of (E.T)^{-1}

                for i in range(n):
                    for k in range(p):
                        E = U[i * m:(i + 1) * m, :] * W[k * n + i, :]  # (m, rank_R)
                        for b in range(m):
                            try:
                                V_val = np.linalg.solve(E.T, np.eye(m)[:, b])
                            except np.linalg.LinAlgError:
                                V_val, _, _, _ = np.linalg.lstsq(E.T, np.eye(m)[:, b], rcond=None)
                            V[b * p + k, :] = V_val

                # Fix U and V, solve for W
                # For each (i,k): sum_c U[i*m+a,c] * V[b*p+k,c] * W[k*n+i,c] = delta_{a,b}
                # => W[k*n+i,c] appears linearly. For fixed (i,k):
                # sum_c W[k*n+i,c] * (U[i*m+a,c] * V[b*p+k,c]) = delta_{a,b}
                # Let F_{(a,b),c} = U[i*m+a,c] * V[b*p+k,c]
                # We need: for all a,b: sum_c F[(a,b),c] * W[k*n+i,c] = delta_{a,b}
                # This is a system with m^2 equations and rank_R unknowns per (i,k).
                # Stack: F is (m*m, rank_R), target is vec(I_m) (length m*m)
                # F @ W[k*n+i, :].T = vec(I_m)

                for i in range(n):
                    for k in range(p):
                        # Build F: (m*m, rank_R)
                        F = np.zeros((m * m, rank_R))
                        for a in range(m):
                            for b in range(m):
                                F[a * m + b, :] = U[i * m + a, :] * V[b * p + k, :]
                        target = np.eye(m).flatten()  # length m*m
                        try:
                            W_val = np.linalg.solve(F.T, target)
                        except np.linalg.LinAlgError:
                            W_val, _, _, _ = np.linalg.lstsq(F.T, target, rcond=None)
                        W[k * n + i, :] = W_val

                # Compute error
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    break
                # Decay learning rate
                lr *= 0.99

            if best_err < 1e-10:
                break

        if best_err < 1e-10:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    def structured_search(rank_R):
        """Try structured decompositions for specific ranks."""
        if remaining() < 5.0:
            return None, None, None, float('inf')

        # Try the "identity block" structure:
        # For each a in 0..m-1, create rank terms that handle the a-th diagonal entry.
        # U[i*m+a, c] nonzero only for c associated with a
        # V[b*p+k, c] nonzero only for c associated with b
        # W[k*n+i, c] = 1 for all i,k when c is the "identity" term for a=b
        # This gives rank = m * (something)

        # Simple structured attempt: rank = m * n * p / something
        # Let's try rank = m * n = 12 or m * p = 20

        # Attempt: use m independent "channels"
        # For channel a: U[i*m+a, a] = 1, V[b*p+k, a] = 1 if b==a else 0, W[k*n+i, a] = 1
        # Then sum_c U[i*m+a',c] V[b*p+k,c] W[k*n+i,c] = U[i*m+a',a'] V[b*p+k,a'] W[k*n+i,a'] (only c=a' contributes)
        # = 1 * (1 if b==a' else 0) * 1 = delta_{a',b}
        # But we need delta_{a',b} where a' is the row index of U. So this works!
        # Rank = m = 4? Let's check:
        # U has shape (12, 4), V has shape (20, 4), W has shape (15, 4)
        # U[i*m+a, c] = 1 if c == a else 0
        # V[b*p+k, c] = 1 if c == b else 0
        # W[k*n+i, c] = 1 for all c
        # Then for fixed (i,k): M[a,b] = sum_c U[i*m+a,c] V[b*p+k,c] W[k*n+i,c]
        # = sum_c (1 if c==a) (1 if c==b) (1) = 1 if a==b else 0 = delta_{a,b} ✓

        U = np.zeros((N1, m))
        V = np.zeros((N2, m))
        W = np.ones((N3, m))
        for i in range(n):
            for a in range(m):
                U[i * m + a, a] = 1.0
        for b in range(m):
            for k in range(p):
                V[b * p + k, b] = 1.0

        if verify(U, V, W):
            return U, V, W, 0.0

        return None, None, None, float('inf')

    # First try the structured solution (rank = m = 4)
    U_struct, V_struct, W_struct, err_struct = structured_search(m)
    if U_struct is not None and err_struct < 1e-10:
        if verify(U_struct, V_struct, W_struct):
            return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}

    # Try ALS for various ranks
    U_best, V_best, W_best = schoolbook()
    r_best = n * m * p

    # Try ranks from lower bound upward
    for rank_R in range(m, r_best + 1):
        if remaining() < 10.0:
            break
        if rank_R >= r_best:
            break

        U_try, V_try, W_try, err = als_decompose(rank_R, n_trials=5, max_iter=300)
        if U_try is not None and err < 1e-10:
            if verify(U_try, V_try, W_try):
                U_best, V_best, W_best = U_try, V_try, W_try
                r_best = rank_R
                break
        elif err < 0.01 and rank_R < r_best:
            # Keep as candidate if much better than current
            pass

    # Final verification of best result
    if not verify(U_best, V_best, W_best, tol=1e-5):
        # Fallback to schoolbook
        U_best, V_best, W_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
