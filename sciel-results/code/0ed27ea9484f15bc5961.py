"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()

    def tensor_to_matrix(U, V, W):
        """Compute the reconstructed tensor from factor matrices."""
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
        return T_rec

    def check_exact(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_rec = tensor_to_matrix(U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def try_round_and_check(U, V, W, scale=2):
        """Try to round U, V, W to nearest 1/scale and check."""
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        T_rec = tensor_to_matrix(U_r, V_r, W_r)
        if np.allclose(T_rec, T, atol=1e-10):
            return U_r, V_r, W_r
        return None

    def als_step(U, V, W, T_flat, dim_U, dim_V, dim_W, r):
        """One ALS iteration: update U, V, W in place."""
        # Update U: for each row b of U
        for b in range(dim_U):
            # X has shape (dim_W * dim_V, r)
            X = np.zeros((dim_W * dim_V, r))
            for rr in range(r):
                X[:, rr] = np.kron(W[:, rr], V[:, rr])
            y = T[:, b, :].flatten()
            try:
                U[b, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                pass
        
        # Update V: for each row c of V
        for c in range(dim_V):
            X = np.zeros((dim_W * dim_U, r))
            for rr in range(r):
                X[:, rr] = np.kron(W[:, rr], U[:, rr])
            y = T[:, :, c].flatten()
            try:
                V[c, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                pass
        
        # Update W: for each row a of W
        for a in range(dim_W):
            X = np.zeros((dim_U * dim_V, r))
            for rr in range(r):
                X[:, rr] = np.kron(U[:, rr], V[:, rr])
            y = T[a, :, :].flatten()
            try:
                W[a, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                pass

    def polish(U, V, W, T, max_iters=50, tol=1e-12):
        """High-precision polish using more iterations."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            als_step(U, V, W, T.flatten(), dim_U, dim_V, dim_W, U.shape[1])
            T_rec = tensor_to_matrix(U, V, W)
            error = np.linalg.norm(T - T_rec)
            if error < tol:
                break
        return U, V, W

    best_result = None
    best_rank = 24
    
    # Try ranks from 19 to 23
    for r in range(19, 24):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 8
        max_iters = 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Better initialization: mix random and structured
            U = np.random.randn(dim_U, r) * 0.5
            V = np.random.randn(dim_V, r) * 0.5
            W = np.random.randn(dim_W, r) * 0.5
            
            # Add some structure: initialize with partial schoolbook-like patterns
            if restart == 0:
                # First restart: use a structured initialization
                for rr in range(r):
                    i = rr % n
                    j = (rr // n) % m
                    k = (rr // (n * m)) % p
                    U[i * m + j, rr] += 1.0
                    V[j * p + k, rr] += 1.0
                    W[k * n + i, rr] += 1.0
            
            # Normalize
            U /= max(np.linalg.norm(U), 1e-10)
            V /= max(np.linalg.norm(V), 1e-10)
            W /= max(np.linalg.norm(W), 1e-10)
            
            # Run ALS
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                als_step(U, V, W, T_flat, dim_U, dim_V, dim_W, r)
                
                # Check convergence
                T_rec = tensor_to_matrix(U, V, W)
                error = np.linalg.norm(T - T_rec)
                if error < 1e-8:
                    # Polish with higher precision
                    U, V, W = polish(U, V, W, T)
                    T_rec = tensor_to_matrix(U, V, W)
                    error = np.linalg.norm(T - T_rec)
                    if error < 1e-10:
                        # Try rounding
                        rounded = try_round_and_check(U, V, W, scale=2)
                        if rounded is not None:
                            U_r, V_r, W_r = rounded
                            best_result = (U_r, V_r, W_r, r)
                            found = True
                            break
                        else:
                            rounded_int = try_round_and_check(U, V, W, scale=1)
                            if rounded_int is not None:
                                U_r, V_r, W_r = rounded_int
                                best_result = (U_r, V_r, W_r, r)
                                found = True
                                break
                    break
            if found:
                break

        if found:
            break

    if best_result is not None:
        U_r, V_r, W_r, r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
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
