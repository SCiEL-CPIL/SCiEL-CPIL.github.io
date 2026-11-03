# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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

    def check_solution(U, V, W):
        """Check if U, V, W exactly decompose T."""
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
        return np.allclose(T_rec, T, atol=1e-10)

    def try_round_and_check(U, V, W, scale=2):
        """Try to round U, V, W to nearest 1/scale and check."""
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U_r.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W_r[:, rr], U_r[:, rr], V_r[:, rr])
        
        if np.allclose(T_rec, T, atol=1e-10):
            return U_r, V_r, W_r
        return None

    def regularized_als(r, mu, max_iters=100, tol=1e-10):
        """
        ALS with L2 regularization towards half-integers.
        """
        # Initialize
        U = np.random.randn(dim_U, r)
        V = np.random.randn(dim_V, r)
        W = np.random.randn(dim_W, r)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for b in range(dim_U):
                y = T[:, b, :].flatten()
                X = np.zeros((dim_W * dim_V, r))
                for rr in range(r):
                    X[:, rr] = np.kron(W[:, rr], V[:, rr])
                
                C = np.round(2 * U[b, :]) / 2.0
                
                A = X.T @ X + 2 * mu * np.eye(r)
                b_vec = X.T @ y + 2 * mu * C
                
                try:
                    u_new = np.linalg.solve(A, b_vec)
                    U[b, :] = u_new
                except np.linalg.LinAlgError:
                    try:
                        u_new, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                        U[b, :] = u_new
                    except:
                        pass

            # Update V
            for c in range(dim_V):
                y = T[:, :, c].flatten()
                X = np.zeros((dim_W * dim_U, r))
                for rr in range(r):
                    X[:, rr] = np.kron(W[:, rr], U[:, rr])
                
                C = np.round(2 * V[c, :]) / 2.0
                
                A = X.T @ X + 2 * mu * np.eye(r)
                b_vec = X.T @ y + 2 * mu * C
                
                try:
                    v_new = np.linalg.solve(A, b_vec)
                    V[c, :] = v_new
                except np.linalg.LinAlgError:
                    try:
                        v_new, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                        V[c, :] = v_new
                    except:
                        pass

            # Update W
            for a in range(dim_W):
                y = T[a, :, :].flatten()
                X = np.zeros((dim_U * dim_V, r))
                for rr in range(r):
                    X[:, rr] = np.kron(U[:, rr], V[:, rr])
                
                C = np.round(2 * W[a, :]) / 2.0
                
                A = X.T @ X + 2 * mu * np.eye(r)
                b_vec = X.T @ y + 2 * mu * C
                
                try:
                    w_new = np.linalg.solve(A, b_vec)
                    W[a, :] = w_new
                except np.linalg.LinAlgError:
                    try:
                        w_new, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                        W[a, :] = w_new
                    except:
                        pass

            # Check convergence
            T_rec = np.zeros((dim_W, dim_U, dim_V))
            for rr in range(r):
                T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
            
            error = np.linalg.norm(T - T_rec)
            if error < tol:
                rounded = try_round_and_check(U, V, W, scale=2)
                if rounded is not None:
                    return rounded
                rounded_int = try_round_and_check(U, V, W, scale=1)
                if rounded_int is not None:
                    return rounded_int
                return None

        return None

    best_result = None
    best_rank = 24

    # Try ranks from 19 to 23
    mus = [0.1, 1.0, 10.0]
    
    for r in range(19, 24):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(15):
            if time.time() > deadline - 10:
                break
                
            for mu in mus:
                if time.time() > deadline - 10:
                    break
                    
                result = regularized_als(r, mu, max_iters=80)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_result = (U_r, V_r, W_r, r)
                    found = True
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
