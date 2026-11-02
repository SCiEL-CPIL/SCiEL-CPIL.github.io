"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Build the target tensor T
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Function to check validity and snap to half-integers
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2

        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)

        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None

        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]

        return {"U": U_out, "V": V_out, "W": W_out}

    def als_decompose(U, V, W, max_iter=50, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            # M_U has shape (m*p*p*n, r)
            # M_U[b*(p*n) + c, r] = V[b, r] * W[c, r]
            M_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, r)
            
            for a in range(n * m):
                target_a = T_target[a].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, target_a, rcond=None)[0]
                except:
                    pass

            # Update V
            # M_V has shape (n*m*p*n, r)
            # M_V[a*(p*n) + c, r] = U[a, r] * W[c, r]
            M_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, r)
            
            for b in range(m * p):
                target_b = T_target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, target_b, rcond=None)[0]
                except:
                    pass

            # Update W
            # M_W has shape (n*m*m*p, r)
            # M_W[a*(m*p) + b, r] = U[a, r] * V[b, r]
            M_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, r)
            
            for c in range(p * n):
                target_c = T_target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, target_c, rcond=None)[0]
                except:
                    pass

            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T_target) ** 2)
            if err < tol:
                break

        return U, V, W, err

    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None

        np.random.seed(seed)

        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Random
        U1 = np.random.randn(n * m, r) * 0.5
        V1 = np.random.randn(m * p, r) * 0.5
        W1 = np.random.randn(p * n, r) * 0.5
        inits.append((U1, V1, W1))
        
        # Strategy 2: Based on structure - initialize with patterns
        U2 = np.zeros((n * m, r))
        V2 = np.zeros((m * p, r))
        W2 = np.zeros((p * n, r))
        for idx in range(min(r, n * m * m * p)):
            # Spread out the non-zeros
            U2[idx % (n*m), idx % r] = 1.0
            V2[idx % (m*p), idx % r] = 1.0
            W2[idx % (p*n), idx % r] = 1.0
        # Add some randomness
        U2 += np.random.randn(n * m, r) * 0.1
        V2 += np.random.randn(m * p, r) * 0.1
        W2 += np.random.randn(p * n, r) * 0.1
        inits.append((U2, V2, W2))
        
        # Strategy 3: Random with different scale
        U3 = np.random.randn(n * m, r)
        V3 = np.random.randn(m * p, r)
        W3 = np.random.randn(p * n, r)
        # Normalize
        U3 /= np.linalg.norm(U3) * np.sqrt(r)
        V3 /= np.linalg.norm(V3) * np.sqrt(r)
        W3 /= np.linalg.norm(W3) * np.sqrt(r)
        inits.append((U3, V3, W3))

        best_err = float('inf')
        best_U, best_V, best_W = None, None, None

        for U0, V0, W0 in inits:
            if time.time() > deadline - 5:
                break
            U_curr, V_curr, W_curr, err = als_decompose(U0.copy(), V0.copy(), W0.copy(), max_iter=200, tol=1e-12)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()

        if best_U is None:
            return None

        # Try to snap and check
        if best_err < 1e-6:
            result = check_and_snap(best_U.tolist(), best_V.tolist(), best_W.tolist(), r)
            if result:
                return result

        # If close but not exact, try a final polish with scipy
        if best_err < 1e-3:
            x0 = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
            size_U = (n * m) * r
            size_V = (m * p) * r

            def loss(params):
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U + size_V]
                W_flat = params[size_U + size_V:]
                U_p = U_flat.reshape((n * m), r)
                V_p = V_flat.reshape((m * p), r)
                W_p = W_flat.reshape((p * n), r)
                T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                err = np.sum((T_recon - T_target) ** 2)
                return err

            from scipy.optimize import minimize
            try:
                res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 500})
                if res.fun < 1e-8:
                    U_sol = res.x[:size_U].reshape((n * m), r)
                    V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
                    W_sol = res.x[size_U + size_V:].reshape((p * n), r)
                    result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
                    if result:
                        return result
            except:
                pass

        return None

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank
    # Based on known bounds, try ranks that might work
    ranks_to_try = [45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        seeds = [42, 123, 999, 111, 222, 333, 444, 555, 666, 777, 888, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 12345]
        for seed in seeds:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
