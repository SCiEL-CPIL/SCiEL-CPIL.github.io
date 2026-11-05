import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0

    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    # Flatten T for efficient operations
    # Shape: (n*m, m*p * p*n)
    T_flat = T.reshape(n*m, -1)

    def get_schoolbook():
        rank = n * m * p
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

    def verify(U, V, W):
        R = U.shape[1]
        # Compute sum_r U[:,r] * V[:,r] * W[:,r]
        # Using einsum for efficiency
        res = np.einsum('ar,br,cr->abc', U, V, W)
        # Check exactness with tight tolerance
        if not np.allclose(res, T, atol=1e-6):
            return False
        # Check if entries are close to half-integers
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        if not np.allclose(U, U_check, atol=1e-5):
            return False
        if not np.allclose(V, V_check, atol=1e-5):
            return False
        if not np.allclose(W, W_check, atol=1e-5):
            return False
        return True

    def als_decompose(R, max_iters=30, tol=1e-8):
        """
        Perform ALS decomposition of T into U, V, W of rank R.
        Returns (U, V, W, converged).
        """
        # Initialize randomly with entries in {-1, 0, 1} or similar
        # Using integers helps with convergence to half-integers
        U = np.random.randint(-1, 2, (n*m, R)).astype(float)
        V = np.random.randint(-1, 2, (m*p, R)).astype(float)
        W = np.random.randint(-1, 2, (p*n, R)).astype(float)
        
        # Add small noise to break symmetries
        U += np.random.randn(n*m, R) * 0.01
        V += np.random.randn(m*p, R) * 0.01
        W += np.random.randn(p*n, R) * 0.01

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False

            # Update U
            # Fix V, W.
            # Compute VV = V.T @ V (R x R)
            VV = V.T @ V
            WW = W.T @ W
            # Denominator D = VV * WW (element-wise product)
            D = VV * WW
            # Add small regularization for stability
            D += np.eye(R) * 1e-10
            
            # Numerator N: shape (n*m, R)
            # N[i, r] = sum_{j, k} T[i, j, k] V[j, r] W[k, r]
            # Let's compute this efficiently.
            # T is (n*m, m*p, p*n)
            # We can reshape V to (m, p, R) and W to (p, n, R)
            V_reshaped = V.reshape(m, p, R)
            W_reshaped = W.reshape(p, n, R)
            
            # For each i, j, k: T[i, j, k] * V[j, k, r] * W[k, i, r]
            # Wait, V index is j*p+k, so V_reshaped[j, k, r] is correct.
            # W index is k*n+i, so W_reshaped[k, i, r] is correct.
            
            # We want N[i, r] = sum_{j, k} T[i, j, k] V[j, k, r] W[k, i, r]
            # Let's do this in steps.
            # Temp[j, i, r] = sum_k T[i, j, k] W[k, i, r] ? No, T depends on i, j, k.
            
            # Alternative:
            # For each r, we are solving a least squares problem?
            # No, standard ALS for CP is:
            # U = (T x_2 V x_3 W) (V^T V \odot W^T W)^{-1}
            
            # Let's compute T x_2 V first.
            # (T x_2 V)_{i, k, r} = sum_j T[i, j, k] V[j, r]
            # T shape (n*m, m*p, p*n) -> indices (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
            # This is messy to do directly with standard einsum on the flat indices.
            
            # Let's use the flat representation.
            # T_flat is (n*m, m*p * p*n)
            # This is hard to align with V and W.
            
            # Let's go back to the 3D tensor T.
            # N[i, r] = sum_{j, k} T[i*m+j, j*p+k, k*n+i] V[j*p+k, r] W[k*n+i, r]
            
            # We can iterate over i, j, k.
            # N = np.zeros((n*m, R))
            # for i in range(n):
            #   for j in range(m):
            #     for k in range(p):
            #       if T[i*m+j, j*p+k, k*n+i] != 0:
            #         N[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
            
            # This is O(n*m*p*R) which is small (36*R).
            N = np.zeros((n*m, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        val = T[a, b, c]
                        if val != 0:
                            N[a, :] += val * (V[b, :] * W[c, :])
            
            # Solve U = N D^{-1}
            try:
                U_new = np.linalg.solve(D, N.T).T
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse or just keep old U
                U_new = N @ np.linalg.pinv(D)
            
            # Update V
            UU = U_new.T @ U_new
            WW = W.T @ W
            D = UU * WW
            D += np.eye(R) * 1e-10
            N = np.zeros((m*p, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        val = T[a, b, c]
                        if val != 0:
                            N[b, :] += val * (U_new[a, :] * W[c, :])
            try:
                V_new = np.linalg.solve(D, N.T).T
            except np.linalg.LinAlgError:
                V_new = N @ np.linalg.pinv(D)

            # Update W
            UU = U_new.T @ U_new
            VV = V_new.T @ V_new
            D = UU * VV
            D += np.eye(R) * 1e-10
            N = np.zeros((p*n, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        val = T[a, b, c]
                        if val != 0:
                            N[c, :] += val * (U_new[a, :] * V_new[b, :])
            try:
                W_new = np.linalg.solve(D, N.T).T
            except np.linalg.LinAlgError:
                W_new = N @ np.linalg.pinv(D)

            U, V, W = U_new, V_new, W_new
            
            # Compute error
            # err = || T - sum U V W ||
            # res = np.einsum('ar,br,cr->abc', U, V, W)
            # err = np.linalg.norm(T - res)
            # This is expensive. Let's approximate or do it every few iterations.
            if it % 5 == 0 or it == max_iters - 1:
                res = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - res)
                if abs(prev_err - err) < tol:
                    return U, V, W, True
                prev_err = err

        return U, V, W, False

    def polish(U, V, W):
        """
        Try to polish the solution to half-integers.
        """
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r, True
        
        # Try local search: flip signs or add/subtract 0.5 to random entries
        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        best_err = np.inf
        
        for _ in range(50):
            if time.time() > deadline:
                break
            # Pick a random matrix and a random entry
            mat = np.random.randint(0, 3)
            if mat == 0:
                idx = np.random.randint(n*m)
                col = np.random.randint(W_r.shape[1])
                delta = np.random.choice([-0.5, 0.5, 1.0, -1.0])
                cand = best_U.copy()
                cand[idx, col] += delta
                U_cand, V_cand, W_cand = cand, best_V, best_W
            elif mat == 1:
                idx = np.random.randint(m*p)
                col = np.random.randint(W_r.shape[1])
                delta = np.random.choice([-0.5, 0.5, 1.0, -1.0])
                cand = best_V.copy()
                cand[idx, col] += delta
                U_cand, V_cand, W_cand = best_U, cand, best_W
            else:
                idx = np.random.randint(p*n)
                col = np.random.randint(W_r.shape[1])
                delta = np.random.choice([-0.5, 0.5, 1.0, -1.0])
                cand = best_W.copy()
                cand[idx, col] += delta
                U_cand, V_cand, W_cand = best_U, best_V, cand
            
            res = np.einsum('ar,br,cr->abc', U_cand, V_cand, W_cand)
            err = np.linalg.norm(T - res)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_cand, V_cand, W_cand
                
                # If it's a perfect match, return it
                if err < 1e-6:
                    # Check half-integer constraint
                    if np.allclose(best_U * 2, np.round(best_U * 2)) and \
                       np.allclose(best_V * 2, np.round(best_V * 2)) and \
                       np.allclose(best_W * 2, np.round(best_W * 2)):
                        return np.round(best_U * 2)/2, np.round(best_V * 2)/2, np.round(best_W * 2)/2, True

        if best_err < 1e-6:
            if np.allclose(best_U * 2, np.round(best_U * 2)) and \
               np.allclose(best_V * 2, np.round(best_V * 2)) and \
               np.allclose(best_W * 2, np.round(best_W * 2)):
                return np.round(best_U * 2)/2, np.round(best_V * 2)/2, np.round(best_W * 2)/2, True
                
        return None, None, None, False

    # Search for lower ranks
    # Start from a known high rank and go down
    # Schoolbook is 36.
    # Try ranks from 20 to 35.
    for R in range(20, 36):
        if time.time() > deadline - 5:
            break
            
        # Number of attempts depends on rank. Lower ranks are harder.
        # But we have limited time.
        attempts = 100 if R > 30 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            U, V, W, converged = als_decompose(R, max_iters=20)
            
            # Try to polish
            U_pol, V_pol, W_pol, success = polish(U, V, W)
            if success:
                return {
                    "U": U_pol.tolist(),
                    "V": V_pol.tolist(),
                    "W": W_pol.tolist()
                }
        
        # If we found a converged solution but couldn't polish to exact half-integers,
        # we might still try to return it if it's very close, but the task requires exactness.
        # So we continue searching.

    # Fallback to schoolbook
    return get_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
