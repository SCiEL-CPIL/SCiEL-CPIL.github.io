import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T
    # Shape: (n*m, m*p, p*n)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def tensor_to_matrices(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute sum of outer products
        R = U.shape[1]
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_approx += np.einsum('i,j,k->ijk', u, v, w)
        return T_approx

    def error(U, V, W):
        T_approx = tensor_to_matrices(U, V, W)
        return np.linalg.norm(T - T_approx)

    def is_close_to_half_integers(mat, tol=1e-4):
        # Check if all entries are close to x/2 for integer x
        for val in mat.flatten():
            nearest = np.round(val * 2) / 2
            if abs(val - nearest) > tol:
                return False
        return True

    def round_to_half_integers(mat):
        return np.round(mat * 2) / 2

    def verify(U, V, W):
        # Check if rounded matrices exactly reproduce T
        U_r = round_to_half_integers(U)
        V_r = round_to_half_integers(V)
        W_r = round_to_half_integers(W)
        T_approx = tensor_to_matrices(U_r, V_r, W_r)
        # Check exact match (with float tolerance for 0/1)
        if np.allclose(T_approx, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Try to find rank 10 decomposition
    best_result = None
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Random search for Rank 10
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        R = 10
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Simple gradient descent or use scipy
        # Since dimensions are small, we can use a simple local optimization
        # But gradient descent might be slow. Let's try a few random restarts with refinement.
        
        # Use scipy.optimize.minimize on flattened parameters
        from scipy.optimize import minimize
        
        def flatten_params(x):
            U_f = x[:n*m*R].reshape(n*m, R)
            V_f = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W_f = x[n*m*R+m*p*R:].reshape(p*n, R)
            return U_f, V_f, W_f

        def objective(x):
            U_f, V_f, W_f = flatten_params(x)
            return error(U_f, V_f, W_f)

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B or Nelder-Mead
        # L-BFGS-B is good for this
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success or res.fun < 1e-8:
            U_opt, V_opt, W_opt = flatten_params(res.x)
            if is_close_to_half_integers(U_opt) and is_close_to_half_integers(V_opt) and is_close_to_half_integers(W_opt):
                result = verify(U_opt, V_opt, W_opt)
                if result:
                    best_result = result
                    break

    if best_result:
        U, V, W = best_result
        # Convert to lists of floats/ints
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook rank 12
        rank = rank_schoolbook
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
