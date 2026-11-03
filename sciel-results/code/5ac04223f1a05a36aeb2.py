import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def check_valid(U, V, W, tol=1e-8):
        R = U.shape[1]
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        res = T_comp - T
        return np.max(np.abs(res)) < tol

    def snap_to_halfint(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_valid(U_s, V_s, W_s, tol=1e-9):
            return U_s, V_s, W_s, True
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_valid(U_i, V_i, W_i, tol=1e-9):
            return U_i, V_i, W_i, True
        return None, None, None, False

    def try_rank_optimization(R, max_time=10.0):
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        # Objective function
        def objective(params):
            # Unpack
            U = params[:nm*R].reshape(nm, R)
            V = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = params[nm*R+mp*R:].reshape(pn, R)
            
            T_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            res = T_comp - T
            return np.sum(res**2)
        
        best_solution = None
        best_obj = np.inf
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with small random integers
            # This helps find integer solutions
            U_init = rng.randint(-1, 2, (nm, R))
            V_init = rng.randint(-1, 2, (mp, R))
            W_init = rng.randint(-1, 2, (pn, R))
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                result = opt.minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_solution = result.x
            except Exception:
                pass
        
        if best_solution is not None and best_obj < 1e-6:
            U = best_solution[:nm*R].reshape(nm, R)
            V = best_solution[nm*R:nm*R+mp*R].reshape(mp, R)
            W = best_solution[nm*R+mp*R:].reshape(pn, R)
            U_s, V_s, W_s, valid = snap_to_halfint(U, V, W)
            if valid:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None

    # Try rank 23
    result_23 = try_rank_optimization(23, max_time=10.0)
    if result_23 is not None:
        return result_23

    # Fallback to rank 24
    R = n * m * p
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
