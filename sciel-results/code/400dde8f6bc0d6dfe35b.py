import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def _construct_rank_22():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = a * p + c
                T[i, j, k] = 1.0
                
    T_flat = T.reshape(n*m, m*p * p*n)
    T_V = T.transpose(0, 2, 1).reshape(n*m*p*n, m*p)
    T_W = T.reshape(n*m*m*p, p*n)

    def check_solution(U, V, W, tol=1e-9):
        T_approx = np.einsum('il,jl,kl->ijk', U, V, W)
        if not np.allclose(T, T_approx, atol=tol):
            return False
        for mat in [U, V, W]:
            for val in mat.flatten():
                if not np.isclose(val * 2, np.round(val * 2), atol=1e-7):
                    return False
        return True

    def polish_solution(U, V, W, r):
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_r = x[:n*m*r].reshape(n*m, r)
            V_r = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_r = x[n*m*r + m*p*r :].reshape(p*n, r)
            T_approx = np.einsum('il,jl,kl->ijk', U_r, V_r, W_r)
            # Add penalty for non-half-integer values to guide towards valid solutions
            penalty = 0
            for mat in [U_r, V_r, W_r]:
                diff = mat * 2 - np.round(mat * 2)
                penalty += np.sum(diff**2)
            return np.concatenate([(T_approx - T).flatten(), np.sqrt(penalty) * 10])
            
        try:
            result = opt.least_squares(objective, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            x_opt = result.x
            U_opt = x_opt[:n*m*r].reshape(n*m, r)
            V_opt = x_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = x_opt[n*m*r + m*p*r :].reshape(p*n, r)
            return U_opt, V_opt, W_opt
        except Exception:
            return U, V, W

    def solve_als(r, max_iters=30, tol=1e-12):
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iters):
            Z = np.einsum('jl,kl->jkl', V, W).reshape(m*p*p*n, r)
            try:
                U, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
                U = U.T
            except:
                return None

            Y = np.einsum('il,kl->ikl', U, W).reshape(n*m*p*n, r)
            try:
                V, _, _, _ = np.linalg.lstsq(Y.T, T_V.T, rcond=None)
                V = V.T
            except:
                return None

            X = np.einsum('il,jl->ijl', U, V).reshape(n*m*m*p, r)
            try:
                W, _, _, _ = np.linalg.lstsq(X.T, T_W.T, rcond=None)
                W = W.T
            except:
                return None

            err = np.linalg.norm(T - np.einsum('il,jl,kl->ijk', U, V, W))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            if time.time() > deadline:
                return None
                
        U_pol, V_pol, W_pol = polish_solution(U, V, W, r)
        
        if check_solution(U_pol, V_pol, W_pol):
            return U_pol, V_pol, W_pol
        
        def snap_half_int(A):
            return np.round(A * 2) / 2.0

        U_snap = snap_half_int(U_pol)
        V_snap = snap_half_int(V_pol)
        W_snap = snap_half_int(W_pol)
        
        if check_solution(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
            
        if check_solution(U, V, W):
            return U, V, W
            
        return None

    # Search for ranks 19, 20, 21
    for r in [19, 20, 21]:
        if time.time() > deadline:
            break
        for restart in range(50):
            if time.time() > deadline:
                break
            np.random.seed(restart + r * 1000)
            sol = solve_als(r)
            if sol is not None:
                U, V, W = sol
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    return _construct_rank_22()
