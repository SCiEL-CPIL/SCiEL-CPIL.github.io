import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before the 150s kill

    n, m, p = 2, 3, 3
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6

    # Construct the target tensor T
    # T[u, v, w] = 1 if indices match, else 0
    # u = i*m + j, v = j*p + k, w = k*n + i
    # Condition: j_u == j_v and k_v == k_w and i_w == i_u
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for u in range(dim_u):
        i_u = u // m
        j_u = u % m
        for v in range(dim_v):
            j_v = v // p
            k_v = v % p
            if j_u != j_v:
                continue
            for w in range(dim_w):
                k_w = w // n
                i_w = w % n
                if k_v == k_w and i_w == i_u:
                    T[u, v, w] = 1.0

    def tensor_from_factors(U, V, W):
        return np.einsum('uc,vc,wc->uvw', U, V, W)

    def objective(params, rank):
        if time.time() - start_time > time_limit:
            return float('inf')
        size_u = dim_u * rank
        size_v = dim_v * rank
        U = params[:size_u].reshape(dim_u, rank)
        V = params[size_u:size_u + size_v].reshape(dim_v, rank)
        W = params[size_u + size_v:].reshape(dim_w, rank)
        T_calc = tensor_from_factors(U, V, W)
        diff = T_calc - T
        return np.sum(diff * diff)

    def round_to_half(x):
        return np.round(2 * x) / 2.0

    def verify_solution(U, V, W):
        T_calc = tensor_from_factors(U, V, W)
        return np.allclose(T_calc, T, atol=1e-9)

    def try_rank(rank, max_restarts=50):
        if time.time() - start_time > time_limit - 15.0:
            return None
        
        # Use L-BFGS-B for better convergence on larger problems
        # We'll use a structured initialization based on the schoolbook method
        # and then perturb it slightly to allow the optimizer to find a lower rank solution
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            np.random.seed(r)
            
            # Initialize with schoolbook-like structure but with noise
            # This helps the optimizer start in a "good" basin
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            # For rank < 18, we can't just use schoolbook directly.
            # Instead, we use random initialization but with a bias towards
            # the structure of the problem.
            # A common heuristic is to start with small random values.
            
            U = np.random.uniform(-0.5, 0.5, size=(dim_u, rank))
            V = np.random.uniform(-0.5, 0.5, size=(dim_v, rank))
            W = np.random.uniform(-0.5, 0.5, size=(dim_w, rank))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                import scipy
                res = scipy.optimize.minimize(objective, params, args=(rank,), method='L-BFGS-B',
                                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
                
                if res.fun < 1e-6:
                    size_u = dim_u * rank
                    size_v = dim_v * rank
                    U_opt = res.x[:size_u].reshape(dim_u, rank)
                    V_opt = res.x[size_u:size_u + size_v].reshape(dim_v, rank)
                    W_opt = res.x[size_u + size_v:].reshape(dim_w, rank)
                    
                    # Round to half-integers
                    U_r = round_to_half(U_opt)
                    V_r = round_to_half(V_opt)
                    W_r = round_to_half(W_opt)
                    
                    if verify_solution(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # If rounding failed, try a local polish around the rounded values?
                    # Or accept if unrounded is very close? No, must be half-integers.
                    # Let's try a few more restarts.
                    
            except Exception as e:
                continue
                
        return None

    best_solution = None
    
    # Try ranks in increasing order
    # Theoretical lower bound for 2x3x3 is likely around 15 or 16.
    # Schoolbook is 18.
    for rank in range(15, 19):
        if time.time() - start_time > time_limit - 20.0:
            break
            
        sol = try_rank(rank, max_restarts=20)
        if sol is not None:
            best_solution = sol
            break
    
    if best_solution is None:
        # Fallback to schoolbook
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
        best_solution = {"U": U, "V": V, "W": W}

    return best_solution

import scipy
