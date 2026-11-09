"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 150s limit, use 140s as hard stop with 10s margin for safety
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Target tensor T
    # T[u, v, w] = 1 if u%3 == v//3 and v%3 == w//3 and w%3 == u%3
    target_flat = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_flat[u, v, w] = 1.0
    target_flat_list = target_flat.flatten()
    
    # Precompute for fast tensor computation
    def compute_tensor(U, V, W):
        """Fast vectorized tensor computation: T = sum_r U_r * V_r * W_r"""
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify_and_round(U, V, W):
        """Try multiple rounding strategies and verify"""
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        err = np.dot(diff, diff)
        
        # If the continuous solution is not good, rounding won't help
        if err > 1e-4:
            return None, None, None, False
        
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Verify rounded
        T_rounded = compute_tensor(U_r, V_r, W_r)
        diff_r = T_rounded.flatten() - target_flat_list
        err_r = np.dot(diff_r, diff_r)
        
        if err_r < 1e-6:
            return U_r.tolist(), V_r.tolist(), W_r.tolist(), True
        
        # Try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        T_int = compute_tensor(U_i, V_i, W_i)
        diff_i = T_int.flatten() - target_flat_list
        err_i = np.dot(diff_i, diff_i)
        
        if err_i < 1e-6:
            return U_i.tolist(), V_i.tolist(), W_i.tolist(), True
        
        return None, None, None, False

    # Schoolbook solution (Rank 27)
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
                
    best_result = {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

    # Try to find lower ranks using Adam Optimizer with Penalty
    # Ranks to try: 23, 24, 25, 26
    # Time allocation: prioritize lower ranks
    
    def adam_step(params, grads, m_v, v_v, t, lr, beta1, beta2, eps):
        m_v = beta1 * m_v + (1 - beta1) * grads
        v_v = beta2 * v_v + (1 - beta2) * (grads ** 2)
        m_hat = m_v / (1 - beta1 ** t)
        v_hat = v_v / (1 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return params, m_v, v_v

    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
            
        # Budget per rank
        budget = 30.0 if rank == 23 else 15.0
        rank_start = time.time()
        
        # Multiple random starts
        for trial in range(10):
            if time.time() - rank_start > budget:
                break
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 100 + rank * 7)
            
            # Initialize
            init_U = np.random.normal(0, 1.0, (n * m, rank))
            init_V = np.random.normal(0, 1.0, (m * p, rank))
            init_W = np.random.normal(0, 1.0, (p * n, rank))
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            # Adam parameters
            m_v = np.zeros_like(params)
            v_v = np.zeros_like(params)
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            t = 0
            max_iter = 2000
            
            # Penalty for non-integer/half-integer
            # We want values to be close to k/2
            def penalty(x):
                # Distance to nearest half-integer
                return np.sum((x * 2 - np.round(x * 2)) ** 2)
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                U = params[:rank * n * m].reshape((n * m, rank))
                V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
                
                T_calc = compute_tensor(U, V, W)
                diff = T_calc.flatten() - target_flat_list
                err = np.dot(diff, diff)
                
                # Early stop if error is very small
                if err < 1e-8:
                    break
                
                # Gradient of L2 error
                diff_3d = diff.reshape((n * m, m * p, p * n))
                grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff_3d, V, W)
                grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff_3d, U, W)
                grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff_3d, U, V)
                
                # Gradient of penalty
                # d/dx (2x - round(2x))^2 is 0 almost everywhere, but we can use a soft approximation
                # Or just add a small push towards nearest half-integer if err is small
                pen_weight = 10.0 if err < 1.0 else 0.0
                if pen_weight > 0:
                    # Gradient of distance to nearest half-integer
                    # f(x) = (x - round(x)/2)^2 ? No, we want x to be k/2.
                    # Let g(x) = x - 0.5 * round(2x).
                    # grad_g is 1 almost everywhere, but jumps.
                    # We can just add a small force towards the nearest half-integer
                    target_U = np.round(U * 2) / 2.0
                    target_V = np.round(V * 2) / 2.0
                    target_W = np.round(W * 2) / 2.0
                    grad_U += pen_weight * (U - target_U)
                    grad_V += pen_weight * (V - target_V)
                    grad_W += pen_weight * (W - target_W)
                
                grads = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                t += 1
                params, m_v, v_v = adam_step(params, grads, m_v, v_v, t, lr, beta1, beta2, eps)
                
                # Adaptive learning rate
                if it % 100 == 0:
                    lr *= 0.95
            
            # Check result
            U_final = params[:rank * n * m].reshape((n * m, rank))
            V_final = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
            W_final = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
            
            U_r, V_r, W_r, ok = verify_and_round(U_final, V_final, W_final)
            if ok:
                best_result = {
                    "U": U_r,
                    "V": V_r,
                    "W": W_r
                }
                return best_result

    return best_result
# EVOLVE-BLOCK-END
