import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=int)
    V_fb = np.zeros((m*p, rank_fb), dtype=int)
    W_fb = np.zeros((p*n, rank_fb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb

    def loss_func(x, rank):
        # x is flattened U, V, W
        size_u = n*m
        size_v = m*p
        size_w = p*n
        u_end = size_u * rank
        v_end = u_end + size_v * rank
        w_end = v_end + size_w * rank
        
        U = x[:u_end].reshape(size_u, rank)
        V = x[u_end:v_end].reshape(size_v, rank)
        W = x[v_end:w_end].reshape(size_w, rank)
        
        # Compute T_decomp
        # T_decomp[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        diff = T - T_decomp
        return np.sum(diff * diff)

    def snap_and_check(U, V, W, rank):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T_decomp = np.einsum('at,bt,ct->abc', U_s, V_s, W_s, optimize=True)
        if np.allclose(T_decomp, T, atol=1e-9):
            # Convert to list of lists
            return U_s, V_s, W_s
        return None

    def polish(x, rank):
        """High-precision polish pass starting from a good initial point."""
        try:
            res = minimize(loss_func, x, args=(rank,), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            return res.x
        except Exception:
            return x

    # Try ranks from 24 to 35
    # We prioritize lower ranks
    for rank in range(24, 36):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        # Number of restarts
        # Lower ranks need more tries
        max_restarts = 30 if rank < 26 else 15
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Use values in {-1, -0.5, 0, 0.5, 1}
            # This helps the optimizer stay in the "right" basin
            init_vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m*rank + m*p*rank + p*n*rank))
            
            # Optimization
            try:
                res = minimize(loss_func, init_vals, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < 1e-4:
                    # If we get close, do a high-precision polish
                    polished_x = polish(res.x, rank)
                    
                    size_u = n*m
                    size_v = m*p
                    size_w = p*n
                    u_end = size_u * rank
                    v_end = u_end + size_v * rank
                    w_end = v_end + size_w * rank
                    
                    U_opt = polished_x[:u_end].reshape(size_u, rank)
                    V_opt = polished_x[u_end:v_end].reshape(size_v, rank)
                    W_opt = polished_x[v_end:w_end].reshape(size_w, rank)
                    
                    snapped = snap_and_check(U_opt, V_opt, W_opt, rank)
                    if snapped:
                        U_s, V_s, W_s = snapped
                        best_result = (U_s, V_s, W_s)
                        best_rank = rank
                        break
            except Exception:
                pass

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
