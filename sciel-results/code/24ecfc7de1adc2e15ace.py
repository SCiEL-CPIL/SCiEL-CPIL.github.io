import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 135

    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition (rank 36)
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

    best_result = (U_fb, V_fb, W_fb, rank_fb)
    best_rank = rank_fb

    def check_decomp(U, V, W, T, atol=1e-10):
        """Check if U,V,W decompose T."""
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=atol)

    def loss_func(x, rank):
        size_u = n*m
        size_v = m*p
        size_w = p*n
        u_end = size_u * rank
        v_end = u_end + size_v * rank
        w_end = v_end + size_w * rank
        
        U = x[:u_end].reshape(size_u, rank)
        V = x[u_end:v_end].reshape(size_v, rank)
        W = x[v_end:w_end].reshape(size_w, rank)
        
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        diff = T - T_decomp
        return np.sum(diff * diff)

    def snap_and_check(U, V, W):
        """Snap to nearest 0.5 and check if exact."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T_decomp = np.einsum('at,bt,ct->abc', U_s, V_s, W_s, optimize=True)
        if np.allclose(T_decomp, T, atol=1e-9):
            return U_s, V_s, W_s
        return None

    def try_optimization(rank, max_restarts=20):
        """Try to find rank-rank decomposition using optimization."""
        size_u = n*m
        size_v = m*p
        size_w = p*n
        total_size = size_u * rank + size_v * rank + size_w * rank
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization with structured values
            init_vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=total_size)
            
            try:
                res = minimize(loss_func, init_vals, args=(rank,), method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e-6:
                    # Polish with higher precision
                    res2 = minimize(loss_func, res.x, args=(rank,), method='L-BFGS-B',
                                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                    x = res2.x if res2.fun <= res.fun else res.x
                    
                    u_end = size_u * rank
                    v_end = u_end + size_v * rank
                    w_end = v_end + size_w * rank
                    
                    U_opt = x[:u_end].reshape(size_u, rank)
                    V_opt = x[u_end:v_end].reshape(size_v, rank)
                    W_opt = x[v_end:w_end].reshape(size_w, rank)
                    
                    snapped = snap_and_check(U_opt, V_opt, W_opt)
                    if snapped:
                        return snapped
            except Exception:
                pass
        
        return None

    # Phase 1: Try structured random search for quick wins
    for rank in range(24, 36):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue

        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break

        attempts = int(5000 * remaining / 100)
        attempts = max(100, min(attempts, 20000))

        for attempt in range(attempts):
            if time.time() > deadline - 15:
                break

            # Structured: combine schoolbook columns with random coefficients
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            for t in range(rank):
                n_terms = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                for _ in range(n_terms):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    sign = np.random.choice([-1, 1])
                    val = np.random.choice([0.5, 1.0], p=[0.3, 0.7])
                    U[i*m+j, t] += sign * val
                    V[j*p+k, t] += sign * val
                    W[k*n+i, t] += sign * val

            if check_decomp(U, V, W, T):
                best_result = (U, V, W, rank)
                best_rank = rank
                break

    # Phase 2: Try optimization for lower ranks
    for rank in range(20, 36):
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue

        remaining = deadline - 15 - time.time()
        if remaining < 10:
            break

        max_restarts = int(10 * remaining / 30)
        max_restarts = max(3, min(max_restarts, 30))

        result = try_optimization(rank, max_restarts)
        if result:
            U_opt, V_opt, W_opt = result
            best_result = (U_opt, V_opt, W_opt, rank)
            best_rank = rank

    U_best, V_best, W_best, _ = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
