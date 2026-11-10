# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    # Deadline guard: evaluator kills at 630s; keep margin.
    t0 = time.time()
    deadline = t0 + 590.0

    N = 50

    # Compute Möbius function up to N
    mu = np.ones(N + 1, dtype=np.float64)
    is_prime = np.ones(N + 1, dtype=bool)
    for i in range(2, N + 1):
        if is_prime[i]:
            for j in range(i, N + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            for j in range(i2, N + 1, i2):
                mu[j] = 0.0

    sq_free = np.array([k for k in range(2, N + 1) if mu[k] != 0], dtype=np.int64)
    m = len(sq_free)

    try:
        from scipy.optimize import linprog
        have_scipy = True
    except ImportError:
        have_scipy = False

    c = np.array([np.log(k) / k for k in sq_free], dtype=np.float64)

    # Vectorized construction of A_ub
    x_vals = np.arange(1, 10 * N + 1, dtype=np.float64)
    x_col = x_vals[:, np.newaxis]  # (10N, 1)
    k_row = sq_free[np.newaxis, :].astype(np.float64)  # (1, m)
    A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10N, m)

    b_ub = np.ones(len(x_vals), dtype=np.float64)
    bounds = [(-10.0, 10.0)] * m

    best_g = None
    best_obj = -np.inf

    if have_scipy:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            best_g = result.x.copy()
            best_obj = -np.dot(best_g, c)

    # Coordinate descent polish with maintained M_vec
    if best_g is not None and time.time() < deadline - 30:
        g = best_g.copy()
        M_vec = A_ub @ g  # (10N,) - maintain this incrementally

        for iteration in range(100):
            if time.time() > deadline - 30:
                break
            improved = False
            for j in range(m):
                k_j = sq_free[j]
                col_j = A_ub[:, j]  # (10N,)
                
                # Try pushing to -10 (maximizes objective since log(k)/k > 0)
                for v in [-10.0, 10.0]:
                    if abs(v - g[j]) < 1e-12:
                        continue
                    delta = v - g[j]
                    new_M = M_vec + delta * col_j
                    if new_M.max() <= 1.0:
                        old_obj_contrib = -g[j] * np.log(k_j) / k_j
                        new_obj_contrib = -v * np.log(k_j) / k_j
                        if new_obj_contrib > old_obj_contrib + 1e-12:
                            g[j] = v
                            M_vec = new_M
                            improved = True
                            break
            
            if not improved:
                break
        
        obj_g = -np.dot(g, c)
        if obj_g > best_obj:
            best_g = g
            best_obj = obj_g

    # Final result
    if best_g is not None:
        g_vals = best_g
        f1 = -np.sum(g_vals / sq_free)
        domain = np.concatenate(([1], sq_free))
        values = np.concatenate(([f1], g_vals))
        
        # Verify
        K = int(domain.max())
        M_max = 0.0
        for x in range(1, 10 * K + 1):
            Mx = f1 * x + np.sum(g_vals * np.floor(x / sq_free))
            if Mx > M_max:
                M_max = Mx
        
        if M_max <= 1.0001:
            return domain.astype(np.int64), values.astype(np.float64)

    # Fallback: Use Möbius values directly
    mu_vals = np.array([mu[k] for k in sq_free], dtype=np.float64)
    f1 = -np.sum(mu_vals / sq_free)
    domain = np.concatenate(([1], sq_free))
    values = np.concatenate(([f1], mu_vals))
    return domain.astype(np.int64), values.astype(np.float64)

# EVOLVE-BLOCK-END
