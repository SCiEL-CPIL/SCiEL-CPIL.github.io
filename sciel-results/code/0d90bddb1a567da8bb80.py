import numpy as np
from math import factorial

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    dt_total = t1 - t0
    
    if dt_total == 0:
        return y0_arr

    # Reconstruct A and b
    y_zero = np.zeros_like(y0_arr)
    b = f(t0, y_zero)
    
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros_like(y0_arr)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Verify
    pred_t0 = A @ y0_arr + b
    actual_t0 = f(t0, y0_arr)
    err_t0 = np.linalg.norm(pred_t0 - actual_t0)
    actual_t1 = f(t1, y0_arr)
    err_t1 = np.linalg.norm(pred_t0 - actual_t1)
    scale = max(1.0, np.linalg.norm(actual_t0))
    
    if err_t0 < 1e-8 * scale and err_t1 < 1e-8 * scale:
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = A
        M[:n, n] = b
        
        X = M * dt_total
        norm_X = np.linalg.norm(X, np.inf)
        
        if norm_X == 0:
            z1 = np.zeros(n + 1)
            z1[:n] = y0_arr
            z1[n] = 1.0
            return z1[:n]
            
        theta = 0.935
        s = int(np.ceil(np.log2(norm_X / theta)))
        if s < 0: s = 0
        
        X_s = X / (2 ** s)
        
        m = 13
        coeffs = np.zeros(m + 1)
        for k in range(m + 1):
            num = factorial(2*m - k) * factorial(m)
            den = factorial(2*m) * factorial(k) * factorial(m - k)
            coeffs[k] = num / den
            
        # We need to solve Q(X_s) u = P(X_s) v
        # Q(X_s) = sum c_k (-1)^k X_s^k
        # P(X_s) = sum c_k X_s^k
        
        # Precompute powers of X_s? No, we apply to vector.
        # But we need to factor Q(X_s) once.
        # So we must form Q(X_s) as a matrix.
        
        I = np.eye(n + 1)
        Q_mat = np.zeros((n + 1, n + 1))
        X_power = I.copy()
        for k in range(m + 1):
            Q_mat += coeffs[k] * ((-1) ** k) * X_power
            if k < m:
                X_power = X_power @ X_s
                
        # Factor Q_mat
        # Use LU decomposition
        from scipy.linalg import lu, solve
        try:
            lu_piv, lu_data = lu(Q_mat)
        except:
            from scipy.linalg import expm
            z0 = np.zeros(n + 1)
            z0[:n] = y0_arr
            z0[n] = 1.0
            z1 = expm(X) @ z0
            return z1[:n]
            
        # Function to apply R_13(X_s) to a vector v
        def apply_pade(v):
            # Compute r = P(X_s) v
            # r = sum c_k X_s^k v
            r = np.zeros_like(v)
            v_power = v.copy()
            for k in range(m + 1):
                r += coeffs[k] * v_power
                if k < m:
                    v_power = X_s @ v_power
                    
            # Solve Q_mat u = r
            u = solve(lu_data, r, piv=lu_piv)
            return u
            
        z0 = np.zeros(n + 1)
        z0[:n] = y0_arr
        z0[n] = 1.0
        
        v = z0
        for _ in range(2 ** s):
            v = apply_pade(v)
            
        return v[:n]
    else:
        steps = 10000
        h = dt_total / steps
        t = t0
        y = y0_arr.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y
