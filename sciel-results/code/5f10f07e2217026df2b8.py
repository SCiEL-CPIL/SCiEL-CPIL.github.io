"""Convex optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    n = N
    
    # Deterministic initialization: Random with fixed seed
    np.random.seed(42)
    h_init = np.random.rand(n)
    h_init = h_init - np.mean(h_init) + 0.5
    h_init = np.clip(h_init, 0.0, 1.0)
    diff = (n / 2.0 - np.sum(h_init)) / n
    h_init = np.clip(h_init + diff, 0.0, 1.0)
    
    lags_init = np.correlate(h_init, 1 - h_init, mode="full")
    t_init = np.max(lags_init) + 0.1
    
    x0 = np.concatenate([np.array([t_init]), h_init])
    
    bounds = [(0, None)] + [(0, 1)] * n
    
    def objective(x):
        return x[0]
        
    def objective_grad(x):
        grad = np.zeros(n + 1)
        grad[0] = 1.0
        return grad

    def sum_con(x):
        return np.sum(x[1:]) - n / 2.0
        
    def sum_con_jac(x):
        J = np.zeros((1, n + 1))
        J[0, 1:] = 1.0
        return J

    def corr_con(x):
        t = x[0]
        h = x[1:]
        lags = np.correlate(h, 1 - h, mode="full")
        return lags - t
        
    def corr_con_jac(x):
        J = np.zeros((2 * n - 1, n + 1))
        J[:, 0] = -1.0
        
        h = x[1:]
        for i in range(2 * n - 1):
            k = i - (n - 1)
            n_start = max(0, -k)
            n_end = min(n - 1, n - 1 - k)
            
            for m in range(n):
                deriv = 0.0
                n_val = m - k
                if n_start <= n_val <= n_end:
                    deriv += (1.0 - h[n_val])
                if n_start <= m <= n_end:
                    deriv -= h[m + k]
                J[i, 1 + m] = deriv
        
        return J

    constraints = [
        {'type': 'eq', 'fun': sum_con, 'jac': sum_con_jac},
        {'type': 'ineq', 'fun': corr_con, 'jac': corr_con_jac}
    ]
    
    try:
        result = minimize(
            objective,
            x0,
            method='trust-constr',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if result.success:
            h = result.x[1:]
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            if abs(current_sum - n / 2.0) > 1e-9 * n:
                diff = (n / 2.0 - current_sum) / n
                h = h + diff
                h = np.clip(h, 0.0, 1.0)
                
            if not np.all(np.isfinite(h)):
                return np.full(n, 0.5)
            if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
                return np.full(n, 0.5)
            if abs(np.sum(h) - n / 2.0) > 1e-9 * n:
                return np.full(n, 0.5)
                
            return h
        else:
            h = h_init
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            diff = (n / 2.0 - current_sum) / n
            h = h + diff
            h = np.clip(h, 0.0, 1.0)
            return h
    except Exception:
        return np.full(n, 0.5)
# EVOLVE-BLOCK-END
