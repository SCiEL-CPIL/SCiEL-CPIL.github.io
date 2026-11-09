import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB-style function f5 (Linear slope).
    Uses central finite differences to estimate the gradient at the origin,
    identifies the optimal box vertex for the linear approximation, and then
    uses a derivative-free local search (Nelder-Mead) to refine the solution,
    accounting for potential non-linearities (e.g., quadratic terms) in the function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        d = self.dim
        calls_made = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
            except Exception:
                return np.inf
            calls_made += 1
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x, dtype=np.float64)
            return val

        # Phase 1: Recover gradient using central differences
        # Evaluate at origin
        x0 = np.zeros(d)
        f0 = safe_call(x0)
        if not np.isfinite(f0):
            return best_x, best_f

        # Probe along each axis with step h=1.0 (central difference)
        h = 1.0
        v = np.zeros(d)
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            xi_pos = np.zeros(d)
            xi_pos[i] = h
            fi_pos = safe_call(xi_pos)
            
            if calls_made >= self.budget or time.time() > deadline:
                # Fallback to forward difference if budget runs out
                if np.isfinite(fi_pos):
                    v[i] = (fi_pos - f0) / h
                break
                
            xi_neg = np.zeros(d)
            xi_neg[i] = -h
            fi_neg = safe_call(xi_neg)
            
            if np.isfinite(fi_pos) and np.isfinite(fi_neg):
                v[i] = (fi_pos - fi_neg) / (2 * h)
            elif np.isfinite(fi_pos):
                v[i] = (fi_pos - f0) / h
            elif np.isfinite(fi_neg):
                v[i] = (f0 - fi_neg) / h

        # Phase 2: Compute the optimal box vertex for the linear approximation
        # For linear f(x) = c + v·x on [-5,5]^d, min is at:
        # x_i = -5 if v_i > 0, x_i = 5 if v_i < 0
        opt_x = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                opt_x[i] = -5.0
            elif v[i] < 0:
                opt_x[i] = 5.0
            else:
                opt_x[i] = 0.0

        # Evaluate the candidate vertex
        f_opt = safe_call(opt_x)
        if not np.isfinite(f_opt):
            return best_x, best_f

        # Phase 3: Refine using Nelder-Mead if budget remains
        # This handles non-linearities (e.g., quadratic terms) that make the true minimum
        # slightly different from the affine vertex.
        remaining_budget = self.budget - calls_made
        if remaining_budget > 20 and time.time() < deadline:
            # Define a wrapper to count calls for scipy
            scipy_calls = [0]
            def scipy_func(x):
                if scipy_calls[0] >= remaining_budget:
                    return np.inf
                if time.time() > deadline:
                    return np.inf
                val = safe_call(x)
                scipy_calls[0] += 1
                return val

            # Use Nelder-Mead as it is robust for non-smooth/non-linear functions
            # and does not require gradient information.
            try:
                res = minimize(
                    scipy_func, 
                    opt_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': remaining_budget, 'maxfev': remaining_budget, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res.success or res.fun < best_f:
                    # safe_call already updated best_x/best_f if res.fun was better
                    pass
            except Exception:
                pass

        # Phase 4: Stall on best point to consume remaining budget
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            try:
                func(best_x)
            except Exception:
                break
            calls_made += 1

        return best_x, best_f
