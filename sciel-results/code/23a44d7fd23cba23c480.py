import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def _compute_c4_bound_fast(roots):
        """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
        roots = np.asarray(roots, dtype=np.float64)
        if np.any(roots <= 0) or np.any(np.isnan(roots)):
            return np.inf
        
        n_quad = 150
        x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
        
        # Vectorized: compute log_P for all roots at once
        diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
        log_abs_diff = np.log(np.abs(diff) + 1e-300)
        log_P = 2.0 * np.sum(log_abs_diff, axis=1)
        
        P = np.exp(log_P)
        P2 = P * P
        
        int_P = np.sum(w_quad * P)
        int_P2 = np.sum(w_quad * P2)
        
        if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
            return np.inf
        
        c4 = int_P2 / (int_P * int_P)
        return c4
    
    def _solve_dyson_roots(params, x0):
        """Solve for roots given potential parameters using Newton-Raphson."""
        a1, a2, a3, b = params
        
        def V_prime(x):
            """Derivative of V(x) = (a1*x^2 + a2*x + a3)/(x+b)"""
            num = a1 * x**2 + a2 * x + a3
            den = x + b
            dnum = 2 * a1 * x + a2
            return (dnum * den - num) / (den**2)
        
        def equations(roots):
            """Dyson gas equilibrium equations."""
            n = len(roots)
            eqs = np.zeros(n)
            for i in range(n):
                s = 0.0
                for j in range(n):
                    if j != i:
                        s += 2.0 / (roots[i] - roots[j])
                eqs[i] = s + V_prime(roots[i])
            return eqs
        
        def jacobian(roots):
            """Jacobian of the Dyson gas equations."""
            n = len(roots)
            J = np.zeros((n, n))
            for i in range(n):
                # Diagonal: sum of 2/(r_i - r_j)^2 + V''(r_i)
                J[i, i] = V_prime(roots[i]) * 0  # Placeholder, need V''
                s = 0.0
                for j in range(n):
                    if j != i:
                        s += 2.0 / (roots[i] - roots[j])**2
                J[i, i] = s
                # Off-diagonal: -2/(r_i - r_j)^2
                for j in range(n):
                    if j != i:
                        J[i, j] = -2.0 / (roots[i] - roots[j])**2
            return J
        
        # Newton-Raphson iteration
        roots = x0.copy()
        for it in range(100):
            if time.time() > deadline:
                break
            f = equations(roots)
            if np.max(np.abs(f)) < 1e-12:
                break
            J = jacobian(roots)
            try:
                delta = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break
            # Damped step
            lam = 1.0
            for _ in range(10):
                new_roots = roots + lam * delta
                if np.all(new_roots > 0):
                    roots = new_roots
                    break
                lam *= 0.5
            else:
                break
        
        return roots
    
    def _objective(params):
        """Objective: C4 bound for roots solved from potential parameters."""
        if time.time() > deadline:
            return 1e10
        
        a1, a2, a3, b = params
        
        # Initial guess for roots: spread out
        x0 = np.linspace(1.0, 20.0, 12)
        
        try:
            roots = _solve_dyson_roots(params, x0)
            if np.any(roots <= 0) or not np.all(np.isfinite(roots)):
                return 1e10
            c4 = _compute_c4_bound_fast(roots)
            return c4
        except Exception:
            return 1e10
    
    # Multistart optimization over potential parameters
    best_val = 1e10
    best_params = None
    best_roots = None
    
    np.random.seed(42)
    n_starts = 30
    
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if s == 0:
            # Start near reasonable values
            a1 = 0.1 + np.random.randn() * 0.05
            a2 = 0.0 + np.random.randn() * 0.1
            a3 = 0.5 + np.random.randn() * 0.2
            b = 1.0 + np.random.randn() * 0.2
        else:
            a1 = np.random.uniform(0.01, 1.0)
            a2 = np.random.uniform(-1.0, 1.0)
            a3 = np.random.uniform(0.1, 5.0)
            b = np.random.uniform(0.1, 5.0)
        
        x0 = np.array([a1, a2, a3, b])
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
                # Solve for roots with best params
                x0_roots = np.linspace(1.0, 20.0, 12)
                best_roots = _solve_dyson_roots(best_params, x0_roots)
        except Exception:
            pass
    
    # Final polish: if we have good roots, do a high-precision local optimization
    if best_roots is not None and np.all(best_roots > 0):
        # Polish with direct root optimization
        def _root_objective(roots):
            roots = np.sort(roots)
            if np.any(roots <= 0) or not np.all(np.isfinite(roots)):
                return 1e10
            return _compute_c4_bound_fast(roots)
        
        try:
            result = minimize(_root_objective, best_roots, method='L-BFGS-B',
                             options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_roots = result.x
        except Exception:
            pass
    
    if best_roots is not None and np.all(best_roots > 0):
        return [float(r) for r in np.sort(best_roots)]
    else:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
