import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    
    # Precompute moments: int_0^inf x^4 * H_n(x) * exp(-x^2/2) dx for n = 0, 4, 8, 12
    def compute_moment(n):
        def integrand(x):
            return x**4 * eval_hermite(n, x) * np.exp(-0.5 * x**2)
        val, _ = quad(integrand, 0, np.inf, limit=200)
        return val
    
    M0 = compute_moment(0)
    M4 = compute_moment(4)
    M8 = compute_moment(8)
    M12 = compute_moment(12)
    
    # Precompute the "abs" branch integral coefficients
    # When val < 0, we need int_0^inf x^4 |P(x)| exp(-x^2/2) dx
    # This is piecewise linear in coeffs, so we can precompute
    # the integral of x^4 * H_n(x) * exp(-x^2/2) * sign(P(x)) dx
    # But sign depends on coeffs, so we use a different approach:
    # For the abs branch, the objective is piecewise linear, so we can
    # just evaluate it directly with quad when needed.
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        val = c0 * M0 + c4 * M4 + c8 * M8 + c12 * M12
        
        if val < 0:
            def integrand(x):
                h0 = 1.0
                h4 = eval_hermite(4, x)
                h8 = eval_hermite(8, x)
                h12 = eval_hermite(12, x)
                p_val = c0*h0 + c4*h4 + c8*h8 + c12*h12
                g_val = p_val * np.exp(-0.5 * x**2)
                return (x**4) * np.abs(g_val)
            val, _ = quad(integrand, 0, np.inf, limit=300, epsabs=1e-10, epsrel=1e-10)
        
        return 2.0 * val
    
    def objective_grad(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        val = c0 * M0 + c4 * M4 + c8 * M8 + c12 * M12
        
        if val < 0:
            # For the abs branch, use central differences with larger step
            eps = 1e-6
            grad = np.zeros(3)
            for i in range(3):
                x_plus = coeffs.copy()
                x_minus = coeffs.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
        
        dc12_dc0 = -1.0 / 945.0
        dc12_dc4 = -3.0 / 945.0
        dc12_dc8 = -105.0 / 945.0
        
        grad = np.array([
            M0 + dc12_dc0 * M12,
            M4 + dc12_dc4 * M12,
            M8 + dc12_dc8 * M12
        ])
        
        return 2.0 * grad
    
    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score
    
    np.random.seed(42)
    # Tighter bounds based on the seed solution
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Use SLSQP with analytic gradient
    num_starts = 40
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 40:
            break
        
        # Mix of random starts and perturbed seed
        if i % 3 == 0:
            x0 = best_coeffs + np.random.normal(0, 0.1, 3)
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])
        
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(objective, x0, jac=objective_grad, method='SLSQP',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish pass
    if time.time() < TIME_LIMIT - 15:
        try:
            res = minimize(objective, best_coeffs, jac=objective_grad, method='SLSQP',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Second polish with L-BFGS-B for different convergence path
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(objective, best_coeffs, jac=objective_grad, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
