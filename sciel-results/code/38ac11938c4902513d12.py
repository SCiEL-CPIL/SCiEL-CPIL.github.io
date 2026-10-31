import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Hermite Polynomials (Physicists')
    # We need H0, H4, H8, H12
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280
    
    # Coefficients at x=0 for constraint
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0 # Corrected value
    
    # Gauss-Hermite quadrature nodes and weights
    # n=100 is usually sufficient for high accuracy with polynomials of degree ~12
    n_quad = 100
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    
    # Precompute Hermite values at nodes for L2 (x) and L1 (sqrt(2)*x)
    # L2 integral: sum w_i * P(x_i)^2
    # L1 integral: sqrt(2) * sum w_i * |P(sqrt(2)*x_i)|
    
    # Precompute H_k(x_i) and H_k(sqrt(2)*x_i) for k in [0, 4, 8, 12]
    # This is a one-time cost
    
    def eval_hermite(k, x):
        if k == 0:
            return np.ones_like(x)
        elif k == 4:
            return 16*x**4 - 48*x**2 + 12
        elif k == 8:
            return 256*x**8 - 4480*x**6 + 26880*x**4 - 48384*x**2 + 1680
        elif k == 12:
            return 4096*x**12 - 129024*x**10 + 1548288*x**8 - 7654656*x**6 + 16636928*x**4 - 15974400*x**2 + 665280
        else:
            raise ValueError("Unsupported degree")

    H_vals_x = {k: eval_hermite(k, x_gh) for k in [0, 4, 8, 12]}
    H_vals_s2x = {k: eval_hermite(k, np.sqrt(2)*x_gh) for k in [0, 4, 8, 12]}
    
    def objective_and_grad(c):
        c0, c4, c8 = c
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280.0
        coeffs = np.array([c0, c4, c8, c12])
        degrees = [0, 4, 8, 12]
        
        # Evaluate P(x) and P(sqrt(2)x)
        P_x = np.zeros_like(x_gh)
        P_s2x = np.zeros_like(x_gh)
        for i, k in enumerate(degrees):
            P_x += coeffs[i] * H_vals_x[k]
            P_s2x += coeffs[i] * H_vals_s2x[k]
            
        # L2^2 = sum w_i * P(x_i)^2
        L2_sq = np.sum(w_gh * P_x**2)
        
        # Smoothed L1 to handle non-differentiability of abs()
        eps = 1e-8
        abs_P_s2x = np.sqrt(P_s2x**2 + eps)
        L1 = np.sqrt(2) * np.sum(w_gh * abs_P_s2x)
        
        if L2_sq < 1e-15:
            return 1e10, np.zeros(3)
            
        J = (L1**2) / L2_sq
        
        # Gradients
        dL2_sq_dc = np.zeros(3)
        dL1_dc = np.zeros(3)
        
        # dL2_sq/dc_j = 2 * sum w_i * P(x_i) * H_j(x_i)
        for j in range(3):
            k = degrees[j]
            dL2_sq_dc[j] = 2 * np.sum(w_gh * P_x * H_vals_x[k])
            
        # dL1/dc_j = sqrt(2) * sum w_i * (P_s2x / sqrt(P_s2x^2 + eps)) * H_j(sqrt(2)x_i)
        grad_abs = P_s2x / np.sqrt(P_s2x**2 + eps)
        for j in range(3):
            k = degrees[j]
            dL1_dc[j] = np.sqrt(2) * np.sum(w_gh * grad_abs * H_vals_s2x[k])
            
        grad_J = (2 * L1 / L2_sq) * dL1_dc - (L1**2 / L2_sq**2) * dL2_sq_dc
        
        return J, grad_J

    def objective(c):
        val, _ = objective_and_grad(c)
        return val

    def objective_grad(c):
        _, grad = objective_and_grad(c)
        return grad

    # Initial guess
    c_init = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_c = c_init
    best_val, _ = objective_and_grad(c_init)
    
    # Optimization with L-BFGS-B
    try:
        res = minimize(objective, c_init, jac=objective_grad, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass
        
    # Aggressive Multi-start Optimization
    rng = np.random.default_rng(42)
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Distribute restarts among scales
    restarts_per_scale = 20
    
    for scale in scales:
        for _ in range(restarts_per_scale):
            if time.time() > deadline - 10:
                break
            c_rand = c_init + rng.normal(0, scale, 3)
            try:
                res_r = minimize(objective, c_rand, jac=objective_grad, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_c = res_r.x
            except Exception:
                pass
        if time.time() > deadline - 10:
            break

    return best_c
