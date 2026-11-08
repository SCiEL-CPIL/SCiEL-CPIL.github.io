import numpy as np
import scipy.optimize as opt
import time

def construct():
    """
    Optimizes the 12 double-root locations for the Laguerre combination to minimize
    the L2 norm squared of the polynomial product (x-r_i)^2 with respect to the 
    Laguerre weight e^-x. This is a proxy for the uncertainty bound C4.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 600  # seconds
    
    # Number of roots
    m = 12
    
    # Initial guess: Laguerre nodes of order 12
    # These are positive and distinct
    initial_roots, _ = np.polynomial.laguerre.laggauss(m)
    initial_roots = sorted(initial_roots)
    
    # Define the objective function
    # We want to minimize J(r) = int_0^inf (prod (x-r_i)^2)^2 e^-x dx
    # We use Gauss-Laguerre quadrature with N points to approximate the integral.
    # The integrand is a polynomial of degree 2*m*2 = 48.
    # Gauss-Laguerre quadrature with N points is exact for polynomials of degree 2N-1.
    # So we need 2N-1 >= 48 => N >= 24.5 => N=25 is enough for exactness if we use the standard nodes.
    # However, the nodes x_q are fixed. The integrand depends on r.
    # Let's use N=50 for safety and stability.
    N_quad = 50
    x_quad, w_quad = np.polynomial.laguerre.laggauss(N_quad)
    
    def objective(r):
        # r is a vector of length m
        # Compute P(x) = prod (x - r_i)^2
        # We can compute this efficiently
        # P(x) = (prod (x - r_i))^2
        # Let Q(x) = prod (x - r_i)
        # Q(x) is a polynomial of degree m.
        # We can evaluate Q(x_quad) using np.polyval or iterative multiplication.
        
        # Efficient evaluation of product of (x - r_i)
        # Start with 1
        Q_vals = np.ones_like(x_quad)
        for ri in r:
            Q_vals *= (x_quad - ri)
        
        P_vals = Q_vals ** 2
        integrand = P_vals * P_vals * w_quad # (P^2) * e^-x approximated by w_quad
        # Wait, w_quad includes the e^-x weight?
        # np.polynomial.laguerre.laggauss returns nodes and weights for int f(x) e^-x dx.
        # So yes, sum(f(x_q) * w_q) approximates the integral.
        # Here f(x) = (prod (x-r_i)^2)^2 = (prod (x-r_i))^4.
        # So integrand is Q_vals**4 * w_quad.
        
        return np.sum(Q_vals**4 * w_quad)
    
    def objective_grad(r):
        # Numerical gradient might be safer/faster to implement correctly than analytic for this product
        # But let's try analytic or use scipy's approx.
        # For now, use numerical gradient via finite differences inside minimize?
        # L-BFGS-B can use numerical gradient.
        return None # Let scipy handle it if we don't provide it, or we provide a numerical one.
        
    # Let's use a wrapper for numerical gradient to be safe
    eps = 1e-8
    def grad(r):
        g = np.zeros_like(r)
        f0 = objective(r)
        for i in range(m):
            r_pert = r.copy()
            r_pert[i] += eps
            f_pert = objective(r_pert)
            g[i] = (f_pert - f0) / eps
        return g

    best_roots = initial_roots.copy()
    best_score = objective(initial_roots)
    
    # Multi-start optimization
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            r0 = initial_roots.copy()
        else:
            # Perturb initial roots
            r0 = initial_roots + np.random.normal(0, 0.1, m)
            # Ensure positivity
            r0 = np.abs(r0) + 0.1
            # Sort to maintain some structure, though order doesn't matter for the product
            r0 = sorted(r0)
            
        try:
            res = opt.minimize(
                objective, 
                r0, 
                method='L-BFGS-B', 
                jac=None, # Use numerical gradient
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                # Check if roots are positive
                if np.all(res.x > 0):
                    # Verify distinctness (optional, but good for stability)
                    # If roots collide, the gradient might be bad, but the value is valid.
                    if res.fun < best_score:
                        best_score = res.fun
                        best_roots = res.x.copy()
                        
        except Exception as e:
            pass

    # Final safety check: if optimization failed or went bad, return initial
    if not np.all(best_roots > 0):
        return sorted(initial_roots)
        
    return sorted(best_roots)
