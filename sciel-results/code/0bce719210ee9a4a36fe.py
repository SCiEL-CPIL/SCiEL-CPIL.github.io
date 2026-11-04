import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.laguerre import laggauss, lagval

def construct():
    m = 12
    # Initial guess: Roots of the Laguerre polynomial L_m
    # laggauss returns nodes and weights for integration, but we can get roots via laggauss nodes?
    # No, laggauss nodes are the roots of L_n(x). So laggauss(m)[0] are the roots of L_m.
    roots_init, _ = laggauss(m)
    
    # Quadrature for integration
    # Degree of P(x)^2 is 2*m = 24.
    # Integrand is e^-x * x^k * P(x)^2.
    # Max power of x in integrand for variance: x^2 * P(x)^2. Degree 2+24=26.
    # Need 2N-1 >= 26 => N >= 13.5 => N=14.
    # Use N=50 for safety.
    n_quad = 50
    x_nodes, w_nodes = laggauss(n_quad)
    
    def cost_func(roots):
        # Compute P(x) = prod (x - xi)^2
        # To avoid overflow/underflow issues with large products, 
        # we compute log(P) or just P directly if values are reasonable.
        # Roots are ~1-40. x_nodes are ~1-100.
        # (x-xi) can be ~100. 100^24 is 1e48. e^-x is small.
        # It's safer to compute in log space or normalize.
        # However, for variance, we need ratios, so scale cancels out.
        
        diff = x_nodes[:, None] - roots[None, :]
        # P(x) = prod (x-xi)^2
        # log P(x) = 2 * sum log |x-xi|
        # But we need P(x)^2 = prod (x-xi)^4
        # Let's compute P_sq = prod (x-xi)^4
        # To prevent overflow, we can subtract the max log value?
        # Or just use float64 which goes up to 1e308.
        # 100^48 = 1e96. This is fine.
        
        prod_diff_sq = np.prod(diff**4, axis=1)
        weights = w_nodes * prod_diff_sq * np.exp(-x_nodes)
        
        norm = np.sum(weights)
        if norm < 1e-300:
            return 1e300
        
        mu = np.sum(weights * x_nodes) / norm
        mu2 = np.sum(weights * x_nodes**2) / norm
        var = mu2 - mu**2
        
        return var

    # Bounds for roots
    bounds = [(1e-6, 100) for _ in range(m)]
    
    # Optimize
    # Use L-BFGS-B for bounded variables
    result = minimize(cost_func, roots_init, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    # If optimization fails or goes out of bounds, return initial
    if not result.success:
        # Check if it's better than initial
        if result.fun < cost_func(roots_init):
            return list(result.x)
        else:
            return list(roots_init)
            
    return list(result.x)
