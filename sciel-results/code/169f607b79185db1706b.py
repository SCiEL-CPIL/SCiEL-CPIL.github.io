import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_laguerre

def construct():
    N = 12
    
    # Standard Laguerre Jacobi matrix entries
    d_init = np.array([2*i + 1 for i in range(N)], dtype=float)
    o_init = np.array([float(i) for i in range(1, N)], dtype=float)
    
    def cost(params):
        d = params[:N]
        o = params[N:]
        
        try:
            w, _ = eigh_tridiagonal(d, o, select='all')
        except Exception:
            return 1e10
            
        if np.any(w <= 1e-6):
            return 1e10
            
        # Proxy for C4 bound: Product of spatial variance and inverse-square sum
        # This balances spread (variance) and proximity to origin (momentum-like term)
        var_x = np.var(w)
        # Use sum of inverse squares as a proxy for the 'frequency' uncertainty
        # which diverges if roots approach 0
        inv_sq_sum = np.sum(1.0 / (w**2))
        
        # The C4 bound is roughly proportional to sqrt(var_x * inv_sq_sum)
        # Minimizing the product is equivalent to minimizing the bound
        return var_x * inv_sq_sum

    x0 = np.concatenate([d_init, o_init])
    
    # Bounds: keep roots positive and well-separated
    bounds = [(1.0, 100.0)] * N + [(0.1, 50.0)] * (N - 1)
    
    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
    
    if res.success and res.fun < cost(x0):
        d_opt = res.x[:N]
        o_opt = res.x[N:]
        w, _ = eigh_tridiagonal(d_opt, o_opt, select='all')
        if np.any(w <= 1e-6):
            roots, _ = roots_laguerre(N)
            return [float(r) for r in roots]
        return [float(r) for r in w]
    else:
        # Fallback to standard Laguerre roots if optimization fails or doesn't improve
        roots, _ = roots_laguerre(N)
        return [float(r) for r in roots]
