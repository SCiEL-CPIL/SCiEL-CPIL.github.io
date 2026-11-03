import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.laguerre import laggauss, Laguerre

def construct():
       # 1. Get roots of L_24
       # Laguerre.roots() gives roots of L_n
       L24 = Laguerre.basis(24)
       roots_L24 = L24.roots()
       roots_L24 = np.sort(np.real(roots_L24))
       
       # 2. Pair adjacent roots
       # Shape (24,) -> (12, 2)
       pairs = roots_L24.reshape(12, 2)
       initial_roots = np.mean(pairs, axis=1)
       
       # 3. Setup Quadrature
       # Degree of integrand is 48.
       # laggauss(N) is exact for degree 2N-1.
       # 2N-1 >= 48 => N >= 24.5 => N=25.
       # Use N=50 for accuracy.
       n_quad = 50
       x_nodes, w_nodes = laggauss(n_quad)
       
       def cost_func(roots):
           # roots: (12,)
           # diff: (N, 12)
           diff = x_nodes[:, None] - roots[None, :]
           # prod (x-xi)^4
           # log-sum-exp trick might be needed if values are huge?
           # But roots are positive, x_nodes positive.
           # (x-xi)^4 can be large.
           # Let's compute directly first.
           prod_diff = np.prod(diff**4, axis=1)
           integral_val = np.sum(w_nodes * prod_diff)
           return integral_val
       
       # 4. Optimize
       # Bounds: roots must be positive.
       bounds = [(1e-8, 100) for _ in range(12)]
       
       # Use L-BFGS-B
       result = minimize(cost_func, initial_roots, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
       
       # If L-BFGS fails, try Powell
       if not result.success:
           result = minimize(cost_func, initial_roots, method='Powell', bounds=bounds, options={'maxiter': 1000})
           
       return list(result.x)
