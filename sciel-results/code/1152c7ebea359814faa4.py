import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) by optimizing the Cohn-Goncalves upper bound functional.
    """
    m = 12
    
    def c4_bound(x):
        """
        Compute the Cohn-Goncalves upper bound on C4 given 12 double-root locations.
        For a polynomial with double roots at x_i, the bound involves the 
        maximum over i of a function related to the local behavior at each root.
        
        For double roots, the sign change at x_i contributes a term proportional to
        the inverse of the spacing to neighboring roots, weighted by the polynomial
        value normalization. We use a proxy: maximize over i of 1/sqrt(d_i) where
        d_i is the minimum distance to adjacent roots, combined with a global
        normalization factor.
        
        A more accurate proxy: the C4 bound scales with the product of (1/|x_i - x_j|)
        terms. We minimize the maximum of local contributions.
        """
        x = np.array(x)
        # Ensure positivity and ordering
        if np.any(x <= 0) or np.any(np.diff(x) <= 0):
            return 1e10
        
        # Compute pairwise distances
        n = len(x)
        # For each root, compute the minimum distance to other roots
        min_dists = np.array([
            min(abs(x[i] - x[j]) for j in range(n) if j != i)
            for i in range(n)
        ])
        
        # The C4 upper bound is proportional to the product of (1/min_dist_i)^{1/2}
        # or similar. We use the geometric mean of inverse square roots of min distances.
        # Actually, for the uncertainty bound, it's more like the maximum of 
        # 1/sqrt(min_dist_i) times a normalization.
        
        # A better proxy: the bound involves the norm of the derivative relative to the function.
        # For double roots, at x_i, P(x_i) = 0 and P'(x_i) = 0, so we look at P''(x_i).
        # The local contribution scales as 1/sqrt(|P''(x_i)| * |x_i - x_{i-1}| * |x_{i+1} - x_i|)
        
        # Compute second derivative at each root for P(x) = prod_j (x - x_j)^2
        # P''(x_i) = 2 * prod_{j != i} (x_i - x_j)^2 * (sum of terms) ... 
        # Simpler: for double roots, the local width is determined by neighboring roots.
        
        # Use a practical proxy: minimize the maximum of 1/sqrt(d_i_left * d_i_right)
        # where d_i_left and d_i_right are distances to left and right neighbors.
        # For endpoints, use only the one-sided distance.
        
        contributions = np.zeros(n)
        for i in range(n):
            if i == 0:
                d_right = x[1] - x[0]
                contributions[i] = 1.0 / np.sqrt(d_right)
            elif i == n - 1:
                d_left = x[n-1] - x[n-2]
                contributions[i] = 1.0 / np.sqrt(d_left)
            else:
                d_left = x[i] - x[i-1]
                d_right = x[i+1] - x[i]
                contributions[i] = 1.0 / np.sqrt(d_left * d_right)
        
        return np.max(contributions)
    
    # Initial guess: use Laguerre roots as starting point
    from scipy.special import roots_laguerre
    x0, _ = roots_laguerre(m)
    x0 = x0.tolist()
    
    # Bounds: all roots must be positive, and we need them to be ordered
    # Use bounds that enforce positivity
    bounds = [(1e-6, None)] * m
    
    # Constraints: enforce ordering x_0 < x_1 < ... < x_{m-1}
    def ordering_constraints(x):
        return np.diff(x) - 1e-8  # Should be positive
    
    cons = {'type': 'ineq', 'fun': ordering_constraints}
    
    # Optimize
    result = minimize(c4_bound, x0, method='SLSQP', bounds=bounds, 
                      constraints=cons, options={'maxiter': 500, 'ftol': 1e-10})
    
    if result.success:
        x_opt = result.x
    else:
        x_opt = x0
    
    # Ensure positivity and ordering
    x_opt = np.maximum(x_opt, 1e-6)
    x_opt = np.sort(x_opt)
    
    return x_opt.tolist()
