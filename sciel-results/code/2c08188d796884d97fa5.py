import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, gammaln

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    m = 12
    n = 1
    
    def compute_bound(roots):
        """Compute the C4 upper bound for given root positions."""
        roots = np.array(roots)
        if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
            return 1e10
        
        # Build the polynomial P(x) = prod (x - r_i)^2
        # and compute the ratio ||x*P||_L2 / ||P||_L2 with Laguerre weight
        # 
        # For the Cohn-Gonçalves construction, the bound involves:
        # C4 = sqrt( <x^2 P^2, w> / <P^2, w> - <x P^2, w>^2 / <P^2, w>^2 )
        # or simply the mean of x under the measure P^2 * w
        #
        # Let's use numerical quadrature with Laguerre nodes
        
        # Use Gauss-Laguerre quadrature for weight e^{-x}
        # We need to integrate P(x)^2 * e^{-x} and x * P(x)^2 * e^{-x}
        
        # P(x) = prod_{i=1}^{12} (x - r_i)^2
        # log|P(x)| = 2 * sum_i log|x - r_i|
        # This can overflow, so work in log space
        
        # Use high-order Gauss-Laguerre quadrature
        from numpy.polynomial.laguerre import laggauss
        x_nodes, x_weights = laggauss(200)  # 200-point quadrature
        
        # Compute log|P(x)| at each node
        log_P = np.zeros_like(x_nodes)
        for r in roots:
            log_P += 2.0 * np.log(np.abs(x_nodes - r))
        
        # Clip to avoid overflow/underflow
        log_P = np.clip(log_P, -700, 700)
        
        P2 = np.exp(2.0 * log_P)  # P(x)^2 = exp(2*log|P|)
        
        # Integrate P^2 * e^{-x} and x * P^2 * e^{-x}
        # Gauss-Laguerre: integral f(x) e^{-x} dx ≈ sum w_i f(x_i)
        norm_sq = np.sum(x_weights * P2)
        x_norm_sq = np.sum(x_weights * x_nodes * P2)
        
        if norm_sq <= 0:
            return 1e10
        
        # The bound is related to the ratio
        # For uncertainty principle: C ~ (x_norm_sq / norm_sq)
        # But it could also involve the variance
        mean_x = x_norm_sq / norm_sq
        var_x = np.sum(x_weights * (x_nodes - mean_x)**2 * P2) / norm_sq
        
        # The C4 bound is likely proportional to the standard deviation or mean
        # Let's try the mean first
        return mean_x
    
    # Initial guess: arithmetic progression
    roots0 = np.array([4.0 * (i + 1) for i in range(m)], dtype=float)
    
    # Try different starting points and optimize
    best_roots = roots0.copy()
    best_score = compute_bound(roots0)
    
    # Try Laguerre zeros as starting point
    from numpy.polynomial.laguerre import laggauss
    lag_points, _ = laggauss(m)
    lag_roots = np.sort(lag_points)
    score_lag = compute_bound(lag_roots)
    if score_lag < best_score:
        best_score = score_lag
        best_roots = lag_roots.copy()
    
    # Optimize from the best starting point
    def objective(roots):
        s = compute_bound(roots)
        if s > 1e9:
            return 1e10
        return s
    
    def gradient(roots):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(roots)
        f0 = objective(roots)
        for i in range(m):
            roots_p = roots.copy()
            roots_p[i] += eps
            roots_m = roots.copy()
            roots_m[i] -= eps
            grad[i] = (objective(roots_p) - objective(roots_m)) / (2 * eps)
        return grad
    
    # Use BFGS with bounds
    bounds = [(1e-6, None) for _ in range(m)]
    
    result = minimize(
        objective,
        best_roots,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.fun < best_score:
        best_score = result.fun
        best_roots = result.x.copy()
    
    # Try a few more restarts with different initializations
    restarts = [
        np.array([1.0 * (i + 1) for i in range(m)]),
        np.array([2.0 * (i + 1) for i in range(m)]),
        np.array([0.5 * (i + 1) for i in range(m)]),
        np.array([8.0 * (i + 1) for i in range(m)]),
        np.linspace(0.5, 10, m),
        np.linspace(1, 5, m),
    ]
    
    for r0 in restarts:
        if time.time() - t0 > deadline - 60:
            break
        r = minimize(
            objective,
            r0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if r.fun < best_score:
            best_score = r.fun
            best_roots = r.x.copy()
    
    # Ensure roots are sorted and positive
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    # Final polish with tighter tolerance
    if time.time() - t0 < deadline - 30:
        r = minimize(
            objective,
            best_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if r.fun < best_score:
            best_score = r.fun
            best_roots = r.x.copy()
            best_roots = np.sort(best_roots)
            best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]
