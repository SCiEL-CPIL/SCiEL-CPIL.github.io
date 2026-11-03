import numpy as np
from scipy.optimize import differential_evolution
import time
from math import log, factorial

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Differential Evolution to minimize the ratio of Laguerre norms, which corresponds
    to the upper bound on the uncertainty constant C4.
    """
    N = 12
    start_time = time.time()
    deadline = 600.0
    safety_margin = 60.0  # Leave 60s for safety

    # Precompute factorials up to degree 48 (2 * 24)
    # We need (i+j)! where i,j <= 24, so max 48.
    # Actually P is degree 24. P^2 is degree 48.
    # P' is degree 23. (P')^2 is degree 46.
    max_deg = 48
    fact = np.array([factorial(i) for i in range(max_deg + 1)], dtype=float)
    # To avoid overflow in factorials, we can scale.
    # 48! is approx 1e60. Float64 max is 1e308. So it's safe.
    
    def _compute_norms(r):
        """
        Computes A = <P, P> and B = <P', P'> for P(x) = prod (x - r_i)^2
        with weight e^-x.
        r is an array of 12 positive roots.
        """
        # Construct P(x) = prod (x - r_i)^2
        # We can do this iteratively.
        # Start with P = 1
        coeffs = np.array([1.0])
        
        for ri in r:
            # Multiply by (x - ri)^2 = x^2 - 2*ri*x + ri^2
            # New coeffs length increases by 2
            new_len = len(coeffs) + 2
            new_coeffs = np.zeros(new_len)
            
            # x^2 * coeffs
            new_coeffs[2:] += coeffs
            # -2*ri*x * coeffs
            new_coeffs[1:-1] += -2.0 * ri * coeffs
            # ri^2 * coeffs
            new_coeffs[:len(coeffs)] += ri * ri * coeffs
            
            coeffs = new_coeffs
        
        # coeffs[k] is coefficient of x^k in P(x)
        # P(x) = sum_{k=0}^{24} coeffs[k] x^k
        
        # Compute A = sum_{i,j} coeffs[i] coeffs[j] (i+j)!
        # We can use matrix multiplication or loops.
        # i, j up to 24.
        # A = sum_{i=0}^{24} sum_{j=0}^{24} coeffs[i] coeffs[j] fact[i+j]
        
        # Optimization: Precompute the matrix M[i,j] = fact[i+j]
        # M is 25x25.
        # A = coeffs @ M @ coeffs
        
        # Compute derivative coeffs
        # P'(x) = sum_{k=1}^{24} k * coeffs[k] x^{k-1}
        # d_coeffs[m] = (m+1) * coeffs[m+1]
        d_coeffs = np.zeros(24)
        for k in range(1, 25):
            d_coeffs[k-1] = k * coeffs[k]
            
        # Compute B = sum_{i,j} d_coeffs[i] d_coeffs[j] (i+j)!
        # i, j up to 23.
        
        # We can compute A and B using the same matrix M if we pad d_coeffs to 25.
        # Or just compute sums.
        
        # Let's build the matrix M once outside? No, r changes coeffs, but M is constant.
        # M[i,j] = fact[i+j] for i,j in 0..24.
        
        # Inside the function, we have coeffs.
        # A = np.sum(np.outer(coeffs, coeffs) * M_25)
        # B = np.sum(np.outer(d_coeffs, d_coeffs) * M_24)
        
        # To be faster, we can precompute M_25 and M_24.
        # But we are inside the objective function called many times.
        # Let's create M_25 and M_24 in the closure.
        
        A = np.dot(coeffs, M_25.dot(coeffs))
        B = np.dot(d_coeffs, M_24.dot(d_coeffs))
        
        if B == 0:
            return 1e100
        return A / B

    # Precompute matrices
    # M_25[i,j] = fact[i+j] for i,j in 0..24
    idx = np.arange(25)
    M_25 = fact[np.add.outer(idx, idx)]
    M_24 = fact[np.add.outer(np.arange(24), np.arange(24))]

    def objective(log_r):
        r = np.exp(log_r)
        # Check for validity
        if np.any(r <= 0) or np.any(r > 1e100):
            return 1e100
        try:
            val = _compute_norms(r)
            if not np.isfinite(val):
                return 1e100
            return val
        except Exception:
            return 1e100

    # Initial bounds for log_r
    # Roots of Laguerre L_12 are in [1, 20] roughly.
    # Double roots might be spread out.
    # Let's use bounds for r in [0.1, 100] => log_r in [-2.3, 4.6]
    bounds = [(-2.0, 5.0) for _ in range(N)]
    
    # Multi-start DE
    best_score = 1e100
    best_log_r = None
    
    # Check time
    if time.time() > deadline - safety_margin:
        # Fallback
        return [1.0]*N

    # Run DE
    # maxiter and popsize to fit in time
    # DE is stochastic.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6,
            seed=42,
            polish=True,
            maxfev=10000
        )
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_log_r = res.x
    except Exception:
        pass

    # Check time again
    if time.time() > deadline - safety_margin:
        if best_log_r is not None:
            r_opt = np.exp(best_log_r)
            r_opt = np.sort(r_opt)
            return [float(ri) for ri in r_opt]
        else:
            return [1.0]*N

    # If we have time, try a local refinement on the best found
    if best_log_r is not None:
        from scipy.optimize import minimize
        def obj_local(log_r):
            r = np.exp(log_r)
            if np.any(r <= 0):
                return 1e100
            try:
                return _compute_norms(r)
            except:
                return 1e100
        def grad_local(log_r):
            # Numerical gradient
            eps = 1e-8
            grad = np.zeros(N)
            f0 = obj_local(log_r)
            for i in range(N):
                log_r_plus = log_r.copy()
                log_r_plus[i] += eps
                fp = obj_local(log_r_plus)
                grad[i] = (fp - f0) / eps
            return grad

        res_local = minimize(
            obj_local, 
            best_log_r, 
            jac=grad_local,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res_local.fun < best_score:
            best_score = res_local.fun
            best_log_r = res_local.x

    if best_log_r is None:
        # Fallback to Laguerre roots
        x0, _ = np.polynomial.laguerre.laggauss(N)
        return [float(xi) for xi in x0]
        
    r_opt = np.exp(best_log_r)
    r_opt = np.sort(r_opt)
    
    # Ensure positivity
    r_opt = np.maximum(r_opt, 1e-6)
    
    return [float(ri) for ri in r_opt]
