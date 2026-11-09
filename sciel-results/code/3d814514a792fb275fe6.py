"""Second autocorrelation inequality: maximize ratio via smooth optimization with log-sum-exp."""
import numpy as np
import time
import sys

N = 50
DT = 1.0 / N  # interval spacing on [-1/4, 1/4]

def _autoconv(x):
    """Linear autoconvolution of x with itself."""
    return np.convolve(x, x, mode='full')

def _ratio(x):
    """Compute the ratio R = ||x||_2^2 / (||x||_1 * ||x*x||_inf)."""
    x = np.asarray(x, dtype=np.float64)
    l2_sq = np.sum(x ** 2)
    if l2_sq < 1e-30:
        return 0.0
    l1 = np.sum(np.abs(x))
    if l1 < 1e-30:
        return 0.0
    ac = _autoconv(x)
    ac_max = np.max(np.abs(ac))
    if ac_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _soft_max(v, beta):
    """Smooth approximation to max(v) using log-sum-exp."""
    v_max = np.max(v)
    if v_max < 1e-30:
        return 0.0
    # Use log-sum-exp for stability
    # max(v) ≈ (1/beta) * log(sum(exp(beta * v)))
    # But for large beta, this overflows. Use the shifted version.
    # Actually, for max, we want:
    # max(v) ≈ (1/beta) * log(sum(exp(beta * (v - v_max)))) + v_max
    # This is numerically stable.
    exp_terms = np.exp(beta * (v - v_max))
    return v_max + (1.0 / beta) * np.log(np.sum(exp_terms))

def _smooth_objective(x, beta=100.0):
    """
    Smooth surrogate for the objective.
    We want to maximize R = ||x||_2^2 / (||x||_1 * ||x*x||_inf).
    Equivalent to minimizing f(x) = ||x||_1 * ||x*x||_inf / ||x||_2^2.
    Or maximizing -log(f(x)) = 2*log(||x||_2) - log(||x||_1) - log(||x*x||_inf).
    
    Since ||x||_2 is fixed (we normalize), we minimize:
    g(x) = log(||x||_1) + log(soft_max(conv(x)))
    
    Wait, let's just minimize the denominator directly with a smooth max.
    Denominator D(x) = ||x||_1 * soft_max(conv(x)).
    We minimize D(x) subject to ||x||_2 = 1, x >= 0.
    """
    x = np.asarray(x, dtype=np.float64)
    l1 = np.sum(x)
    if l1 < 1e-30:
        return 1e30
    ac = _autoconv(x)
    # Use soft max for the supremum norm of autoconvolution
    ac_max_smooth = _soft_max(ac, beta)
    if ac_max_smooth < 1e-30:
        return 1e30
    return l1 * ac_max_smooth

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _ratio(best_x)
    
    # Try several initial configurations
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Triangular
    t = np.linspace(-1, 1, N)
    inits.append(np.maximum(0, 1 - np.abs(t)))
    # Gaussian-like
    t = np.linspace(-1, 1, N)
    inits.append(np.exp(-t**2))
    # Sinc-like (positive part)
    t = np.linspace(-1, 1, N)
    sinc = np.sinc(t)
    inits.append(np.maximum(0, sinc))
    # Two peaks
    x2 = np.zeros(N)
    x2[N//4] = 1.0
    x2[3*N//4] = 1.0
    inits.append(x2)
    # Ramp up then down
    x3 = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    inits.append(x3)
    # Cosine bump
    t = np.linspace(0, np.pi, N)
    inits.append(0.5 * (1 + np.cos(t)))
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        x_rand = rng.random(N)
        inits.append(x_rand)
    
    # Use scipy.optimize.minimize with smooth objective
    from scipy.optimize import minimize
    
    # Start with a moderate beta, then increase for refinement
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for x0 in inits:
        if time.time() > deadline - 10:
            break
        
        x0 = np.asarray(x0, dtype=np.float64)
        if np.sum(x0**2) < 1e-30:
            continue
        # Normalize to unit L2
        x0 = x0 / np.sqrt(np.sum(x0**2))
        
        # Ensure nonnegative
        x0 = np.maximum(x0, 0)
        if np.sum(x0**2) < 1e-30:
            continue
        x0 = x0 / np.sqrt(np.sum(x0**2))
        
        # Optimize with increasing beta
        current_x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            def objective(x, b=beta):
                return _smooth_objective(x, b)
            
            # Use SLSQP with nonnegativity constraints
            bounds = [(0, None)] * N
            # Add L2 norm constraint
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                            'jac': lambda x: 2 * x}]
            
            try:
                result = minimize(objective, current_x, method='SLSQP', bounds=bounds,
                                 constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
                if result.success:
                    x_opt = np.maximum(result.x, 0)
                    if np.sum(x_opt**2) > 1e-30:
                        x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                        score = _ratio(x_opt)
                        if score > best_score:
                            best_score = score
                            best_x = x_opt.copy()
                        current_x = x_opt
            except Exception:
                pass
    
    # Additional local refinement with Nelder-Mead on the raw ratio (non-smooth)
    # This helps escape local minima of the smooth approximation
    if time.time() < deadline - 10:
        # Restart from best_x and a few random points
        restarts = [best_x.copy()]
        for _ in range(10):
            x_rand = rng.random(N)
            x_rand = np.maximum(x_rand, 0)
            if np.sum(x_rand**2) > 1e-30:
                x_rand = x_rand / np.sqrt(np.sum(x_rand**2))
                restarts.append(x_rand)
        
        for x0 in restarts:
            if time.time() > deadline - 5:
                break
            
            try:
                def neg_ratio(x):
                    return -_ratio(x)
                
                result = minimize(neg_ratio, x0, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                x_opt = np.maximum(result.x, 0)
                if np.sum(x_opt**2) > 1e-30:
                    x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                    score = _ratio(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
            except Exception:
                pass
    
    # Final polish: local search on the best solution
    if time.time() < deadline - 2:
        # Try small perturbations
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            x_pert = best_x + 0.01 * rng.standard_normal(N)
            x_pert = np.maximum(x_pert, 0)
            if np.sum(x_pert**2) < 1e-30:
                continue
            x_pert = x_pert / np.sqrt(np.sum(x_pert**2))
            score = _ratio(x_pert)
            if score > best_score:
                best_score = score
                best_x = x_pert
    
    return list(best_x)

# Keep the interface exactly as required
if __name__ == '__main__':
    result = construct()
    print(f"Score: {_ratio(result)}")
