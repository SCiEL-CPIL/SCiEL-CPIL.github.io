"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_objective(h, beta, log_scale):
    """
    Smooth objective: maximize l2 / (norm_1 * norm_inf_smooth)
    where norm_inf_smooth = (1/beta) * log(sum(exp(beta*g_i)))
    We minimize the negative of this.
    h is nonnegative; we work in log-space or directly with bounds.
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth max via log-sum-exp
    max_g = np.max(g)
    if max_g < 1e-30:
        return 0.0
    # log-sum-exp for numerical stability
    shifted = beta * (g - max_g)
    log_sum_exp = max_g + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    
    norm_inf_smooth = log_sum_exp
    
    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0
    
    val = l2 / (norm_1 * norm_inf_smooth)
    return -val  # minimize negative

def _gradient_smooth(h, beta):
    """Numerical gradient for the smooth objective."""
    eps = 1e-7
    grad = np.zeros_like(h)
    f0 = _smooth_objective(h, beta, 0)
    for i in range(len(h)):
        h_plus = h.copy()
        h_plus[i] += eps
        grad[i] = (_smooth_objective(h_plus, beta, 0) - f0) / eps
    return grad

def construct():
    global START_TIME, TIME_LIMIT
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_value(best_h)
    
    # Multi-stage optimization with increasing beta
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Normalize to fix scaling: set norm_1 to a reasonable value
        g = np.convolve(current_h, current_h)
        norm_1 = np.sum(g) / (len(g) + 1)
        if norm_1 > 1e-10:
            current_h = current_h / np.sqrt(norm_1)  # This changes norm_1, let me think...
        
        # Actually, due to scaling invariance, let's just work with the shape
        # and let the optimizer handle it. Use bounds [0, inf).
        
        def obj(x):
            return _smooth_objective(x, beta, 0)
        
        def grad(x):
            return _gradient_smooth(x, beta)
        
        try:
            result = minimize(
                obj,
                current_h,
                jac=grad,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success or result.fun < obj(current_h):
                candidate = np.maximum(result.x, 0.0)
                val = _compute_value(candidate)
                if val > best_val:
                    best_val = val
                    best_h = candidate.copy()
                current_h = candidate
        except Exception:
            pass
        
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            break
    
    # Final polish: try a few random perturbations from best
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            perturbation = rng.randn(N) * 0.1
            candidate = np.maximum(best_h + perturbation, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)

# Ensure the function is defined at module level for the evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Value: {_compute_value(result)}")
