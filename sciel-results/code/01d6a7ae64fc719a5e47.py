import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _evaluate(h):
    """Compute the objective value for a given set of heights h."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)  # m = 2*N - 1 = 99
    y = np.zeros(m + 2)
    y[1:-1] = g  # y = [0, g[0], ..., g[m-1], 0]
    
    w = 1.0 / (m + 1)  # spacing on [-0.5, 0.5] with m+2 points
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _log_sum_exp(x, beta=100.0):
    """Differentiable approximation to max(x) using log-sum-exp."""
    max_x = np.max(x)
    scaled = beta * (x - max_x)
    # Clip to avoid overflow in exp
    scaled = np.clip(scaled, -50, 50)
    return max_x + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))

def _negative_value_smooth(h, beta=100.0):
    """Negative of the objective with differentiable max surrogate."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or not np.all(np.isfinite(h)):
        return 1e10
    
    # Enforce non-negativity softly or hard? Hard bounds in optimizer.
    # But for safety in evaluation:
    h = np.maximum(h, 0)
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Use log-sum-exp surrogate for norm_inf
    abs_g = np.abs(g)
    norm_inf_approx = _log_sum_exp(abs_g, beta=beta)
    
    if norm_1 * norm_inf_approx < 1e-300:
        return 1e10
    
    return -l2 / (norm_1 * norm_inf_approx)

def _parametric_h(params, family='gaussian'):
    """Generate heights h from parametric family."""
    x = np.linspace(-0.25, 0.25, N)
    if family == 'gaussian':
        amp, mu, sigma = params
        sigma = abs(sigma) + 1e-6
        h = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    elif family == 'super_gaussian':
        amp, mu, sigma, p = params
        sigma = abs(sigma) + 1e-6
        p = abs(p) + 1e-6
        h = amp * np.exp(-((abs(x - mu) / sigma) ** p))
    elif family == 'polynomial':
        amp, mu, sigma, p = params
        sigma = abs(sigma) + 1e-6
        p = abs(p) + 1e-6
        t = (x - mu) / sigma
        # Flat top or polynomial decay
        h = amp * np.maximum(1 - t**2, 0) ** p
    elif family == 'hamming':
        amp, mu, sigma = params
        sigma = abs(sigma) + 1e-6
        t = (x - mu) / sigma
        h = amp * (0.54 + 0.46 * np.cos(np.pi * t / 0.5)) * np.maximum(1 - np.abs(t)/0.5, 0)
    elif family == 'two_gauss':
        amp1, mu1, sigma1, amp2, mu2, sigma2 = params
        sigma1 = abs(sigma1) + 1e-6
        sigma2 = abs(sigma2) + 1e-6
        h1 = amp1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        h2 = amp2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        h = h1 + h2
    else:
        h = np.ones(N)
    
    return np.maximum(h, 0)

def _optimize_parametric(family, start_params, bounds, start_time, deadline):
    """Optimize parameters for a specific family."""
    def obj(params):
        if time.time() > deadline:
            return 1e10
        h = _parametric_h(params, family)
        return _negative_value_smooth(h, beta=100.0)
    
    try:
        result = minimize(
            obj,
            x0=start_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success:
            best_h = _parametric_h(result.x, family)
            val = _evaluate(best_h)
            return val, best_h
    except Exception:
        pass
    return 0.0, None

def construct():
    """Construct 50 nonnegative step heights to maximize the objective."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # 1. Try various parametric families
    families = [
        ('gaussian', [(1.0, 0.0, 0.1)], [(-10, 10), (-0.25, 0.25), (0.01, 0.5)]),
        ('super_gaussian', [(1.0, 0.0, 0.1, 2.0)], [(-10, 10), (-0.25, 0.25), (0.01, 0.5), (0.5, 10.0)]),
        ('polynomial', [(1.0, 0.0, 0.25, 1.0)], [(-10, 10), (-0.25, 0.25), (0.01, 0.5), (0.5, 10.0)]),
        ('two_gauss', [(1.0, -0.1, 0.1, 1.0, 0.1, 0.1)], 
                    [(-10, 10), (-0.25, 0.25), (0.01, 0.5), (-10, 10), (-0.25, 0.25), (0.01, 0.5)])
    ]
    
    for family, start_params, bounds in families:
        if time.time() > deadline - 10:
            break
        val, h_candidate = _optimize_parametric(family, start_params, bounds, start_time, deadline)
        if h_candidate is not None and val > best_val:
            best_val = val
            best_h = h_candidate

    # 2. Refine the best candidate using full 50-dimensional optimization
    # Use a smooth surrogate to handle the max term
    if time.time() < deadline - 10:
        try:
            result = minimize(
                fun=lambda x: _negative_value_smooth(x, beta=200.0),
                x0=best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                val = _evaluate(result.x)
                if val > best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass

    # 3. Final local search / hill climbing if time remains
    if time.time() < deadline - 2:
        # Simple coordinate ascent
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    h_new = best_h.copy()
                    h_new[i] += delta
                    if h_new[i] < 0:
                        continue
                    val_new = _evaluate(h_new)
                    if val_new > best_val:
                        best_val = val_new
                        best_h = h_new
                        improved = True
                        break

    best_h = np.maximum(best_h, 0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
