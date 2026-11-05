"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h)
    peak = np.max(conv)
    return abs(2.0 * N * peak / (s * s))

def _smooth_max(conv, temperature=1.0):
    """Soft-max approximation using log-sum-exp."""
    m = np.max(conv)
    return m + temperature * np.log(np.sum(np.exp((conv - m) / temperature)))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline for 150s limit
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Try several initial patterns
    patterns = [
        np.ones(N),
        np.ones(N) * 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(N) / N),
        np.ones(N) * 0.5 + 0.5 * np.cos(2 * np.pi * 2 * np.arange(N) / N),
        np.ones(N) * 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / N),
        np.ones(N) * 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(N) / (N * 0.5)),
    ]
    
    for p in patterns:
        v = _evaluate(p)
        if v < best_val:
            best_val = v
            best_h = p.copy()
    
    # Gradient descent on smooth surrogate
    def objective(x):
        if time.time() > deadline:
            return 1e10
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        # Use soft-max to make it differentiable
        temp = 1.0
        m = np.max(conv)
        soft_max = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
        val = 2.0 * N * soft_max / (s * s)
        return val
    
    def gradient(x):
        if time.time() > deadline:
            return np.zeros(N)
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.ones(N)
        conv = np.convolve(h, h)
        temp = 1.0
        m = np.max(conv)
        exp_conv = np.exp((conv - m) / temp)
        softmax_weights = exp_conv / np.sum(exp_conv)
        # d(soft_max)/d(conv[k]) = softmax_weights[k]
        # d(conv[k])/d(h[i]) = h[i-k] + h[i+k-2N+2] (with bounds)
        # Simplified: use numerical gradient for reliability
        eps = 1e-6
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            v_plus = objective(h_plus)
            h_minus = h.copy()
            h_minus[i] -= eps
            v_minus = objective(h_minus)
            grad[i] = (v_plus - v_minus) / (2 * eps)
        return grad
    
    # Use L-BFGS-B with the smooth objective
    x0 = best_h.copy()
    x0 = x0 / np.sum(x0)  # normalize
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8},
            callback=lambda x: None
        )
        if result.fun < best_val:
            best_h = result.x
            best_val = _evaluate(best_h)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_h
    
    # Local refinement with finer temperature
    for temp in [0.5, 0.2, 0.1]:
        def obj_fine(x):
            h = x
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h)
            m = np.max(conv)
            soft_max = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
            return 2.0 * N * soft_max / (s * s)
        
        try:
            result = minimize(
                obj_fine,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            v = _evaluate(result.x)
            if v < best_val:
                best_h = result.x
                best_val = v
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Final polish: try to reduce the actual max by shifting
    # The max of conv is at some lag; try to reduce it
    conv = np.convolve(best_h, best_h)
    max_idx = np.argmax(conv)
    
    # Try small perturbations
    rng = np.random.RandomState(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        h_try = best_h + rng.randn(N) * 0.01 * np.std(best_h)
        v = _evaluate(h_try)
        if v < best_val:
            best_h = h_try
            best_val = v
    
    return best_h
# EVOLVE-BLOCK-END
