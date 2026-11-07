"""Optimized program for the second autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _evaluate(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    yi = y[:m+1]
    yip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yip1 + yip1**2)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_params(params):
    """Convert parameter vector to height array."""
    x = np.linspace(-1, 1, N)
    a1, a2, a3, a4, a5, a6, mu, sigma, p = params
    gauss = np.exp(-0.5 * ((x - mu) / max(sigma, 0.01))**2)
    cos_bump = np.cos(np.pi * x / 2) ** 2
    tent = 1.0 - np.abs(x)
    plateau = np.ones(N)
    edge = max(1, int(N * max(0.05, p)))
    for i in range(edge):
        plateau[i] = i / edge
        plateau[N-1-i] = i / edge
    bimodal = np.exp(-4.0 * (x - 0.4)**2) + np.exp(-4.0 * (x + 0.4)**2)
    power = np.maximum(0.0, 1.0 - np.abs(x)) ** 2
    h = a1*gauss + a2*cos_bump + a3*tent + a4*plateau + a5*bimodal + a6*power
    return np.maximum(h, 0.0)

def _neg_value(params):
    """Negative objective for minimization."""
    h = _h_from_params(params)
    if np.all(h == 0):
        return 0.0
    return -_evaluate(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    initial_params_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.2],
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.5, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.4, 0.2],
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.3, 0.1],
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.15],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2],
    ]
    
    np.random.seed(42)
    for _ in range(15):
        initial_params_list.append(np.array([
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.2, 1.0),
            np.random.uniform(0.05, 0.4),
        ]))
    
    for init_params in initial_params_list:
        if time.time() > deadline:
            break
        init_params = np.asarray(init_params, dtype=np.float64)
        try:
            result = minimize(_neg_value, init_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.success or result.fun < _neg_value(init_params):
                h = _h_from_params(result.x)
                val = _evaluate(h)
                if val > best_val:
                    best_val = val
                    best_h = h.copy()
        except Exception:
            continue
    
    if time.time() < deadline - 5:
        def _neg_direct(h):
            return -_evaluate(np.maximum(h, 0.0))
        
        def _grad_direct(h):
            eps = 1e-7
            grad = np.zeros(N)
            for i in range(N):
                hp = np.maximum(h.copy(), 0.0); hp[i] += eps
                hm = np.maximum(h.copy(), 0.0); hm[i] = max(0, hm[i] - eps)
                grad[i] = (_neg_direct(hp) - _neg_direct(hm)) / (2 * eps)
            return grad
        
        try:
            result = minimize(_neg_direct, best_h, jac=_grad_direct, method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            h = np.maximum(result.x, 0.0)
            val = _evaluate(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        np.random.seed(123)
        current_h = best_h.copy()
        current_val = best_val
        for _ in range(2000):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            old = current_h[idx]
            new = max(0.0, old * (1.0 + np.random.randn() * 0.05))
            current_h[idx] = new
            v = _evaluate(current_h)
            if v > current_val:
                current_val = v
                if v > best_val:
                    best_val = v
                    best_h = current_h.copy()
            else:
                current_h[idx] = old
    
    return list(best_h)
# EVOLVE-BLOCK-END
