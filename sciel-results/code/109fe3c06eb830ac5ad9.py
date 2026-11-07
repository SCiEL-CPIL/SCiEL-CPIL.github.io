import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _value(h):
    """Exact objective value (nonsmooth due to max)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf <= 0 or norm_1 <= 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _soft_max(g, beta=100.0):
    """Smooth approximation of max(g) via log-sum-exp."""
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    # log-sum-exp trick for numerical stability
    shifted = g / g_max
    log_sum = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
    return g_max * np.exp(log_sum)

def _neg_value_smooth(h, beta=100.0):
    """Negative of smooth surrogate objective for minimization."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf_soft = _soft_max(g, beta)
    if norm_inf_soft <= 0 or norm_1 <= 0:
        return 1e10
    return -(l2 / (norm_1 * norm_inf_soft))

def _neg_value_grad(h, beta=100.0):
    """Negative smooth objective with numerical gradient."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    f = _neg_value_smooth(h, beta)
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] = max(0.0, h[i] + eps)
        grad[i] = (_neg_value_smooth(h_plus, beta) - f) / eps
    return f, grad

def _normalize(h):
    s = h.sum()
    if s <= 0:
        return np.ones(N) / N
    return h / s

def _construct_hump(power=1.0, shift=0.5, width=1.0, flat_top=0.0):
    x = np.linspace(0, 1, N)
    t = np.abs(x - shift) / (0.5 * width)
    t = np.clip(t, 0, 1)
    if power <= 0:
        f = np.ones(N)
    else:
        f = (1.0 - t**power) ** 1.0
    if flat_top > 0:
        mask = t < flat_top
        f = np.where(mask, 1.0, f * (1.0 - flat_top))
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def _local_search(h0, max_iters=5000, tol=1e-9):
    """Coordinate ascent on shape (sum=1 normalization after each step)."""
    h = np.array(h0, dtype=float)
    h = _normalize(h)
    cur_val = _value(h)
    if cur_val < 0:
        return h, cur_val
    step = 0.05
    for it in range(max_iters):
        improved = False
        for i in range(N):
            if h[i] <= 1e-12:
                continue
            for sign in [1.0, -1.0]:
                h_try = h.copy()
                h_try[i] = max(0.0, h[i] + sign * step)
                h_try = _normalize(h_try)
                v = _value(h_try)
                if v > cur_val + 1e-12:
                    h = h_try
                    cur_val = v
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-4:
                break
        if it % 20 == 0:
            for i in range(N):
                if h[i] <= 1e-12:
                    h_try = h.copy()
                    h_try[i] = step * 0.1
                    h_try = _normalize(h_try)
                    v = _value(h_try)
                    if v > cur_val + 1e-12:
                        h = h_try
                        cur_val = v
    return h, cur_val

def _lbfgsb_optimize(h0, beta=100.0, maxiter=200):
    """Optimize using L-BFGS-B on the smooth surrogate."""
    h = np.array(h0, dtype=np.float64)
    h = _normalize(h)
    
    def objective(x):
        return _neg_value_smooth(x, beta)
    
    def gradient(x):
        f, g = _neg_value_grad(x, beta)
        return g
    
    try:
        result = minimize(
            objective, h, method='L-BFGS-B', jac=gradient,
            bounds=[(0.0, None)] * N,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        h_opt = _normalize(result.x)
        v = _value(h_opt)
        if v > 0:
            return h_opt, v
    except Exception:
        pass
    return h, _value(h)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    best_h = None
    best_val = -1.0

    # 1) Seed
    seed_h = np.ones(N)
    seed_val = _value(seed_h)
    if seed_val > best_val:
        best_val = seed_val
        best_h = seed_h.copy()

    # 2) Analytic initializations with L-BFGS-B + local search
    params = []
    for power in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        params.append((power, 0.5, 1.0, 0.0))
    for flat in [0.1, 0.2, 0.3, 0.4]:
        for power in [2.0, 3.0, 4.0]:
            params.append((power, 0.5, 1.0, flat))
    for power in [2.0, 3.0, 4.0]:
        params.append((power, 0.5, 0.9, 0.0))
        params.append((power, 0.5, 0.8, 0.0))

    for power, shift, width, flat in params:
        if time.time() > deadline - 10.0:
            break
        h_init = _construct_hump(power, shift, width, flat)
        
        # L-BFGS-B with smooth surrogate
        h_lbf, v_lbf = _lbfgsb_optimize(h_init, beta=100.0, maxiter=150)
        if v_lbf > best_val:
            best_val = v_lbf
            best_h = h_lbf.copy()
        
        # Polish with coordinate ascent on exact objective
        h_polish, v_polish = _local_search(h_lbf, max_iters=2000)
        if v_polish > best_val:
            best_val = v_polish
            best_h = h_polish.copy()

    # 3) Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        x = np.linspace(0, 1, N)
        center = rng.uniform(0.35, 0.65)
        width = rng.uniform(0.7, 1.0)
        power = rng.uniform(1.5, 5.0)
        base = (1.0 - np.clip(np.abs(x - center) / (0.5 * width), 0, 1)) ** power
        noise = rng.random(N) * 0.15
        h_init = base + noise
        h_init = np.maximum(h_init, 0.0)
        if h_init.sum() <= 0:
            h_init = np.ones(N)
        h_init = h_init / h_init.sum()
        
        h_lbf, v_lbf = _lbfgsb_optimize(h_init, beta=100.0, maxiter=100)
        if v_lbf > best_val:
            best_val = v_lbf
            best_h = h_lbf.copy()
        
        h_polish, v_polish = _local_search(h_lbf, max_iters=1500)
        if v_polish > best_val:
            best_val = v_polish
            best_h = h_polish.copy()

    # 4) Final high-precision polish
    if best_h is not None and time.time() < deadline - 15.0:
        # Try with higher beta for smoother max approximation
        h_high_beta, v_high_beta = _lbfgsb_optimize(best_h, beta=200.0, maxiter=300)
        if v_high_beta > best_val:
            best_val = v_high_beta
            best_h = h_high_beta.copy()
        
        # Final coordinate ascent with very small steps
        h_final, v_final = _local_search(best_h, max_iters=5000)
        if v_final > best_val:
            best_val = v_final
            best_h = h_final

    if best_h is None:
        best_h = np.ones(N)

    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
