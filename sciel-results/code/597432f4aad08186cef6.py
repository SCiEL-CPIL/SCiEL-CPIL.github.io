import numpy as np
import time
from scipy.optimize import minimize

N = 50
_start_time = time.time()
_TIMEOUT = 135.0  # safety margin before 150s kill

def _score(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Full linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _score_smooth(h, beta=50.0):
    """
    Smooth surrogate for the score using log-sum-exp to approximate max in autoconvolution.
    This makes the objective differentiable, enabling gradient-based optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    conv = np.convolve(h, h, mode='full')
    
    # Smooth max via log-sum-exp: soft_max = (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, subtract max first
    m = np.max(conv)
    if m < 1e-15:
        return 0.0
    
    # log-sum-exp: m + log(sum(exp(conv - m)))
    # soft_max = (1/beta) * (beta*m + log(sum(exp(beta*(conv-m)))))
    #          = m + (1/beta)*log(sum(exp(beta*(conv-m))))
    shifted = beta * (conv - m)
    # Avoid overflow: clip shifted to reasonable range
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    conv_inf_smooth = m + log_sum / beta
    
    if conv_inf_smooth < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf_smooth)

def _score_neg_smooth(h, beta=50.0):
    """Negative smooth score for minimization."""
    return -_score_smooth(h, beta)

def _random_init():
    """Generate a random nonnegative vector."""
    if np.random.random() < 0.3:
        # Sparse: few nonzero entries
        k = np.random.randint(2, 15)
        idx = np.random.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.exponential(1.0, size=k)
    else:
        # Dense: all entries positive with some structure
        h = np.random.exponential(1.0, size=N)
        if np.random.random() < 0.5:
            x = np.linspace(-1, 1, N)
            sigma = np.random.uniform(0.3, 1.5)
            h *= np.exp(-x**2 / (2 * sigma**2))
    return h

def _local_optimize_smooth(h0, max_iters=300, beta=50.0):
    """
    Gradient-based local optimization using smooth surrogate.
    Uses scipy's L-BFGS-B with bounds to handle nonnegativity.
    """
    h = np.maximum(h0, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    
    # Use L-BFGS-B to minimize negative smooth score
    bounds = [(0.0, None)] * N
    
    def neg_smooth(x):
        return _score_neg_smooth(x, beta)
    
    # Run optimization
    try:
        result = minimize(neg_smooth, h, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
        h_opt = np.maximum(result.x, 0.0)
        s_opt = _score(h_opt)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    except Exception:
        pass
    
    return best_h, best_score

def _local_optimize_hybrid(h0, max_iters=2000):
    """
    Combined local optimization: smooth gradient ascent + random perturbation refinement.
    """
    h = h0.copy()
    h = np.maximum(h, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    
    # Phase 1: Gradient ascent with smooth surrogate
    if time.time() - _start_time < _TIMEOUT:
        h_grad, s_grad = _local_optimize_smooth(h, max_iters=200, beta=50.0)
        if s_grad > best_score:
            best_score = s_grad
            best_h = h_grad.copy()
            h = h_grad.copy()
    
    # Phase 2: Random perturbation refinement
    step_size = 0.05
    for i in range(max_iters):
        if time.time() - _start_time > _TIMEOUT:
            break
        # Try random perturbation
        h_trial = h + np.random.normal(0, step_size, size=N)
        h_trial = np.maximum(h_trial, 0.0)
        s_trial = _score(h_trial)
        if s_trial > best_score:
            h = h_trial
            best_score = s_trial
            best_h = h.copy()
            step_size = max(step_size * 1.03, 0.005)
        else:
            step_size = max(step_size * 0.95, 0.001)
    
    return best_h, best_score

def _try_structural_patterns():
    """Try known good structural patterns."""
    candidates = []
    
    # 2-point cluster with optimal ratio
    for r in np.linspace(0.5, 4.0, 100):
        a = r / (1 + r)
        b = 1 / (1 + r)
        h = np.zeros(N)
        h[0] = a
        h[1] = b
        candidates.append((h, _score(h)))
    
    # 3-point clusters
    for a in np.linspace(0.1, 0.8, 30):
        for b in np.linspace(0.1, 0.8, 30):
            c = 1 - a - b
            if c < 0.01:
                continue
            h = np.zeros(N)
            h[0] = a
            h[1] = b
            h[2] = c
            candidates.append((h, _score(h)))
    
    # Spread out: every other point
    h = np.zeros(N)
    h[::2] = 1.0 / 25.0
    candidates.append((h, _score(h)))
    
    # Spread out: every k-th point
    for k in [3, 4, 5, 7]:
        h = np.zeros(N)
        h[::k] = 1.0 / (N // k)
        candidates.append((h, _score(h)))
    
    # Bell-shaped spread
    x = np.linspace(-1, 1, N)
    for sigma in [0.5, 0.8, 1.0, 1.5, 2.0]:
        h = np.exp(-x**2 / (2 * sigma**2))
        h /= np.sum(h)
        candidates.append((h, _score(h)))
    
    # Two separated clusters
    for gap in range(5, 30):
        for r in [1.0, 1.5, 2.0, 3.0]:
            h = np.zeros(N)
            h[0] = r / (1 + r)
            h[1] = 1 / (1 + r)
            h[gap] = r / (1 + r)
            h[gap + 1] = 1 / (1 + r)
            h /= np.sum(h)
            candidates.append((h, _score(h)))
    
    # Four-point clusters at various spacings
    for spacing in [1, 2, 3, 5]:
        for r in [0.5, 1.0, 1.5, 2.0]:
            h = np.zeros(N)
            a = r / (1 + r)
            b = 1 / (1 + r)
            h[0] = a
            h[spacing] = b
            h[2*spacing] = a
            h[3*spacing] = b
            h /= np.sum(h)
            candidates.append((h, _score(h)))
    
    return candidates

def _polish(h0, max_iters=500):
    """
    Final high-precision polish pass with tighter tolerances.
    Uses smooth surrogate with higher beta for sharper max approximation.
    """
    h = np.maximum(h0, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    
    # Use higher beta for sharper approximation
    h_grad, s_grad = _local_optimize_smooth(h, max_iters=max_iters, beta=100.0)
    if s_grad > best_score:
        best_score = s_grad
        best_h = h_grad.copy()
    
    # Very fine random perturbation
    step_size = 0.005
    for i in range(max_iters):
        if time.time() - _start_time > _TIMEOUT:
            break
        h_trial = h + np.random.normal(0, step_size, size=N)
        h_trial = np.maximum(h_trial, 0.0)
        s_trial = _score(h_trial)
        if s_trial > best_score:
            h = h_trial
            best_score = s_trial
            best_h = h.copy()
            step_size = max(step_size * 1.02, 0.001)
        else:
            step_size = max(step_size * 0.97, 0.0005)
    
    return best_h, best_score

def construct():
    """Construct 50 nonnegative heights maximizing ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    global _start_time
    _start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try structural patterns
    patterns = _try_structural_patterns()
    for h, s in patterns:
        if time.time() - _start_time > _TIMEOUT:
            break
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best structural pattern
    if time.time() - _start_time < _TIMEOUT:
        h_opt, s_opt = _local_optimize_hybrid(best_h, max_iters=2000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Multiple random restarts with smooth-based optimization
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() - _start_time > _TIMEOUT:
            break
        h_rand = _random_init()
        h_opt, s_opt = _local_optimize_hybrid(h_rand, max_iters=1500)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Final polish with high-precision settings
    if time.time() - _start_time < _TIMEOUT - 10:  # leave 10s buffer
        h_final, s_final = _polish(best_h, max_iters=300)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    return list(best_h)
