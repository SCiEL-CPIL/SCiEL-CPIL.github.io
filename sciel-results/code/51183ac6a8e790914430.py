import numpy as np
import time

N = 50
_start_time = time.time()
_TIMEOUT = 140.0  # safety margin before 150s kill

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

def _random_init():
    """Generate a random nonnegative vector."""
    # Mix of sparse and dense patterns
    if np.random.random() < 0.3:
        # Sparse: few nonzero entries
        k = np.random.randint(2, 15)
        idx = np.random.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.exponential(1.0, size=k)
    else:
        # Dense: all entries positive with some structure
        h = np.random.exponential(1.0, size=N)
        # Optionally apply a smooth envelope
        if np.random.random() < 0.5:
            x = np.linspace(-1, 1, N)
            sigma = np.random.uniform(0.3, 1.5)
            h *= np.exp(-x**2 / (2 * sigma**2))
    return h

def _local_optimize(h0, max_iters=2000):
    """Gradient-free local optimization using coordinate ascent / random perturbation."""
    h = h0.copy()
    h = np.maximum(h, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    
    # Use Nelder-Mead-like approach or simple random restarts with perturbation
    step_size = 0.1
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
            step_size = max(step_size * 1.05, 0.01)
        else:
            step_size = max(step_size * 0.95, 0.001)
    
    return best_h, best_score

def _try_structural_patterns():
    """Try known good structural patterns."""
    candidates = []
    
    # 2-point cluster with optimal ratio
    # For [a, b], ratio = (a²+b²)/(max(a², 2ab, b²)) with a+b=1
    # Optimize: try various ratios
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
    
    return candidates

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
    h_opt, s_opt = _local_optimize(best_h, max_iters=5000)
    if s_opt > best_score:
        best_score = s_opt
        best_h = h_opt
    
    # Multiple random restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - _start_time > _TIMEOUT:
            break
        h_rand = _random_init()
        h_opt, s_opt = _local_optimize(h_rand, max_iters=3000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Final polish with smaller steps
    h_final, s_final = _local_optimize(best_h, max_iters=3000)
    if s_final > best_score:
        best_score = s_final
        best_h = h_final
    
    return list(best_h)
