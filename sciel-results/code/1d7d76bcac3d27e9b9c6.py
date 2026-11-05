"""
Combined program: direct-space smooth optimization with diverse init and multi-beta L-BFGS-B.
Uses the structural idea of direct h-space optimization (inspiration 1) with the
gradient machinery and Chebyshev-inspired initialization from the current program.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    """Compute the task value."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the max of autoconvolution.
    Returns the scalar objective value.
    """
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=50.0):
    """
    Gradient of the smooth objective f(h) = | 2N * softmax(conv) / s^2 |.
    Returns (grad, obj) tuple for L-BFGS-B.
    
    d(soft_max)/dh_j = 2 * sum_{k=0}^{N-1} w_{k+j} * h_k
    where w = exp(beta*(conv-m)) / sum(exp(beta*(conv-m)))
    
    This is 2 * np.correlate(weights, h, mode='valid')
    
    f = 2N * M / s^2  (assuming positive)
    df/dh_j = 2N * ( dM/dh_j / s^2 - 2*M/s^3 )
    """
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    total_w = np.sum(weights)
    if total_w < 1e-300:
        weights = np.ones_like(conv) / len(conv)
    else:
        weights /= total_w
    soft_max = m + np.log(total_w) / beta
    
    # d(soft_max)/dh_j = 2 * correlate(weights, h, mode='valid')
    # correlate(w, h, 'valid')[j] = sum_k w[j+k] * h[k] for j=0..N-1
    d_soft_max = 2.0 * np.correlate(weights, h, mode='valid')
    
    # Check sign
    inner = 2.0 * N * soft_max / (s * s)
    sign = 1.0 if inner >= 0 else -1.0
    
    # df/dh_j = sign * 2N * ( dM/dh_j / s^2 - 2*M / s^3 )
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    obj = abs(inner)
    return grad, obj


def _diverse_init(max_time=15.0):
    """Generate diverse initial candidates and return the best."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(10):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(3):
            if time.time() - t0 > max_time * 0.8:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Multi-frequency cosine
    for n_freq in range(2, 5):
        if time.time() - t0 > max_time * 0.85:
            break
        for trial in range(5):
            freqs = sorted(rng.uniform(1, 40, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.0, n_freq)
            dc = rng.uniform(-0.1, 0.1)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [30.0, 60.0, 100.0, 200.0]:
        if time.time() - t0 > max_time * 0.85:
            break
        try:
            def obj_fn(h, b=beta):
                return _smooth_objective(h, beta=b)
            
            def grad_fn(h, b=beta):
                g, _ = _smooth_grad(h, beta=b)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=5, max_time=20.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.1 + 0.05 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            def obj_fn(h, b=50.0):
                return _smooth_objective(h, beta=b)
            
            def grad_fn(h, b=50.0):
                g, _ = _smooth_grad(h, beta=b)
                return g
            
            res = minimize(
                obj_fn,
                h_init,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Diverse initialization
    h_div, val_div = _diverse_init(max_time=15.0)
    
    best_h = h_div.copy()
    best_val = val_div
    
    # Phase 2: Local refinement
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 25.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 20:
        best_h, best_val = _multi_start(best_h, n_starts=5, max_time=min(20.0, time_left() - 15.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 10:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=200.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=200.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-15:
        return np.ones(N)
    
    return best_h


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
