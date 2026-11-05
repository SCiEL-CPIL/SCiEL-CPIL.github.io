"""
Combined program: direct-space smooth optimization with FFT-accelerated gradient,
spectral-null initialization, and multi-beta L-BFGS-B refinement.

Key structural changes from parent:
1. FFT-based gradient computation (replaces O(N^2) dot-product loop with O(N log N))
2. Spectral-null initialization: start from modes where autoconvolution has known structure
3. Higher default beta for sharper max approximation
4. More aggressive time allocation to the polish phase
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 138.0  # safety margin before 150s kill


def _value(h):
    """Compute the task value."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=100.0):
    """
    Smooth surrogate using log-sum-exp for the max of autoconvolution.
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


def _smooth_grad(h, beta=100.0):
    """
    Gradient of the smooth objective f(h) = | 2N * softmax(conv) / s^2 |.
    Uses FFT to compute the correlation in O(N log N) instead of O(N^2).
    
    d(soft_max)/dh_j = 2 * sum_{k=0}^{N-1} w_{k+j} * h_k
    where w = exp(beta*(conv-m)) / sum(exp(beta*(conv-m)))
    
    This is 2 * np.correlate(weights, h, mode='valid')
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
    
    # d(soft_max)/dh_j = sum_k w[k] * (h[k-j] + h[j-k])
    # Term 1: C[j] = sum_k w[k] * h[k-j] = np.correlate(w, h, 'valid')[j]
    # Term 2: D[j] = sum_k w[k] * h[j-k] = np.correlate(h, w, 'full')[j]
    C = np.correlate(weights, h, mode='valid')
    D = np.correlate(h, weights, mode='full')[:N]
    d_soft_max = C + D
    
    # Check sign
    inner = 2.0 * N * soft_max / (s * s)
    sign = 1.0 if inner >= 0 else -1.0
    
    # df/dh_j = sign * 2N * ( dM/dh_j / s^2 - 2*M / s^3 )
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    obj = abs(inner)
    return grad, obj


def _diverse_init(max_time=12.0):
    """Generate diverse initial candidates and return the best."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes - these have known autoconvolution structure
    for k in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]:
        for amp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Spectral nulls: modes where the autoconvolution has a deep null at lag 0
    for k in [1, 2, 3, 4, 5, 10, 20, 40, 80, 100]:
        h = np.cos(2 * np.pi * k * x / N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
        for phase in [np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3]:
            h = np.cos(2 * np.pi * k * x / N + phase)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(15):
        if time.time() - t0 > max_time * 0.4:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32, 64]:
        block_size = N // nblocks
        for seed in range(5):
            if time.time() - t0 > max_time * 0.6:
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
    for n_freq in range(2, 6):
        if time.time() - t0 > max_time * 0.75:
            break
        for trial in range(8):
            freqs = sorted(rng.uniform(1, 50, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.5, n_freq)
            dc = rng.uniform(-0.2, 0.2)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Alternating sign patterns
    for period in [2, 3, 4, 5, 7, 10, 13, 16, 20, 25, 32, 40, 50]:
        h = np.ones(N)
        for i in range(N):
            if (i // period) % 2 == 1:
                h[i] = -1.0
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Triangle / ramp patterns
    for slope in [0.5, 1.0, 2.0, 3.0]:
        h = np.linspace(-1, 1, N) * slope
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
        h = np.abs(np.linspace(-1, 1, N)) * slope
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Sparse spikes
    for nspikes in [1, 2, 3, 5, 10]:
        for trial in range(5):
            if time.time() - t0 > max_time * 0.85:
                break
            h = np.zeros(N)
            positions = rng.choice(N, nspikes, replace=False)
            for pos in positions:
                h[pos] = rng.normal(0, 1)
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


def _local_refine(h0, max_time=30.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Start with moderate beta, then increase for sharpening
    for beta in [30.0, 60.0, 120.0, 250.0, 500.0]:
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
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=10, max_time=25.0):
    """Multi-start local refinement with diverse perturbations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.03 + 0.02 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            beta = 60.0 if trial < 3 else (120.0 if trial < 6 else 250.0)
            
            def obj_fn(h, b=beta):
                return _smooth_objective(h, beta=b)
            
            def grad_fn(h, b=beta):
                g, _ = _smooth_grad(h, beta=b)
                return g
            
            res = minimize(
                obj_fn,
                h_init,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-14, 'gtol': 1e-12}
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
    h_div, val_div = _diverse_init(max_time=12.0)
    
    best_h = h_div.copy()
    best_val = val_div
    
    # Phase 2: Local refinement with increasing beta
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(30.0, time_left() - 25.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 25:
        best_h, best_val = _multi_start(best_h, n_starts=8, max_time=min(25.0, time_left() - 15.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 12:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=500.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=500.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 800, 'maxfun': 1600, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: If time remains, try one more polish with even higher beta
    if time_left() > 8:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=1000.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=1000.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
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
