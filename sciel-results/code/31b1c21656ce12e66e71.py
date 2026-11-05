import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 400
DEADLINE = 142.0  # safety margin before 150s kill

def _value(h):
    """Compute the exact objective value."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _smooth_objective(h, beta=80.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)

def _smooth_grad(h, beta=80.0):
    """
    Gradient of the smooth objective f(h) = | 2N * softmax(conv) / s^2 |.
    Using FFT-based computation for speed.
    """
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    # dM/dh_j = 2 * sum_k w_{k+j} * h_k
    # This is a cross-correlation: (w * h)[j] = sum_k w_{k+j} h_k
    # Use FFT for efficiency
    # Pad to length 2N-1
    n_fft = 2 * N - 1
    w_padded = np.zeros(n_fft)
    w_padded[:len(weights)] = weights
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    
    # Cross-correlation via FFT: correlate(w, h) where (w*h)[j] = sum_k w_{k+j}h_k
    # This is np.correlate(w, h, mode='valid') for j=0..N-1
    # Using FFT: ifft(fft(w_padded) * conj(fft(h_padded)))
    W = np.fft.rfft(w_padded, n=n_fft)
    H = np.fft.rfft(h_padded, n=n_fft)
    corr = np.fft.irfft(W * np.conj(H), n=n_fft)
    d_soft_max = 2.0 * corr[:N]
    
    # Objective: f = 2N * soft_max / s^2 (assuming positive)
    inner = 2.0 * N * soft_max / (s * s)
    sign = 1.0 if inner >= 0 else -1.0
    
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    obj = abs(inner)
    return grad, obj

def _diverse_init(max_time=12.0):
    """Generate diverse initial candidates."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
        for amp in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Sine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        for amp in [0.5, 1.0, 2.0, 3.0]:
            h = np.ones(N) + amp * np.sin(2 * np.pi * k * x / N)
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
    for nblocks in [2, 4, 8, 16, 32, 64]:
        block_size = N // nblocks
        if block_size < 1:
            continue
        for seed in range(4):
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
        if time.time() - t0 > max_time * 0.7:
            break
        for trial in range(5):
            freqs = sorted(rng.uniform(1, 60, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.5, n_freq)
            dc = rng.uniform(-0.2, 0.2)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Alternating signs
    for period in [2, 4, 8, 16, 32]:
        h = np.ones(N)
        h[::2] = -0.5
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

def _local_refine(h0, max_time=20.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [40.0, 80.0, 150.0, 300.0]:
        if time.time() - t0 > max_time * 0.85:
            break
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _multi_start(h_best, n_starts=6, max_time=25.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.05 + 0.03 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=100.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=100.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _final_polish(h_best, max_time=15.0):
    """Final high-precision polish with very high beta."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    
    for beta in [300.0, 500.0, 800.0]:
        if time.time() - t0 > max_time * 0.85:
            break
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 600, 'maxfun': 1200, 'ftol': 1e-16, 'gtol': 1e-15}
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
    
    # Phase 2: Local refinement
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(20.0, time_left() - 25.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 30:
        best_h, best_val = _multi_start(best_h, n_starts=6, max_time=min(25.0, time_left() - 20.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 15:
        best_h, best_val = _final_polish(best_h, max_time=min(15.0, time_left() - 10.0))
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
