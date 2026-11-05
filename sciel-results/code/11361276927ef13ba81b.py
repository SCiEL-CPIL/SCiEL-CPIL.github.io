import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=30.0):
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


def _smooth_grad(h, beta=30.0):
    """Gradient of the smooth objective."""
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
    obj = abs(2.0 * N * soft_max / (s * s))
    
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    d_soft_max = np.zeros(N)
    for j in range(N):
        k_start = j
        k_end = min(j + N - 1, 2 * N - 2)
        if k_start > k_end:
            continue
        d_soft_max[j] = 2.0 * np.dot(weights[k_start:k_end+1], h[0:k_end-k_start+1])
    
    sign = np.sign(2.0 * N * soft_max / (s * s))
    if sign == 0:
        sign = 1.0
    
    grad = np.zeros(N)
    for j in range(N):
        grad[j] = sign * 2.0 * N * (d_soft_max[j] / (s * s) - 2.0 * soft_max / (s ** 3))
    
    return grad, obj


def _chirp_h(freq, chirp, dc):
    """Generate chirp signal: h[i] = dc + cos(2*pi*freq*i/N + 0.5*chirp*(i/N)^2)"""
    x = np.arange(N, dtype=float)
    t = x / N
    phase = 2 * np.pi * freq * t + 0.5 * chirp * t * t
    h = dc + np.cos(phase)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return None
    return h / s


def _chirp_grid_search(max_time=20.0):
    """Grid search over chirp parameters."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    # Grid over frequency, chirp rate, and DC bias
    freqs = np.arange(1, 80, 1)
    chirps = np.arange(-200, 201, 5)
    dcs = np.linspace(-0.3, 0.3, 13)
    
    for freq in freqs:
        if time.time() - t0 > max_time * 0.7:
            break
        for chirp in chirps:
            if time.time() - t0 > max_time * 0.9:
                break
            for dc in dcs:
                h = _chirp_h(freq, chirp, dc)
                if h is None:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _chirp_refine(h0, max_time=15.0):
    """Refine chirp parameters using local optimization."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Try L-BFGS-B on the full h vector starting from chirp init
    try:
        def obj_fn(h):
            return _smooth_objective(h, beta=40.0)
        
        def grad_fn(h):
            g, _ = _smooth_grad(h, beta=40.0)
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
    
    if time.time() - t0 < max_time * 0.6:
        try:
            def obj_fn2(h):
                return _smooth_objective(h, beta=80.0)
            
            def grad_fn2(h):
                g, _ = _smooth_grad(h, beta=80.0)
                return g
            
            res = minimize(
                obj_fn2,
                best_h,
                jac=grad_fn2,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _diverse_init(max_time=20.0):
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
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Chirp signals
    for freq in [5, 10, 20, 30, 50, 75]:
        for chirp in [-100, -50, 0, 50, 100]:
            h = _chirp_h(freq, chirp, 0.0)
            if h is not None:
                candidates.append(h)
    
    # Random Gaussian
    for trial in range(20):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(5):
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
    
    for beta in [30.0, 50.0, 80.0]:
        if time.time() - t0 > max_time * 0.8:
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
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=8, max_time=30.0):
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
            res = minimize(
                lambda h: _smooth_objective(h, beta=40.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=40.0)[0],
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
    
    # Phase 1: Chirp grid search
    h_chirp, val_chirp = _chirp_grid_search(max_time=15.0)
    
    # Phase 2: Diverse initialization
    h_div, val_div = _diverse_init(max_time=15.0)
    
    # Pick best
    if val_chirp <= val_div:
        best_h = h_chirp.copy()
        best_val = val_chirp
    else:
        best_h = h_div.copy()
        best_val = val_div
    
    # Include seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Phase 3: Chirp-specific refinement
    if time_left() > 20:
        best_h, best_val = _chirp_refine(best_h, max_time=min(15.0, time_left() - 10.0))
    
    # Phase 4: General local refinement
    if time_left() > 25:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 10.0))
    
    # Phase 5: Multi-start
    if time_left() > 20:
        best_h, best_val = _multi_start(best_h, n_starts=10, max_time=min(30.0, time_left() - 10.0))
    
    # Phase 6: Final high-precision polish
    if time_left() > 10:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=100.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=100.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
