import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 150.0
_start = time.time()

def _time_left():
    return _DEADLINE - 5.0 - (time.time() - _start)

def _autoconv(h):
    """Discrete autoconvolution: (h*h)[k] = sum_j h[j]*h[k-j] for valid k"""
    # Full convolution gives indices 0..2N-2
    # The "center" is at index N-1
    return np.convolve(h, h, mode='full')

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0 or l2_sq <= 0:
        return 0.0
    ac = _autoconv(h)
    sup = np.max(ac)
    if sup <= 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _neg_ratio(h):
    return -_ratio(h)

def _ratio_and_grad(h):
    """Compute ratio and gradient via finite differences (sparse)."""
    h = np.asarray(h, dtype=np.float64)
    r = _ratio(h)
    if r <= 0:
        return r, np.zeros_like(h)
    
    # Use central differences for gradient
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] = max(h[i] - eps, 0.0)
        grad[i] = (_ratio(h_plus) - _ratio(h_minus)) / (h_plus[i] - h_minus[i] + 1e-30)
    
    return r, grad

def _local_optimize(h0, max_iter=200, method='L-BFGS-B'):
    """Local optimization from h0."""
    h0 = np.maximum(np.asarray(h0, dtype=np.float64), 0.0)
    
    def obj(x):
        if _time_left() <= 0:
            return 1e10
        return -_ratio(np.maximum(x, 0.0))
    
    try:
        res = minimize(obj, h0, method=method,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxls': 50})
        h_opt = np.maximum(res.x, 0.0)
        if res.fun < 0:  # means ratio > 0
            return h_opt
    except Exception:
        pass
    return None

def _coordinate_descent(h, max_iters=500):
    """Simple coordinate descent for refinement."""
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0).copy()
    best_r = _ratio(h)
    
    for it in range(max_iters):
        if _time_left() <= 0:
            break
        improved = False
        for i in range(N):
            # Try a few perturbations of coordinate i
            h_old = h[i]
            candidates = [h_old * 0.5, h_old * 0.75, h_old * 1.0, h_old * 1.25, h_old * 1.5,
                         h_old + 0.01, h_old - 0.01, 0.0, h_old * 0.25]
            candidates = [max(c, 0.0) for c in candidates]
            best_c = h_old
            best_cr = _ratio(h)
            for c in candidates:
                h[i] = c
                r = _ratio(h)
                if r > best_cr:
                    best_cr = r
                    best_c = c
            h[i] = best_c
            if best_c != h_old:
                improved = True
        
        if not improved:
            break
        if _ratio(h) > best_r:
            best_r = _ratio(h)
    
    return h

def _generate_starts(rng):
    """Generate diverse initial candidates."""
    starts = []
    
    # 1. Power law on one side (asymmetric)
    t = np.linspace(0, 1, N)
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        h = (1 - t) ** alpha
        h /= np.max(h)
        starts.append(h.copy())
        # Also reversed
        starts.append(h[::-1].copy())
    
    # 2. Power law with offset
    for alpha in [1.0, 2.0, 3.0]:
        for offset in [0.1, 0.2, 0.3, 0.5]:
            h = np.maximum(1 - t - offset, 0) ** alpha
            if np.max(h) > 0:
                h /= np.max(h)
            starts.append(h.copy())
            starts.append(h[::-1].copy())
    
    # 3. Gaussian (narrow and wide)
    for sigma in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        for center in [0.2, 0.3, 0.5, 0.7, 0.8]:
            h = np.exp(-0.5 * ((t - center) / sigma) ** 2)
            starts.append(h.copy())
    
    # 4. Lorentzian
    for gamma in [0.02, 0.05, 0.1, 0.2, 0.3]:
        for center in [0.2, 0.5, 0.8]:
            h = gamma / (gamma**2 + (t - center)**2)
            h /= np.max(h)
            starts.append(h.copy())
    
    # 5. Tapered plateau (raised cosine)
    for width in [0.3, 0.5, 0.7]:
        for center in [0.3, 0.5, 0.7]:
            half = width / 2
            h = np.zeros(N)
            mask = (t >= center - half) & (t <= center + half)
            # Raised cosine taper
            d = np.abs(t - center)
            taper = np.where(d < half, 0.5 * (1 + np.cos(np.pi * d / half)), 0)
            h = taper
            if np.max(h) > 0:
                h /= np.max(h)
            starts.append(h.copy())
    
    # 6. Linear ramp
    h = np.linspace(1, 0, N)
    starts.append(h.copy())
    starts.append(h[::-1].copy())
    
    # 7. Quadratic
    h = (1 - t) ** 2
    starts.append(h.copy())
    starts.append(h[::-1].copy())
    
    # 8. Random smooth
    for _ in range(10):
        k = np.arange(N)
        coeffs = rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)
        # Keep only low frequencies for smoothness
        n_low = rng.integers(3, 10)
        mask = np.zeros(N, dtype=bool)
        mask[:n_low] = True
        mask[-n_low:] = True
        coeffs[~mask] = 0
        h = np.real(np.fft.ifft(coeffs))
        h = np.maximum(h, 0)
        if np.max(h) > 0:
            h /= np.max(h)
        starts.append(h.copy())
    
    # 9. Sparse peaks
    for _ in range(5):
        h = np.zeros(N)
        n_peaks = rng.integers(1, 4)
        for _ in range(n_peaks):
            pos = rng.integers(0, N)
            width = rng.integers(1, 5)
            h[max(0, pos-width):pos+width+1] += 1.0
        if np.max(h) > 0:
            h /= np.max(h)
        starts.append(h.copy())
    
    return starts

def construct():
    global _start
    _start = time.time()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starts
    starts = _generate_starts(rng)
    
    best_h = None
    best_score = 0.0
    
    # Evaluate all starts
    for h in starts:
        if _time_left() <= 0:
            break
        r = _ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()
    
    # Local optimization from top candidates
    scored = [(_ratio(h), i) for i, h in enumerate(starts)]
    scored.sort(reverse=True)
    
    n_optimize = min(20, len(scored))
    for score, idx in scored[:n_optimize]:
        if _time_left() <= 10:
            break
        
        h0 = starts[idx].copy()
        
        # Try L-BFGS-B
        h_opt = _local_optimize(h0, max_iter=100)
        if h_opt is not None:
            r = _ratio(h_opt)
            if r > best_score:
                best_score = r
                best_h = h_opt.copy()
        
        # Coordinate descent refinement
        if _time_left() > 5:
            h_cd = _coordinate_descent(h_opt if h_opt is not None else h0, max_iters=200)
            r_cd = _ratio(h_cd)
            if r_cd > best_score:
                best_score = r_cd
                best_h = h_cd.copy()
    
    # Additional random perturbations of best
    if _time_left() > 15 and best_h is not None:
        for _ in range(20):
            if _time_left() <= 5:
                break
            h_pert = best_h.copy()
            # Random multiplicative perturbation
            noise = rng.uniform(0.5, 1.5, N)
            h_pert = h_pert * noise
            h_pert = np.maximum(h_pert, 0.0)
            
            h_opt = _local_optimize(h_pert, max_iter=80)
            if h_opt is not None:
                r = _ratio(h_opt)
                if r > best_score:
                    best_score = r
                    best_h = h_opt.copy()
            
            # Also try coordinate descent
            if _time_left() > 3:
                h_cd = _coordinate_descent(h_pert, max_iters=100)
                r_cd = _ratio(h_cd)
                if r_cd > best_score:
                    best_score = r_cd
                    best_h = h_cd.copy()
    
    # Final refinement on best
    if _time_left() > 10 and best_h is not None:
        h_final = _local_optimize(best_h, max_iter=200)
        if h_final is not None:
            r = _ratio(h_final)
            if r > best_score:
                best_score = r
                best_h = h_final.copy()
        
        if _time_left() > 5:
            h_cd = _coordinate_descent(best_h, max_iters=300)
            r_cd = _ratio(h_cd)
            if r_cd > best_score:
                best_score = r_cd
                best_h = h_cd.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)
