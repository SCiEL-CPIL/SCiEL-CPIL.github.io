import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100


def compute_value(h):
    """Exact value computation as specified."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _smooth_value(h, beta=100.0):
    """
    Smooth surrogate for the value function.
    Uses log-sum-exp to smooth max(g), and normalizes sum(g)=1 to exploit scale invariance.
    
    Since value is scale-invariant, we can fix sum(g) = 1, which means norm_1 = 1/(m+1).
    Then value = l2 / (norm_1 * norm_inf) = l2 * (m+1) / norm_inf.
    
    With sum(g)=1 fixed, we maximize l2 / max(g).
    Use log-sum-exp for smooth max.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    
    # Normalize: scale h so that sum(g) = 1
    s = np.sum(g)
    if s <= 0:
        return 0.0
    h_norm = h / np.sqrt(s)  # This makes sum(g_norm) = 1
    g_norm = np.convolve(h_norm, h_norm)
    
    # l2 with normalized g
    y = np.zeros(m + 2)
    y[1:-1] = g_norm
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    
    # norm_1 = 1/(m+1) since sum(g_norm) = 1
    norm_1 = 1.0 / (m + 1)
    
    # Smooth max using log-sum-exp
    # log-sum-exp: log(sum(exp(g_norm))) ≈ max(g_norm) for large beta
    beta_safe = max(beta, 1.0)
    max_g = np.max(g_norm)
    if max_g < -1e10:
        norm_inf = 0.0
    else:
        log_sum_exp = max_g + np.log(np.sum(np.exp((g_norm - max_g) * beta_safe)) / len(g_norm)) / beta_safe
        norm_inf = log_sum_exp
    
    if norm_inf <= 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)


def _negative_smooth_value(h, beta=100.0):
    return -_smooth_value(h, beta)


def _generate_starts():
    """Generate diverse starting points."""
    x = np.arange(N, dtype=float)
    starts = []
    
    # Uniform
    starts.append(np.ones(N))
    
    # Single Gaussian bumps
    for center in [10, 15, 20, 25, 30, 35, 40]:
        for width in [2, 3, 5, 8, 12, 15, 20]:
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            starts.append(h)
    
    # Two Gaussian bumps
    for c1 in [8, 12, 15, 18, 20]:
        for c2 in [30, 35, 38, 42]:
            for w in [3, 5, 8, 12]:
                h = np.exp(-0.5 * ((x - c1) / w) ** 2) + np.exp(-0.5 * ((x - c2) / w) ** 2)
                starts.append(h)
    
    # Cosine windows
    for center in [10, 15, 20, 25, 30, 35, 40]:
        for width in [8, 12, 15, 20, 25]:
            arg = np.pi * (x - center) / width
            h = np.cos(arg)
            starts.append(np.maximum(0.0, h))
    
    # Triangle
    for center in [10, 15, 20, 25, 30, 35, 40]:
        for hw in [5, 8, 10, 12, 15, 20]:
            h = np.maximum(0.0, 1.0 - np.abs(x - center) / hw)
            starts.append(h)
    
    # Semicircle
    for center in [15, 20, 25, 30, 35]:
        for radius in [6, 8, 10, 12, 15, 20]:
            d = (x - center) / radius
            h = np.sqrt(np.maximum(0.0, 1.0 - d ** 2))
            starts.append(h)
    
    # Power-law
    for center in [15, 20, 25, 30, 35]:
        for r in [8, 12, 15, 20]:
            for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
                base = np.maximum(0.0, 1.0 - np.abs(x - center) / r)
                h = base ** p
                starts.append(h)
    
    # Flat top with smooth edges
    for center in [20, 25, 30]:
        for flat_half in [3, 5, 8, 10]:
            for edge_width in [2, 3, 5]:
                h = np.ones(N)
                for i in range(N):
                    dist = abs(x[i] - center)
                    if dist > flat_half:
                        t = (dist - flat_half) / edge_width
                        h[i] = max(0.0, 1.0 - t)
                starts.append(h)
    
    # Random smooth starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        h = np.abs(rng.randn(N)) * np.exp(-0.5 * ((x - 25) / 10) ** 2)
        starts.append(h)
    
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Evaluate structured starts
    starts = _generate_starts()
    for s in starts:
        s = np.maximum(0.0, s)
        if np.sum(s) > 0 and time_left() > 5:
            track(s)
    
    # Phase 2: L-BFGS-B optimization with smooth surrogate
    bounds = [(0, None)] * N
    
    # Sort starts by value, take top ones for optimization
    scored = []
    for s in starts:
        s = np.maximum(0.0, s)
        if np.sum(s) > 0:
            scored.append((compute_value(s), s))
    scored.sort(key=lambda t: -t[0])
    top_starts = [s for _, s in scored[:20]]
    
    # Add current best
    if best_h is not None and best_h not in top_starts:
        top_starts.append(best_h.copy())
    
    def _lbgfs_optimize(h0, max_iter=200, beta=50.0):
        if time_left() < 5:
            return
        h0 = np.maximum(0.0, h0)
        
        def obj(h):
            if time_left() < 2:
                return 0.0
            return _negative_smooth_value(h, beta)
        
        try:
            res = opt.minimize(
                obj, h0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8,
                         'maxfun': 5000}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                track(res.x)
        except Exception:
            pass
    
    # Optimize from top starts
    for s in top_starts:
        if time_left() < 10:
            break
        _lbgfs_optimize(s, max_iter=300, beta=50.0)
    
    # Phase 3: Multi-start with different beta values
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time_left() < 15:
            break
        for s in top_starts[:8]:
            if time_left() < 5:
                break
            _lbgfs_optimize(s, max_iter=200, beta=beta)
    
    # Phase 4: Nelder-Mead on full h
    def _nm_optimize(h0, max_iter=500):
        if time_left() < 5:
            return
        h0 = np.maximum(0.0, h0)
        
        def obj(h):
            if time_left() < 2:
                return 0.0
            h = np.maximum(0.0, h)
            return -compute_value(h)
        
        try:
            res = opt.minimize(obj, h0, method='Nelder-Mead',
                               options={'maxiter': max_iter, 'xatol': 1e-6,
                                        'fatol': 1e-8, 'adaptive': True})
            if res.x is not None and np.all(np.isfinite(res.x)):
                track(res.x)
        except Exception:
            pass
    
    nm_starts = [
        best_h.copy(),
        np.ones(N),
        np.exp(-0.5 * ((np.arange(N, dtype=float) - 25) / 8) ** 2),
    ]
    for s in nm_starts:
        if time_left() < 10:
            break
        _nm_optimize(s, max_iter=800)
    
    # Phase 5: Local refinement with small perturbations
    rng = np.random.RandomState(123)
    for _ in range(30):
        if time_left() < 10:
            break
        for scale in [0.01, 0.03, 0.05, 0.1]:
            if time_left() < 5:
                break
            pert = best_h + rng.normal(0, scale, N)
            pert = np.maximum(0.0, pert)
            v = compute_value(pert)
            if v > best_val:
                best_val = v
                best_h = pert.copy()
                # Quick local optimize
                if time_left() > 10:
                    _lbgfs_optimize(pert, max_iter=100, beta=100.0)
    
    # Phase 6: Final polish with high-precision optimization
    if time_left() > 15:
        _lbgfs_optimize(best_h, max_iter=500, beta=200.0)
    
    if time_left() > 10:
        _nm_optimize(best_h, max_iter=1000)
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
