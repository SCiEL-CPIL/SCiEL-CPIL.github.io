import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)
TIME_LIMIT = 150.0

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
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
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _neg_value_normalized(h):
    h = np.maximum(0.0, h)
    if np.max(h) < 1e-12:
        return 0.0
    h = h / np.max(h)
    return -compute_value(h)

def _plateau_start(center, width, height=1.0):
    """Create a plateau-like starting point."""
    h = np.zeros(N)
    left = max(0, center - width // 2)
    right = min(N, center + width // 2 + 1)
    h[left:right] = height
    # Add some smooth decay
    for i in range(N):
        dist = abs(i - center)
        if dist > width // 2:
            h[i] = height * np.exp(-0.1 * (dist - width // 2))
    return h

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    rng = np.random.RandomState(42)
    
    # Phase 1: Differential Evolution on normalized shape
    def objective(x):
        return _neg_value_normalized(x)
    
    try:
        res = opt.differential_evolution(
            objective, 
            bounds=[(0, 1)] * N, 
            seed=42, 
            maxiter=80, 
            popsize=15,
            tol=1e-6,
            init='latinhypercube'
        )
        track(res.x)
    except Exception:
        pass

    # Phase 2: Try diverse starting points including plateau structures
    if time.time() < deadline:
        starts = [best_h.copy()]
        # Add plateau-like structures
        for center in [10, 25, 40]:
            for width in [5, 10, 15]:
                starts.append(_plateau_start(center, width))
        # Add random perturbations
        for _ in range(10):
            pert = best_h + rng.normal(0, 0.1, N)
            pert = np.maximum(0, pert)
            if np.max(pert) > 0:
                pert = pert / np.max(pert)
            starts.append(pert)
        
        for start in starts:
            if time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    _neg_value_normalized,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline:
        try:
            res = opt.minimize(
                _neg_value_normalized,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            track(res.x)
        except Exception:
            pass

    # Normalize to unit max for safety (value is scale-invariant)
    if np.max(best_h) > 0:
        best_h = best_h / np.max(best_h)
    
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
