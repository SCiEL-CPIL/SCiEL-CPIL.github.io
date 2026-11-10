# EVOLVE-BLOCK-START
import numpy as np
import time

def _entropy_bits(p):
    """Shannon entropy in bits, ignoring zero entries."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _project_entropy(P, r):
    """Entropy of distribution of x + r*y where P[y,x] is joint."""
    n = P.shape[0]
    # x in 0..n-1, y in 0..n-1
    # value = x + r*y
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    vals = xs[None, :] + r * ys[:, None]  # shape (n, n): vals[y, x]
    # pool by value
    # Use rounding for integer slopes
    if abs(r - round(r)) < 1e-12:
        ri = int(round(r))
        vals_int = (xs[None, :] + ri * ys[:, None]).astype(np.int64)
        unique_vals, inverse = np.unique(vals_int, return_inverse=True)
        m = len(unique_vals)
        dist = np.zeros(m, dtype=np.float64)
        np.add.at(dist, inverse.ravel(), P.ravel())
    else:
        vals_f = vals.ravel()
        unique_vals, inverse = np.unique(np.round(vals_f, 10), return_inverse=True)
        m = len(unique_vals)
        dist = np.zeros(m, dtype=np.float64)
        np.add.at(dist, inverse, P.ravel())
    return _entropy_bits(dist)

def _objective(P):
    """Return D/M, or -inf if invalid."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        return -np.inf
    Pn = P / s
    D = _project_entropy(Pn, -1.0)
    H0 = _project_entropy(Pn, 0.0)
    H1 = _project_entropy(Pn, 1.0)
    H2 = _project_entropy(Pn, 2.0)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return -np.inf
    return D / M

def _normalize(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / s

def _projected_gradient_step(P, grad, lr):
    """One projected gradient ascent step on the simplex (non-negative, sum=1)."""
    P_new = P + lr * grad
    # Project onto simplex: non-negative, sum=1
    # Simple approach: clip negatives, renormalize
    P_new = np.clip(P_new, 0, None)
    s = P_new.sum()
    if s <= 1e-9:
        return np.ones_like(P) / P.size
    return P_new / s

def _numerical_gradient(P, f, eps=1e-5):
    """Numerical gradient of f(P) w.r.t. P, treating P as free (then we project)."""
    n = P.size
    grad = np.zeros(n, dtype=np.float64)
    P_flat = P.ravel().copy()
    for i in range(n):
        P_plus = P_flat.copy()
        P_plus[i] += eps
        # Renormalize to keep on simplex for fair comparison? 
        # Actually for gradient of the raw objective before projection, 
        # we should compute directional derivative. But since we project after,
        # let's compute gradient of f(P/s) w.r.t. P (unnormalized).
        # Simpler: use central difference on the normalized version.
        P_plus_n = P_plus / P_plus.sum()
        P_minus = P_flat.copy()
        P_minus[i] = max(0, P_flat[i] - eps)
        P_minus_n = P_minus / P_minus.sum() if P_minus.sum() > 0 else P_flat / P_flat.sum()
        fp = f(P_plus_n)
        fm = f(P_minus_n)
        if np.isfinite(fp) and np.isfinite(fm):
            grad[i] = (fp - fm) / (2 * eps)
        else:
            grad[i] = 0.0
    return grad.reshape(P.shape)

def _optimize_from_start(P0, max_iter=300, lr=0.05, deadline=None):
    """Optimize from P0 using projected gradient ascent."""
    P = P0.copy()
    best_P = P.copy()
    best_val = _objective(P)
    if not np.isfinite(best_val):
        best_val = -np.inf
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        val = _objective(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()
        
        # Compute gradient
        grad = _numerical_gradient(P, _objective, eps=1e-4)
        # Projected gradient ascent
        P_new = _projected_gradient_step(P, grad, lr)
        
        # Check improvement
        val_new = _objective(P_new)
        if np.isfinite(val_new) and val_new > val:
            P = P_new
            # Reduce learning rate slowly
            lr *= 0.995
        else:
            # Try smaller step
            P_new2 = _projected_gradient_step(P, grad, lr * 0.1)
            val_new2 = _objective(P_new2)
            if np.isfinite(val_new2) and val_new2 > val:
                P = P_new2
                lr *= 0.99
            else:
                lr *= 0.8
                if lr < 1e-6:
                    break
    return best_P, best_val

def _random_start(seed):
    rng = np.random.RandomState(seed)
    P = rng.dirichlet(np.ones(100))
    return P.reshape(10, 10)

def _structured_start():
    """Create a structured starting point: emphasize diagonals where x-y is uniform."""
    n = 10
    P = np.zeros((n, n), dtype=np.float64)
    # Put mass on all cells but weight diagonals (x-y = const) more
    # Actually, let's start with uniform and add some structure
    # Put extra mass where x + y is concentrated (to reduce H1)
    # and where x is concentrated (to reduce H0)
    # and where x + 2y is concentrated (to reduce H2)
    
    # Strategy: put mass on a "staircase" pattern
    # For each y, put mass on x = c (concentrate x)
    # But that makes x-y = c - y, which is uniform if y is uniform
    # Let's try: P[y, x] = 1/100 for all, then perturb
    
    # Better: start with a rank-1 structure P = u v^T
    # u[y] = 1/10 for all y (uniform in y)
    # v[x] = 1/10 for all x (uniform in x)
    # This gives uniform, which is the baseline.
    
    # Let's try concentrating x: v[x] peaked at x=5
    # Then x-y = 5-y, which is still uniform over y
    # x+y = 5+y, concentrated
    # x+2y = 5+2y, less concentrated
    u = np.ones(10) / 10
    v = np.zeros(10)
    v[5] = 1.0  # all mass at x=5
    P = np.outer(u, v)
    return P

def _diagonal_start():
    """Start with mass on the main diagonal x=y, so x-y=0 (concentrated D, bad)."""
    # Instead, spread over all diagonals equally
    n = 10
    P = np.zeros((n, n), dtype=np.float64)
    # For each diagonal d = x-y in -9..9, put equal mass
    # Number of cells on diagonal d:
    # d=0: 10 cells, d=1: 9, ..., d=9: 1, d=-1: 9, ..., d=-9: 1
    for d in range(-9, 10):
        count = 10 - abs(d)
        if count > 0:
            # Cells: (y, x) where x - y = d, so x = y + d
            # y ranges from max(0, -d) to min(9, 9-d)
            y_start = max(0, -d)
            y_end = min(9, 9 - d)
            for y in range(y_start, y_end + 1):
                x = y + d
                P[y, x] = 1.0 / (10 * 10)  # uniform
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit, leave margin
    safety_deadline = start_time + 590.0
    
    n = 10
    # Baseline: uniform
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_val = _objective(P_uniform)
    
    # Try several starting points
    starts = [
        ("uniform", P_uniform),
        ("structured_x5", _structured_start()),
        ("random1", _random_start(42)),
        ("random2", _random_start(123)),
        ("random3", _random_start(999)),
    ]
    
    for name, P0 in starts:
        if time.time() > safety_deadline:
            break
        # Optimize from this start
        P_opt, val_opt = _optimize_from_start(
            P0, max_iter=200, lr=0.03, deadline=safety_deadline
        )
        if np.isfinite(val_opt) and val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
    
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = P_uniform
    else:
        best_P = best_P / s
    
    return best_P

# EVOLVE-BLOCK-END
