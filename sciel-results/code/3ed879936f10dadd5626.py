# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _objective_fast(x, n=64, n_heights=6000):
    """Fast objective for optimization using fewer heights."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _objective_fast_nocache(x, n=64, n_heights=6000):
    """Fast objective that recomputes a, y, w each call (for minimize)."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _coord_descent(x, n=64, n_heights=3000, deadline=None, max_rounds=500):
    """Coordinate descent with golden-section line search per coordinate.
    Much faster per iteration than full gradient methods, enabling many more steps."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    
    def obj(x):
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
        return np.mean(union_lens)
    
    x = x.copy()
    current = obj(x)
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Golden-section search on x[i] in a window around current value
            # First determine search range by expanding
            lo = x[i] - 0.05
            hi = x[i] + 0.05
            # Quick check: evaluate at lo and hi to bracket
            x_test = x.copy()
            x_test[i] = lo
            f_lo = obj(x_test)
            x_test[i] = hi
            f_hi = obj(x_test)
            
            # If both are worse, shrink window
            if f_lo > current and f_hi > current:
                lo = x[i] - 0.005
                hi = x[i] + 0.005
                x_test[i] = lo
                f_lo = obj(x_test)
                x_test[i] = hi
                f_hi = obj(x_test)
                if f_lo > current and f_hi > current:
                    continue
            
            # Golden section search
            for _ in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                c = hi - golden * (hi - lo)
                d = lo + golden * (hi - lo)
                x_test = x.copy()
                x_test[i] = c
                f_c = obj(x_test)
                x_test[i] = d
                f_d = obj(x_test)
                if f_c < f_d:
                    hi = d
                else:
                    lo = c
            
            x_new_i = (lo + hi) / 2.0
            x_test = x.copy()
            x_test[i] = x_new_i
            f_new = obj(x_test)
            if f_new < current - 1e-15:
                x[i] = x_new_i
                current = f_new
                improved = True
        if not improved:
            break
    return x, current

def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0

def _gradient_descent(x, n=64, n_heights=4000, lr=0.01, n_steps=2000, momentum=0.9):
    """Perform gradient descent with numerical gradient and momentum."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    
    def obj(x):
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
        return np.mean(union_lens)
    
    def grad(x):
        g = np.zeros(n)
        h = 1e-7
        f0 = obj(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += h
            g[i] = (obj(xp) - f0) / h
        return g
    
    v = np.zeros(n)
    for step in range(n_steps):
        g = grad(x)
        v = momentum * v - lr * g
        x = x + v
    return x

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 610  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    np.random.seed(42)
    
    # Strategy 1: Coordinate descent from Keich point (fast, many iterations)
    x_cd, area_cd = _coord_descent(best_x, n=n, n_heights=3000, deadline=deadline - 30, max_rounds=200)
    if area_cd < best_area:
        best_area = area_cd
        best_x = x_cd.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 2: Multiple coordinate descent restarts with varied perturbations
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        scale = 0.005 * (1 + restart * 0.2)
        x_start = best_x + np.random.normal(0, scale, size=n)
        x_cd, area_cd = _coord_descent(x_start, n=n, n_heights=3000, deadline=deadline - 30, max_rounds=100)
        if area_cd < best_area:
            best_area = area_cd
            best_x = x_cd.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 3: Nelder-Mead polish from best coordinate descent result
    if time.time() < deadline - 40:
        result = minimize(
            _objective_fast_nocache, best_x, method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 4: Another round of coordinate descent from NM result
    if time.time() < deadline - 20:
        x_cd, area_cd = _coord_descent(best_x, n=n, n_heights=3000, deadline=deadline - 10, max_rounds=100)
        if area_cd < best_area:
            best_area = area_cd
            best_x = x_cd.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 5: Random restarts with coordinate descent
    if time.time() < deadline - 20:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            x_rand = np.random.uniform(-0.5, 0.5, size=n)
            x_cd, area_cd = _coord_descent(x_rand, n=n, n_heights=3000, deadline=deadline - 10, max_rounds=80)
            if area_cd < best_area:
                best_area = area_cd
                best_x = x_cd.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Final NM polish
    if time.time() < deadline - 5:
        result = minimize(
            _objective_fast_nocache, best_x, method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Verify with full precision
    final_area = _compute_area(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        # Fallback to Keich
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
