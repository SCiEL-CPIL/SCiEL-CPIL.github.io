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
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Strategy 1: Nelder-Mead from Keich point with tight tolerances
    np.random.seed(42)
    
    result = minimize(
        _objective_fast, x0, method='Nelder-Mead',
        options={'maxiter': 15000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
    )
    if result.fun < best_area:
        best_area = result.fun
        best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 2: Multiple Nelder-Mead restarts with varied perturbations
    for restart in range(12):
        if time.time() > deadline - 40:
            break
        # Vary perturbation scale
        scale = 0.003 * (1 + restart * 0.3)
        x_start = best_x + np.random.normal(0, scale, size=n)
        result = minimize(
            _objective_fast, x_start, method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 3: Gradient descent refinement
    if time.time() < deadline - 80:
        x_gd = _gradient_descent(best_x, n=n, n_heights=3000, lr=0.005, n_steps=1500, momentum=0.9)
        # Follow up with Nelder-Mead from GD result
        result = minimize(
            _objective_fast, x_gd, method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Strategy 4: Try different initial constructions
    # Random uniform initialization
    if time.time() < deadline - 60:
        for trial in range(3):
            if time.time() > deadline - 40:
                break
            x_rand = np.random.uniform(-0.5, 0.5, size=n)
            result = minimize(
                _objective_fast, x_rand, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x.astype(np.float64)
    
    # Final refinement with Nelder-Mead from best point
    if time.time() < deadline - 20:
        result = minimize(
            _objective_fast, best_x, method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
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
