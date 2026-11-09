# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

def _compute_area_fast(x, n=64, n_heights=10000):
    """Compute area with fewer heights for optimization speed."""
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

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 610  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Define objective for optimization (using fewer heights for speed)
    def objective(x):
        a = (np.arange(n) + 1) / n
        y_vals = (np.arange(8000) + 0.5) / 8000
        w_vals = (1.0 - y_vals) / n
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
        return np.mean(union_lens)
    
    # Strategy 1: Nelder-Mead from Keich point
    np.random.seed(42)
    
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    if result.fun < best_area:
        best_area = result.fun
        best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 2: Multiple Nelder-Mead restarts with perturbations
    for restart in range(8):
        if time.time() > deadline - 30:
            break
        # Vary perturbation scale
        scale = 0.005 * (1 + restart * 0.5)
        x_start = best_x + np.random.normal(0, scale, size=n)
        result = minimize(
            objective, x_start, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 3: Try L-BFGS-B with gradient approximation
    def objective_grad(x):
        a = (np.arange(n) + 1) / n
        y_vals = (np.arange(5000) + 0.5) / 5000
        w_vals = (1.0 - y_vals) / n
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
        return np.mean(union_lens)
    
    def numerical_grad(x):
        grad = np.zeros(n)
        step = 1e-6
        f0 = objective_grad(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += step
            f_plus = objective_grad(x_plus)
            grad[i] = (f_plus - f0) / step
        return grad
    
    if time.time() < deadline - 60:
        try:
            result = minimize(
                objective_grad, best_x, method='L-BFGS-B',
                jac=numerical_grad,
                options={'maxiter': 300, 'gtol': 1e-8}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 4: Differential evolution for global search (limited)
    if time.time() < deadline - 120:
        try:
            bounds = [(-2, 2)] * n
            de_result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.7,
                polish=False
            )
            if de_result.fun < best_area:
                best_area = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_x
    
    # Final refinement with Nelder-Mead from best point
    if time.time() < deadline - 20:
        result = minimize(
            objective, best_x, method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
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
