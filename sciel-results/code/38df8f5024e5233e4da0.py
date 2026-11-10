# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_area(x):
    """Compute the area of the union of 64 triangles given base positions x."""
    n = 64
    a = (np.arange(n) + 1) / n
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / 64.0
    
    L_sorted = np.sort(L, axis=1)
    d = np.diff(L_sorted, axis=1)
    d_clipped = np.minimum(d, w[:, None])
    union_lengths = w + d_clipped.sum(axis=1)
    area = union_lengths.mean()
    
    return area


def _compute_area_grad(x):
    """Compute gradient of area with respect to x using central differences."""
    n = 64
    eps = 1e-8
    grad = np.zeros(n)
    base_area = _compute_area(x)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_compute_area(x_plus) - _compute_area(x_minus)) / (2 * eps)
    
    return grad


def _keich_positions(n=64):
    """Generate the Keich construction for n=64."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses multiple optimization strategies to find a good solution.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    n = 64
    x0 = _keich_positions(n)
    
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Strategy 1: L-BFGS-B with numerical gradient
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        try:
            result = minimize(
                _compute_area,
                x0,
                method='L-BFGS-B',
                jac=_compute_area_grad,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxfun': 5000,
                    'maxls': 50
                }
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            result2 = minimize(
                _compute_area,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result2.fun < best_area:
                best_area = result2.fun
                best_x = result2.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Try differential evolution with bounded search
    elapsed = time.time() - start_time
    if elapsed < deadline - 180:
        try:
            # Use a reasonable bounding box around the Keich construction
            bounds = [(-2.0, 2.0) for _ in range(n)]
            
            def _obj(x):
                if time.time() - start_time > deadline - 30:
                    return 1e10
                return _compute_area(x)
            
            result3 = differential_evolution(
                _obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True
            )
            if result3.fun < best_area:
                best_area = result3.fun
                best_x = result3.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Final L-BFGS-B polish from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result4 = minimize(
                _compute_area,
                best_x,
                method='L-BFGS-B',
                jac=_compute_area_grad,
                options={
                    'maxiter': 200,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxfun': 3000,
                    'maxls': 50
                }
            )
            if result4.fun < best_area:
                best_area = result4.fun
                best_x = result4.x.copy()
        except Exception:
            pass
    
    # Ensure result is valid
    if not np.all(np.isfinite(best_x)):
        return x0.copy()
    
    return best_x


# EVOLVE-BLOCK-END
