# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Constants
N = 64
M = 20000  # full quadrature points
A = (np.arange(N) + 1) / N  # a_i
W0 = 1.0 / N  # base width


def compute_area(x, M_eval=None):
    """Compute the area of the union of triangles for given base positions x.
    
    Uses the quadrature formula: at each height y_m, the cross-section of triangle i
    is [L_i, L_i + w] where L_i = x_i + a_i * y and w = (1-y)/N.
    The union length is w + sum over consecutive pairs of min(L_{j+1}-L_j, w).
    """
    if M_eval is None:
        M_eval = M
    y = (np.arange(M_eval) + 0.5) / M_eval  # shape (M_eval,)
    w = (1.0 - y) / N  # shape (M_eval,)
    
    # L shape: (M_eval, N)
    # L[i, j] = x[j] + a_j * y[i]
    L = x[None, :] + y[:, None] * A[None, :]
    
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)  # (M_eval, N)
    
    # Differences between consecutive sorted L values
    # For union length: w + sum_{j=0}^{N-2} min(L_sorted[j+1] - L_sorted[j], w)
    diffs = np.diff(L_sorted, axis=1)  # (M_eval, N-1)
    
    # min(diffs, w[:, None])
    mins = np.minimum(diffs, w[:, None])
    
    # Union length at each height
    union_lengths = w + mins.sum(axis=1)  # (M_eval,)
    
    area = np.mean(union_lengths)
    return area


def compute_area_fast(x, M_eval=4000):
    """Faster area computation with reduced resolution for optimization."""
    return compute_area(x, M_eval)


def compute_area_grad(x, M_eval=4000):
    """Compute area and gradient via finite differences or analytic approach.
    
    For speed, we use central finite differences on the reduced-resolution area.
    """
    area = compute_area_fast(x, M_eval)
    
    # Gradient via central differences (step size tuned for reduced resolution)
    eps = 1e-6
    grad = np.zeros_like(x)
    for i in range(N):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (compute_area_fast(x_plus, M_eval) - compute_area_fast(x_minus, M_eval)) / (2 * eps)
    
    return area, grad


def soft_min_area(x, M_eval=4000, beta=50.0):
    """Soft surrogate for the area using log-sum-exp for the min operations.
    
    Instead of exact min(L_{j+1}-L_j, w), we use a smooth approximation.
    This makes the objective differentiable and amenable to gradient methods.
    
    Actually, let's use a different approach: the union length can be written as
    max over subsets, but that's complex. Instead, let's use the fact that the
    union of intervals [L_i, L_i+w] has length = w + sum min(diffs, w) after sorting.
    
    For a smooth version, we can replace min(a, b) with -log(exp(-beta*a) + exp(-beta*b))/beta
    for large beta, which approximates min(a,b).
    
    But the sorting is the main issue. Let's try a different smooth surrogate:
    The union length of intervals [L_i, L_i+w] equals the integral over the line of
    the indicator that the point is covered by at least one interval.
    
    Alternatively, we can use the inclusion-exclusion or the fact that for sorted
    intervals, union = sum of lengths - sum of overlaps + ...
    
    Let me try a simpler approach: just use the exact area but with gradient
    computed via the sorted structure.
    """
    # For now, let's just return the exact area and compute gradient separately
    return compute_area_fast(x, M_eval)


def keich_construction():
    """Classical Keich construction for n=64 (power of 2)."""
    n = N
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        # Get binary representation of i, padded to k bits
        bin_str = bin(i)[2:]
        eps = np.array([int(c) for c in bin_str], dtype=float)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        # x_i = sum_{j=1}^{k} ((1-j)/k) * eps_j * 2^{-j}
        pos = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        positions.append(pos)
    return np.array(positions, dtype=np.float64)


def random_init(scale=0.05):
    """Generate a random initial point near the Keich construction."""
    base = keich_construction()
    return base + np.random.randn(N) * scale


def structured_perturbation(x, rng):
    """Apply a structured perturbation: subset shift or pairwise swap."""
    x_new = x.copy()
    op = rng.integers(0, 3)
    
    if op == 0:
        # Subset shift: shift a random subset of positions
        mask = rng.random(N) < 0.3
        shift = rng.normal(0, 0.01, N)
        x_new[mask] += shift[mask]
    elif op == 1:
        # Pairwise swap-like: swap two positions with some noise
        i, j = rng.choice(N, size=2, replace=False)
        x_new[i], x_new[j] = x_new[j], x_new[i]
        x_new[i] += rng.normal(0, 0.005)
        x_new[j] += rng.normal(0, 0.005)
    else:
        # Local Gaussian perturbation
        x_new += rng.normal(0, 0.005, N)
    
    return x_new


def polish(x, M_eval=4000, max_iter=200, time_limit=None, start_time=None):
    """Polish a point using L-BFGS-B with reduced-resolution gradient."""
    if start_time is None:
        start_time = time.time()
    
    def objective_and_grad(x):
        if time_limit is not None and time.time() - start_time > time_limit:
            raise TimeoutError
        area = compute_area_fast(x, M_eval)
        # Compute gradient via finite differences
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(N):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (compute_area_fast(x_plus, M_eval) - compute_area_fast(x_minus, M_eval)) / (2 * eps)
        return area, grad
    
    try:
        result = minimize(
            objective_and_grad,
            x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < compute_area_fast(x, M_eval):
            return result.x
        else:
            return x
    except (TimeoutError, Exception):
        return x


def basin_hopping(x, rng, max_hops=50, time_limit=None, start_time=None):
    """Basin hopping with structured perturbations and L-BFGS polish."""
    if start_time is None:
        start_time = time.time()
    
    best_x = x.copy()
    best_area = compute_area_fast(x, 2000)
    
    for hop in range(max_hops):
        if time_limit is not None and time.time() - start_time > time_limit - 30:
            break
        
        # Generate perturbed point
        x_pert = structured_perturbation(best_x, rng)
        
        # Polish
        x_polished = polish(x_pert, M_eval=4000, max_iter=100, time_limit=time_limit, start_time=start_time)
        
        # Evaluate
        area = compute_area_fast(x_polished, 2000)
        
        if area < best_area:
            best_area = area
            best_x = x_polished.copy()
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses basin hopping with structured perturbations and L-BFGS polish,
    starting from multiple initial points including the Keich construction.
    """
    start_time = time.time()
    total_time_limit = 600.0
    safety_margin = 30.0  # seconds
    
    rng = np.random.default_rng(42)
    
    # Start with Keich construction
    best_x = keich_construction()
    best_area = compute_area(best_x, 2000)
    
    # Try multiple initial points
    n_inits = 12
    for init_idx in range(n_inits):
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - safety_margin:
            break
        
        if init_idx == 0:
            x_init = best_x.copy()
        else:
            x_init = random_init(scale=0.05 + 0.01 * init_idx)
        
        # Basin hopping from this initial point
        x_result, area_result = basin_hopping(
            x_init, rng,
            max_hops=30,
            time_limit=total_time_limit - safety_margin,
            start_time=start_time
        )
        
        if area_result < best_area:
            best_area = area_result
            best_x = x_result.copy()
    
    # Final high-precision polish on the best solution
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - safety_margin:
        # Use full resolution for final polish
        def final_objective(x):
            return compute_area(x, M)
        
        def final_objective_grad(x):
            area = compute_area(x, M)
            eps = 1e-8
            grad = np.zeros_like(x)
            for i in range(N):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (compute_area(x_plus, M) - compute_area(x_minus, M)) / (2 * eps)
            return area, grad
        
        try:
            result = minimize(
                final_objective_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.fun < best_area:
                best_x = result.x
                best_area = result.fun
        except Exception:
            pass
    
    # Ensure finite values
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
