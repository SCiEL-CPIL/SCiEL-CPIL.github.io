"""Rewritten program for packing 300 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300
DEADLINE = 110.0  # seconds, safety margin before 150s limit


def _hex_lattice_init(n):
    """Generate a hexagonal lattice initialization for n points in unit disk."""
    # Estimate spacing for n points in unit disk
    # Area of unit disk = pi, area per point ~ pi/n, spacing ~ sqrt(pi/n)
    spacing = np.sqrt(np.pi / n) * 0.95
    
    # Generate hexagonal lattice points
    rows = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    
    if len(points) < n:
        # Fallback: spiral initialization
        theta = np.linspace(0, 2 * np.pi * np.sqrt(n), n)
        r = np.sqrt(np.linspace(0, 1, n))
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        # Select n points closest to center
        idx = np.argsort(radius, kind='stable')[:n]
        points = points[idx]
    
    return points


def _compute_score(points):
    """Compute the packing score: min of half-min-distance and min-margin-to-boundary."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    # Compute distances to boundary
    r = np.sqrt(np.sum(points ** 2, axis=1))
    min_margin = np.min(1.0 - r)
    
    return min(min_dist / 2.0, min_margin)


def _soft_min_objective(points, beta=100.0):
    """
    Soft-min objective: -softmin of (pairwise distances/2, margins to boundary).
    We want to maximize the minimum, so we minimize the negative soft-min.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    
    # Margins to boundary
    r = np.sqrt(np.sum(points ** 2, axis=1))
    margin_vals = 1.0 - r
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals.ravel(), margin_vals])
    
    # Soft-min: -logsumexp(-beta * vals) / beta
    # For numerical stability
    max_val = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - max_val))
    soft_min = max_val - np.log(np.sum(exp_terms)) / beta
    
    return -soft_min


def _soft_min_gradient(points, beta=100.0):
    """Gradient of the soft-min objective."""
    n = len(points)
    if n < 2:
        return np.zeros_like(points)
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    
    # Margins to boundary
    r = np.sqrt(np.sum(points ** 2, axis=1))
    margin_vals = 1.0 - r
    
    all_vals = np.concatenate([pair_vals.ravel(), margin_vals])
    
    max_val = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - max_val))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of -soft_min w.r.t. points
    grad = np.zeros_like(points)
    
    # Pairwise part: for each pair (i,j), weight w_ij * d/dp_i (dist_ij/2)
    # d/dp_i (dist_ij) = (p_i - p_j) / dist_ij
    pair_weights = weights[:n * (n - 1)]
    pair_weights = pair_weights.reshape(n, n - 1)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j] if j < n else pair_weights[i, j - 1]
            # Actually, let's restructure this properly
            pass
    
    # Simpler: compute gradient numerically or use a different approach
    # For efficiency, let's use a simpler gradient computation
    
    # Reconstruct: pair_vals has shape (n, n) with inf on diagonal
    # Flatten order: row-major, so pair_vals[i, j] is at index i*n + j
    # But we set diagonal to inf, so those are excluded from soft-min effectively
    
    # Let's redo this more carefully
    pair_weights_full = np.zeros((n, n))
    flat_idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_weights_full[i, j] = weights[flat_idx]
                flat_idx += 1
    
    # Gradient from pairwise terms
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights_full[i, j]
            if dist[i, j] > 1e-10:
                grad[i] += w * (points[i] - points[j]) / dist[i, j]
    
    # Gradient from boundary terms
    margin_weights = weights[flat_idx:]
    for i in range(n):
        w = margin_weights[i]
        if r[i] > 1e-10:
            grad[i] += w * (points[i] / r[i])
    
    return -grad  # Negative because we're minimizing -soft_min


def _refine_with_l_bfgs(points, max_iter=200, time_budget=5.0):
    """Refine points using L-BFGS-B with soft-min objective."""
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return _soft_min_objective(pts, beta=50.0)
    
    def gradient(x):
        pts = x.reshape(-1, 2)
        g = _soft_min_gradient(pts, beta=50.0)
        return g.ravel()
    
    x0 = points.ravel()
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B', jac=gradient,
            options={'maxiter': max_iter, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < objective(x0):
            new_points = result.x.reshape(-1, 2)
            # Verify validity
            if np.all(np.isfinite(new_points)):
                r = np.sqrt(np.sum(new_points ** 2, axis=1))
                if np.all(r < 1.0):
                    return new_points
    except Exception:
        pass
    
    return points


def construct():
    start_time = time.time()
    
    # Initial hexagonal lattice
    points = _hex_lattice_init(N)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Adaptive beta schedule for soft-min optimization
    betas = [10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        def obj(x):
            pts = x.reshape(-1, 2)
            return _soft_min_objective(pts, beta=beta)
        
        def grad(x):
            pts = x.reshape(-1, 2)
            g = _soft_min_gradient(pts, beta=beta)
            return g.ravel()
        
        x0 = best_points.ravel()
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', jac=grad,
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < obj(x0):
                new_points = result.x.reshape(-1, 2)
                if np.all(np.isfinite(new_points)):
                    r = np.sqrt(np.sum(new_points ** 2, axis=1))
                    if np.all(r < 1.0):
                        score = _compute_score(new_points)
                        if score > best_score:
                            best_score = score
                            best_points = new_points.copy()
        except Exception:
            pass
        
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        points = _refine_with_l_bfgs(best_points, max_iter=50, time_budget=5.0)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Ensure validity
    r = np.sqrt(np.sum(best_points ** 2, axis=1))
    if not np.all(r < 1.0) or not np.all(np.isfinite(best_points)):
        # Fallback to initial
        best_points = _hex_lattice_init(N)
    
    return best_points


if __name__ == '__main__':
    result = construct()
    print(f"Score: {_compute_score(result)}")
    print(f"Shape: {result.shape}")
