"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(centers):
    """Compute the packing radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5):
        return -1.0
    if np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
        return -1.0
    
    # Min distance to boundary
    min_boundary = np.min(np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])))
    
    # Min pairwise distance / 2
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    min_pair /= 2.0
    
    return min(min_boundary, min_pair)


def _feasibility_violation(centers, r):
    """
    Return a scalar measure of how much the configuration violates the
    constraints for radius r. Lower is better; 0 means feasible.
    We use a sum of squared violations for smoothness.
    """
    centers = np.asarray(centers).reshape(-1, 2)
    n = centers.shape[0]
    
    # Boundary violations: each center must satisfy |x| <= 0.5 - r, |y| <= 0.5 - r
    # Violation per coordinate: max(0, |coord| - (0.5 - r))
    bound_x = np.maximum(0.0, np.abs(centers[:, 0]) - (0.5 - r))
    bound_y = np.maximum(0.0, np.abs(centers[:, 1]) - (0.5 - r))
    boundary_viol = np.sum(bound_x**2) + np.sum(bound_y**2)
    
    # Pairwise distance violations: for each pair, need dist >= 2r
    # Violation per pair: max(0, 2r - dist)^2
    pair_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            v = max(0.0, 2.0 * r - d)
            pair_viol += v * v
    
    return boundary_viol + pair_viol


def _optimize_for_radius(r, x0, time_budget):
    """
    Try to find a configuration with radius at least r starting from x0.
    Returns (centers, success, violation).
    """
    x0 = np.asarray(x0).reshape(-1)
    n = N
    
    def objective(x):
        return _feasibility_violation(x.reshape(-1, 2), r)
    
    def grad(x):
        # Numerical gradient would be too slow; use L-BFGS with finite diff
        # but we'll let scipy handle it
        return None
    
    best_x = x0.copy()
    best_viol = objective(best_x)
    
    if best_viol < 1e-10:
        return best_x.reshape(-1, 2), True, best_viol
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    start_time = time.time()
    max_iter = 2000
    
    try:
        result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
        )
        if result.fun < best_viol:
            best_x = result.x
            best_viol = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return best_x.reshape(-1, 2), best_viol < 1e-10, best_viol
    
    # If we still have violation, try a few more restarts with small perturbations
    remaining = time_budget - elapsed
    if remaining > 2.0 and best_viol > 1e-8:
        for attempt in range(3):
            if time.time() - start_time > time_budget - 1.0:
                break
            x_try = best_x + np.random.normal(0, 0.01, size=best_x.shape)
            # Clip to bounds
            x_try = np.clip(x_try, -0.5, 0.5)
            try:
                result = minimize(
                    objective,
                    x_try,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                )
                if result.fun < best_viol:
                    best_x = result.x
                    best_viol = result.fun
            except Exception:
                pass
    
    return best_x.reshape(-1, 2), best_viol < 1e-10, best_viol


def _grid_initial(m=None):
    """Create a grid initial configuration."""
    if m is None:
        m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_initial():
    """Create a hexagonal initial configuration."""
    # Approximate hexagonal packing
    points = []
    # Try a hexagonal lattice
    spacing = 0.15
    rows = 8
    cols = 8
    for i in range(rows):
        for j in range(cols):
            x = -0.5 + 0.08 + j * spacing + (i % 2) * spacing * 0.5
            y = -0.5 + 0.08 + i * spacing * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # Pad with grid points
        grid = _grid_initial()
        for p in grid:
            if len(points) >= N:
                break
            if not any(np.allclose(p, q) for q in points):
                points = np.vstack([points, p])
    return points[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    x0_grid = _grid_initial().reshape(-1)
    x0_hex = _hex_initial().reshape(-1)
    
    # Binary search on radius
    # Lower bound: very small, Upper bound: theoretical max ~0.5/sqrt(2) but realistically ~0.075
    r_low = 0.0
    r_high = 0.075
    
    # First, check what radius the grid gives
    grid_score = _score(_grid_initial())
    r_low = max(r_low, grid_score)
    
    best_centers = _grid_initial()
    best_radius = grid_score
    
    # Binary search iterations
    for it in range(20):
        if time.time() > deadline - 30.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find feasible configuration for r_mid
        # Use grid as initial guess
        centers, success, viol = _optimize_for_radius(r_mid, x0_grid, time_budget=min(20.0, deadline - time.time() - 10.0))
        
        if success:
            r_low = r_mid
            best_centers = centers
            best_radius = r_mid
        else:
            r_high = r_mid
            # Even if not fully feasible, keep the best found so far
            if _score(centers) > best_radius:
                best_radius = _score(centers)
                best_centers = centers
        
        # Also try hex initial if we have time
        if time.time() < deadline - 40.0 and it >= 5:
            centers2, success2, viol2 = _optimize_for_radius(r_mid, x0_hex, time_budget=min(15.0, deadline - time.time() - 10.0))
            if success2:
                r_low = r_mid
                best_centers = centers2
                best_radius = r_mid
            else:
                if _score(centers2) > best_radius:
                    best_radius = _score(centers2)
                    best_centers = centers2
    
    # Final polish: try to improve the best configuration found
    if time.time() < deadline - 15.0:
        r_target = best_radius + 0.0001  # Try to push slightly higher
        centers, success, viol = _optimize_for_radius(r_target, best_centers.reshape(-1), time_budget=min(15.0, deadline - time.time() - 5.0))
        if _score(centers) > best_radius:
            best_centers = centers
            best_radius = _score(centers)
    
    # Ensure valid output
    best_centers = np.asarray(best_centers)
    if best_centers.shape != (N, 2):
        best_centers = _grid_initial()
    # Clip to square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
