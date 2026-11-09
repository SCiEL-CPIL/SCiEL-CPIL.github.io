"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _initial_hex_lattice(n):
    """Generate a hexagonal lattice of points inside the unit disk, return first n."""
    # We want to pack n points. Use a hexagonal grid with spacing s.
    # Area of hex cell ~ s^2 * sqrt(3)/2. Disk area = pi.
    # n * s^2 * sqrt(3)/2 <= pi  =>  s <= sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.95
    
    # Generate grid points
    # For hex packing, rows are offset
    # Let's generate a rectangular grid of hex points and clip to disk
    max_r = 1.0
    # Number of points along diameter ~ 2/s
    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            r = np.hypot(x, y)
            if r <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: spiral
        theta = np.linspace(0, 2 * np.pi * 10, n)
        r = np.linspace(0, 1, n)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        # Sort by radius to get a nice set, take first n
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')
        points = points[idx[:n]]
    return points


def _compute_score(points):
    """Compute the packing score: min pair distance / 2, min distance to boundary."""
    n = len(points)
    # Pairwise distances
    # Use broadcasting for small n, but for 300 it's 90k pairs, fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair / 2.0, min_boundary)


def _objective(x, n):
    """Objective function for optimization: negative score (to maximize)."""
    points = x.reshape(n, 2)
    # Check validity: if any point is outside, penalize heavily
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii > 1.0):
        return 1e6 + np.sum(radii[radii > 1.0] - 1.0)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    min_boundary = np.min(1.0 - radii)
    
    score = min(min_pair / 2.0, min_boundary)
    return -score


def _objective_gradient(x, n):
    """Gradient of the objective (negative score)."""
    points = x.reshape(n, 2)
    radii = np.linalg.norm(points, axis=1)
    
    # If outside, gradient is just pointing inward
    if np.any(radii > 1.0):
        grad = np.zeros_like(points)
        mask = radii > 1.0
        grad[mask] = points[mask] / radii[mask, np.newaxis]
        return grad.flatten()
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    min_pair = np.min(dist)
    min_boundary = np.min(1.0 - radii)
    
    score = min(min_pair / 2.0, min_boundary)
    
    # Gradient of negative score
    # d(-score)/dx = -d(score)/dx
    # score = min(min_pair/2, min_boundary)
    # This is non-smooth, but we can approximate
    
    grad = np.zeros_like(points)
    
    if min_pair / 2.0 <= min_boundary:
        # Gradient comes from min_pair
        # min_pair = min_{i<j} |p_i - p_j|
        # Find the pair(s) achieving min
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        if i < j:
            d_ij = points[i] - points[j]
            norm_ij = np.linalg.norm(d_ij)
            if norm_ij > 1e-12:
                grad[i] += 0.5 * d_ij / norm_ij
                grad[j] -= 0.5 * d_ij / norm_ij
        else:
            d_ij = points[j] - points[i]
            norm_ij = np.linalg.norm(d_ij)
            if norm_ij > 1e-12:
                grad[j] += 0.5 * d_ij / norm_ij
                grad[i] -= 0.5 * d_ij / norm_ij
    else:
        # Gradient comes from min_boundary
        # min_boundary = min_i (1 - |p_i|)
        min_idx = np.argmin(radii)
        i = min_idx
        if radii[i] > 1e-12:
            grad[i] += points[i] / radii[i]
    
    return (-grad).flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 15s safety margin
    
    # Step 1: Generate initial hexagonal lattice
    points = _initial_hex_lattice(N)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Step 2: Use L-BFGS-B to optimize
    x0 = points.flatten()
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(N,),
            method='L-BFGS-B',
            jac=_objective_gradient,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < -best_score:
            new_points = result.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
    except Exception:
        pass
    
    # Step 3: If time remains, try a few random restarts with local optimization
    while time.time() < deadline:
        # Random perturbation of best points
        noise_scale = 0.01
        x0 = best_points.flatten() + np.random.normal(0, noise_scale, size=N * 2)
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(N,),
                method='L-BFGS-B',
                jac=_objective_gradient,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    # Final validation: ensure all points are inside and non-overlapping
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0):
        # Clip points that are slightly outside
        mask = radii > 1.0
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
