"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
DEADLINE = 115.0  # safety margin before 120s limit

def _compute_radius(points):
    """Compute the packing radius for given points."""
    if len(points) < 2:
        return 0.0
    # Boundary constraint
    dist_to_origin = np.sqrt(np.sum(points**2, axis=1))
    boundary_term = 1.0 - np.max(dist_to_origin)
    if boundary_term <= 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(points)
    # Get all pairs within a reasonable distance
    pairs = tree.query_pairs(r=1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair_dist = 2.0
    else:
        diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_pair_dist = np.min(dists)
    pair_term = min_pair_dist / 2.0
    return min(boundary_term, pair_term)


def _energy(points, target_r):
    """Energy function to minimize: penalize overlaps and boundary violations."""
    r = target_r
    # Boundary violations
    dist_to_origin = np.sqrt(np.sum(points**2, axis=1))
    boundary_violation = np.maximum(0, dist_to_origin + r - 1.0)
    boundary_penalty = np.sum(boundary_violation**2) * 10.0
    
    # Pairwise overlap violations
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=2*r*1.5, output_type='ndarray')
    if len(pairs) == 0:
        overlap_penalty = 0.0
    else:
        diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        overlaps = np.maximum(0, 2*r - dists)
        overlap_penalty = np.sum(overlaps**2) * 10.0
    
    return boundary_penalty + overlap_penalty


def _grad_energy(points, target_r):
    """Gradient of the energy function."""
    r = target_r
    grad = np.zeros_like(points)
    
    # Boundary gradient
    dist_to_origin = np.sqrt(np.sum(points**2, axis=1))
    mask = dist_to_origin + r > 1.0
    if np.any(mask):
        # grad of (|p| + r - 1)^2 w.r.t. p = 2*(|p|+r-1)*p/|p|
        violations = dist_to_origin[mask] + r - 1.0
        grad[mask] += 20.0 * violations[:, np.newaxis] * points[mask] / dist_to_origin[mask, np.newaxis]
    
    # Pairwise overlap gradient
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=2*r*1.5, output_type='ndarray')
    if len(pairs) > 0:
        diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        overlaps = np.maximum(0, 2*r - dists)
        # grad w.r.t. p_i: 20 * overlap * (p_i - p_j) / dist
        for idx in range(len(pairs)):
            i, j = pairs[idx]
            if overlaps[idx] > 0 and dists[idx] > 1e-10:
                force = 20.0 * overlaps[idx] * diffs[idx] / dists[idx]
                grad[i] += force
                grad[j] -= force
    
    return grad


def _random_initialization():
    """Generate random points inside the unit disk."""
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _hexagonal_initialization():
    """Generate a hexagonal lattice initialization."""
    # Start with a reasonable spacing
    r_est = 0.125
    spacing = 2 * r_est
    points = []
    # Generate hex lattice points
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 <= (1 - r_est)**2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Select N points closest to origin for better packing
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:N]
        return points[idx]
    else:
        return _random_initialization()


def _ring_initialization():
    """Generate points in concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    # Rings
    r_est = 0.125
    ring_radii = [0.5, 1.0, 1.5]
    for ring_idx, rr in enumerate(ring_radii):
        if len(points) >= N:
            break
        # Number of points on this ring
        circumference = 2 * np.pi * rr
        n_points = int(circumference / (2 * r_est))
        n_points = max(6, n_points)
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        for angle in angles:
            if len(points) >= N:
                break
            x = rr * np.cos(angle)
            y = rr * np.sin(angle)
            if x**2 + y**2 <= (1 - r_est)**2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        return points[:N]
    else:
        return _random_initialization()


def _optimize_from_initial(points, target_r, max_time=10.0):
    """Optimize points from initial configuration."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    # Use L-BFGS-B with gradient
    for attempt in range(3):
        if time.time() - start_time > max_time:
            break
        try:
            result = minimize(
                _energy,
                points,
                jac=_grad_energy,
                args=(target_r,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            new_points = result.x
            new_radius = _compute_radius(new_points)
            if new_radius > best_radius:
                best_points = new_points
                best_radius = new_radius
        except Exception:
            pass
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    
    # Best result so far
    best_points = None
    best_radius = 0.0
    
    # Try different initializations
    initializations = [
        _hexagonal_initialization,
        _ring_initialization,
        _random_initialization
    ]
    
    # Target radius to aim for
    target_r = 0.1258
    
    for init_func in initializations:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            points = init_func()
            points, radius = _optimize_from_initial(points, target_r, max_time=30.0)
            if radius > best_radius:
                best_radius = radius
                best_points = points.copy()
        except Exception:
            pass
    
    # Try random restarts with different seeds
    np.random.seed(42)
    for seed in range(5):
        if time.time() - start_time > DEADLINE - 15:
            break
        try:
            points = _random_initialization()
            points, radius = _optimize_from_initial(points, target_r, max_time=15.0)
            if radius > best_radius:
                best_radius = radius
                best_points = points.copy()
        except Exception:
            pass
    
    # Final refinement with a slightly lower target to ensure validity
    if best_points is not None:
        if time.time() - start_time < DEADLINE - 5:
            # Use the computed radius as target for final polish
            final_target = best_radius * 0.99
            try:
                result = minimize(
                    _energy,
                    best_points,
                    jac=_grad_energy,
                    args=(final_target,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                refined_points = result.x
                refined_radius = _compute_radius(refined_points)
                if refined_radius > best_radius:
                    best_radius = refined_radius
                    best_points = refined_points
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return hexagonal lattice
        best_points = _hexagonal_initialization()
    
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = _hexagonal_initialization()
    
    return best_points
