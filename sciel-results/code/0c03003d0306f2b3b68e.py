"""Improved program for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s limit


def _compute_score(points):
    """Compute the radius for a given set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Distance from center
    dists = np.linalg.norm(points, axis=1)
    r_container = np.min(1.0 - dists)
    if r_container <= 0:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_pair = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists_pair, np.inf)
    r_pairs = np.min(dists_pair) / 2.0
    return min(r_container, r_pairs)


def _energy(points, r_target):
    """Energy function: penalize overlaps and going outside."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    # Pairwise overlap penalty
    pair_viol = np.maximum(0, 2 * r_target - dists)
    # Container violation
    cont_viol = np.maximum(0, np.linalg.norm(points, axis=1) + r_target - 1.0)
    return np.sum(pair_viol ** 2) + np.sum(cont_viol ** 2)


def _energy_grad(points, r_target):
    """Gradient of energy function."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    grad = np.zeros_like(points)
    
    # Pairwise forces
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 2 * r_target:
                force = 2 * (2 * r_target - d) / d
                grad[i] += force * (points[i] - points[j])
                grad[j] -= force * (points[i] - points[j])
    
    # Container forces
    for i in range(n):
        r = np.linalg.norm(points[i])
        if r + r_target > 1.0:
            if r > 1e-10:
                grad[i] += 2 * (r + r_target - 1.0) * points[i] / r
            else:
                grad[i] += 2 * (r_target - 1.0) * np.array([1.0, 0.0])
    
    return grad


def _get_hex_initial(r):
    """Generate hexagonal lattice points fitting inside the unit circle."""
    spacing = 2 * r
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Sort by distance from center, take closest N
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists, kind='stable')[:N]
        return points[idx]
    return None


def _get_ring_initial(r):
    """Generate ring-based initial configuration."""
    points = [[0.0, 0.0]]
    # Try different ring configurations
    for n_rings in range(1, 8):
        for angles_per_ring in range(1, 20):
            candidate = [[0.0, 0.0]]
            for ring in range(1, n_rings + 1):
                # Radius of this ring
                ring_r = ring * 2 * r * 0.55
                if ring_r > 1.0 - r:
                    break
                # Number of circles on this ring
                n_circ = max(1, int(2 * np.pi * ring_r / (2 * r)))
                for k in range(n_circ):
                    angle = 2 * np.pi * k / n_circ + np.pi / n_circ
                    x = ring_r * np.cos(angle)
                    y = ring_r * np.sin(angle)
                    if np.hypot(x, y) <= 1.0 - r:
                        candidate.append([x, y])
            if len(candidate) >= N:
                # Check if valid
                cand = np.array(candidate[:N])
                if _compute_score(cand) > 0.9 * r:
                    return cand
    return None


def _optimize_positions(points, r_target, max_iter=2000):
    """Optimize positions for a given target radius using gradient descent."""
    x0 = points.flatten()
    
    def obj(x):
        return _energy(x.reshape(-1, 2), r_target)
    
    def obj_grad(x):
        return _energy_grad(x.reshape(-1, 2), r_target).flatten()
    
    result = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': 1e-8})
    return result.x.reshape(-1, 2)


def _binary_search_radius(points, r_low=0.05, r_high=0.13, tol=1e-9, max_iter=30):
    """Binary search for the maximum feasible radius."""
    best_points = points.copy()
    best_r = r_low
    
    for _ in range(max_iter):
        if r_high - r_low < tol:
            break
        r_mid = (r_low + r_high) / 2
        opt_pts = _optimize_positions(points, r_mid, max_iter=500)
        score = _compute_score(opt_pts)
        if score >= r_mid - 1e-10:
            best_r = r_mid
            best_points = opt_pts
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_r, best_points


def _local_refine(points, r_target, steps=1000):
    """Local refinement using simple gradient descent."""
    pts = points.copy()
    lr = 0.01
    for step in range(steps):
        grad = _energy_grad(pts, r_target)
        pts -= lr * grad
        # Clamp to valid region
        for i in range(len(pts)):
            r = np.linalg.norm(pts[i])
            if r > 1.0 - r_target:
                pts[i] = pts[i] / r * (1.0 - r_target)
        if step % 100 == 0 and step > 0:
            lr *= 0.9
    return pts


def construct():
    start_time = time.time()
    
    # Start with a reasonable initial radius
    r_init = 0.12
    
    # Generate initial points
    points = _get_hex_initial(r_init)
    if points is None:
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.8, N)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    best_r = _compute_score(points)
    best_points = points.copy()
    
    # Try ring-based initialization
    ring_pts = _get_ring_initial(r_init)
    if ring_pts is not None:
        ring_score = _compute_score(ring_pts)
        if ring_score > best_r:
            best_r = ring_score
            best_points = ring_pts
    
    # Multi-stage optimization
    # Stage 1: Optimize with current best radius
    if time.time() - start_time < DEADLINE - 30:
        opt_pts = _optimize_positions(best_points, best_r, max_iter=1000)
        score = _compute_score(opt_pts)
        if score > best_r:
            best_r = score
            best_points = opt_pts
    
    # Stage 2: Try to increase radius with binary search
    if time.time() - start_time < DEADLINE - 15:
        r_high = min(0.135, best_r * 1.1)
        r_low = best_r * 0.95
        for _ in range(15):  # Limited iterations to stay within time
            if time.time() - start_time > DEADLINE - 10:
                break
            r_mid = (r_low + r_high) / 2
            opt_pts = _optimize_positions(best_points, r_mid, max_iter=300)
            score = _compute_score(opt_pts)
            if score >= r_mid - 1e-10:
                best_r = score
                best_points = opt_pts
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Stage 3: Final local refinement
    if time.time() - start_time < DEADLINE - 5:
        refined = _local_refine(best_points, best_r, steps=500)
        score = _compute_score(refined)
        if score > best_r:
            best_r = score
            best_points = refined
    
    # Final check and return
    final_score = _compute_score(best_points)
    if final_score <= 0:
        # Fallback to hexagonal
        pts = _get_hex_initial(0.1)
        if pts is not None:
            return pts
        return best_points
    
    return best_points
