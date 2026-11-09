"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    d = pdist(points)
    min_dist = np.min(d) / 2.0
    return min(min_boundary, min_dist)


def _construct_concentric_rings():
    """Construct concentric rings packing."""
    best_r = 0.0
    best_points = None
    
    # Try different ring configurations
    for n_rings in range(3, 12):
        for base_count in range(1, 15):
            ring_counts = []
            remaining = N
            for i in range(n_rings):
                count = base_count + i * 2
                if remaining <= 0:
                    break
                ring_counts.append(min(count, remaining))
                remaining -= count
            
            if len(ring_counts) == 0 or sum(ring_counts) < N:
                continue
            
            # Build points
            points = []
            # Center point if we have odd total
            if sum(ring_counts) % 2 == 1 and len(ring_counts) > 0:
                points.append([0.0, 0.0])
                ring_counts[0] -= 1
                if ring_counts[0] == 0:
                    ring_counts.pop(0)
            
            for i, count in enumerate(ring_counts):
                if count <= 0:
                    continue
                r = (i + 1) / (len(ring_counts) + 0.5) * 0.95
                angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + i * np.pi / max(count, 1)
                for a in angles:
                    points.append([r * np.cos(a), r * np.sin(a)])
            
            points = np.array(points[:N])
            if len(points) < N:
                continue
            
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    return best_points


def _optimize(points, time_limit=10.0):
    """Local optimization to improve packing."""
    start_time = time.time()
    
    def neg_score(x):
        pts = x.reshape(-1, 2)
        return -_compute_radius(pts)
    
    x0 = points.flatten()
    result = minimize(neg_score, x0, method='L-BFGS-B', 
                     options={'maxiter': 500, 'gtol': 1e-6})
    
    elapsed = time.time() - start_time
    if result.success and elapsed < time_limit:
        return result.x.reshape(-1, 2)
    return points


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with concentric rings
    points = _construct_concentric_rings()
    
    if points is None:
        # Fallback to hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= N:
                points = pts[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
    
    # Local optimization
    if time.time() < deadline - 20:
        points = _optimize(points, time_limit=120.0)
    
    # Verify and return
    r = _compute_radius(points)
    if r <= 0:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return points
# EVOLVE-BLOCK-END
