"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _covering_radius(pts):
    """Compute covering radius in radians using convex hull approach."""
    pts = np.array(pts)
    pts = np.array([_normalize(p) for p in pts])
    
    # Use the fact that the covering radius is half the maximum angular distance
    # between any two points in the Voronoi diagram, or equivalently,
    # we can sample the sphere and find the max distance to nearest point.
    
    # For accuracy, use a dense sampling approach
    # Generate many test points on the sphere
    n_samples = 20000
    # Use Fibonacci spiral for test points (dense)
    i = np.arange(n_samples)
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each test point, find nearest config point
    # Use chunked computation for memory efficiency
    max_dist = 0.0
    chunk_size = 5000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        tp = test_pts[start:end]
        # Compute distances
        # dist = arccos(dot(tp, pts.T))
        dots = np.dot(tp, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        max_dist = max(max_dist, np.max(min_dists))
    
    return max_dist


def _objective(pts_flat):
    """Objective function for optimization: covering radius in radians."""
    pts = pts_flat.reshape(-1, 3)
    # Normalize
    pts = np.array([_normalize(p) for p in pts])
    return _covering_radius(pts)


def _repulsion_energy(pts):
    """Energy that penalizes close points and large gaps."""
    pts = np.array([_normalize(p) for p in pts])
    n = len(pts)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.dot(pts[i], pts[j])
            d = np.clip(d, -1.0, 1.0)
            angle = np.arccos(d)
            # Repulsion when too close, attraction when too far
            # Target separation ~ sqrt(4*pi/N)
            target = np.sqrt(4 * np.pi / N)
            # Soft penalty
            energy += (angle - target) ** 2
    return energy


def _local_optimize(pts, max_iter=5000, tol=1e-8):
    """Local optimization using gradient descent on covering radius proxy."""
    pts = np.array([_normalize(p) for p in pts])
    n = len(pts)
    
    best_pts = pts.copy()
    best_energy = _repulsion_energy(pts)
    
    lr = 0.01
    for it in range(max_iter):
        # Compute forces: each point is pushed away from its neighbors
        # and toward the center of the largest gap
        forces = np.zeros_like(pts)
        
        for i in range(n):
            # Repulsion from all other points
            for j in range(n):
                if i == j:
                    continue
                d = np.dot(pts[i], pts[j])
                d = np.clip(d, -1.0, 1.0)
                angle = np.arccos(d)
                # Repulsion force magnitude
                f = 1.0 / (angle + 0.01)
                # Direction: away from j
                direction = pts[i] - d * pts[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    forces[i] += f * direction
            
            # Attraction to fill gaps: pull toward antipode of nearest neighbor
            # Find nearest neighbor
            dists = []
            for j in range(n):
                if i == j:
                    continue
                d = np.dot(pts[i], pts[j])
                d = np.clip(d, -1.0, 1.0)
                dists.append((np.arccos(d), j))
            dists.sort()
            nearest_j = dists[0][1]
            # Pull slightly toward the gap direction (away from nearest)
            gap_dir = -pts[nearest_j]
            forces[i] += 0.1 * gap_dir
        
        # Update
        for i in range(n):
            pts[i] = pts[i] + lr * forces[i]
            pts[i] = _normalize(pts[i])
        
        energy = _repulsion_energy(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.99
    
    return best_pts


def construct():
    """Construct 50 points on the unit sphere minimizing covering radius."""
    np.random.seed(42)
    
    # Start with a well-distributed initial configuration
    # Use a combination of icosahedral vertices and additional points
    
    # Icosahedron vertices (12 points)
    phi_golden = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi_golden], [0, 1, -phi_golden],
        [0, -1, phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [1, -phi_golden, 0],
        [-1, phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [phi_golden, 0, -1],
        [-phi_golden, 0, 1], [-phi_golden, 0, -1],
    ])
    ico = np.array([_normalize(p) for p in ico])
    
    # Add more points using Fibonacci spiral for the remaining 38
    remaining = N - 12
    i = np.arange(remaining) + 0.5
    z = 1.0 - 2.0 * i / remaining
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Combine
    pts = np.vstack([ico, fib_pts])
    
    # Local optimization
    pts = _local_optimize(pts, max_iter=3000)
    
    # Final refinement using scipy
    pts_flat = pts.flatten()
    
    # Use Nelder-Mead for final polish
    try:
        result = minimize(
            _objective,
            pts_flat,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if result.fun < _covering_radius(pts):
            pts = result.x.reshape(-1, 3)
            pts = np.array([_normalize(p) for p in pts])
    except:
        pass
    
    # Ensure valid output
    pts = np.array([_normalize(p) for p in pts])
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.dot(pts[i], pts[j])
            if d > 0.9999:
                # Perturb
                pts[j] = pts[j] + np.random.normal(0, 0.01, 3)
                pts[j] = _normalize(pts[j])
    
    return pts
# EVOLVE-BLOCK-END
