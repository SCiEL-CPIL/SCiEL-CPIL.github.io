"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius (in radians) of points on the unit sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        n_theta = 300
        n_phi = 600
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(thetas, phis, indexing='ij')
        test_pts = np.column_stack([
            np.sin(T) * np.cos(P).ravel(),
            np.sin(T) * np.sin(P).ravel(),
            np.cos(T).ravel()
        ])
        dists, _ = tree.query(test_pts)
        max_chord = np.max(dists)
        max_angle = 2 * np.arcsin(min(max_chord / 2, 1.0))
        return max_angle
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        a, b, c = facet_pts
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            continue
        o = cross / norm_cross
        
        if np.dot(o, a) < 0:
            o = -o
        
        cos_angle = np.clip(np.dot(o, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        max_radius = max(max_radius, angle)
    
    return max_radius


def _electrostatic_energy(points, exponent=2.0):
    """Compute the electrostatic potential energy with repulsive forces."""
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    dots = np.clip(dots, -0.9999, 0.9999)
    
    # Use a repulsive kernel: E = sum_{i<j} 1/(1 - p_i·p_j)^exponent
    # This blows up as points get close, encouraging separation
    pairs = dots[~np.eye(len(points), dtype=bool)]
    energy = np.sum(1.0 / (1.0 - pairs) ** exponent)
    return energy


def _electrostatic_forces(points, exponent=2.0):
    """Compute repulsive forces on each point."""
    n = len(points)
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    dots = np.clip(dots, -0.9999, 0.9999)
    
    # Weight: w_ij = exponent / (1 - dots[i,j])^(exponent+1)
    weights = exponent / (1.0 - dots) ** (exponent + 1)
    np.fill_diagonal(weights, 0.0)
    
    # Force direction: p_i - (p_i·p_j)*p_j (component of p_i perpendicular to p_j)
    proj = dots[:, :, None] * points[None, :, :]
    perp = points[:, None, :] - proj
    
    perp_norms = np.linalg.norm(perp, axis=2, keepdims=True)
    perp_norms = np.maximum(perp_norms, 1e-15)
    perp_unit = perp / perp_norms
    
    force = np.sum(weights[:, :, None] * perp_unit, axis=1)
    return force


def _refine_electrostatic(points, n_iterations=5000, exponent=2.0, lr_init=0.05, lr_min=0.001):
    """Refine points using electrostatic repulsive force simulation."""
    points = points.copy()
    best_points = points.copy()
    best_energy = _electrostatic_energy(points, exponent)
    
    for it in range(n_iterations):
        # Compute forces
        force = _electrostatic_forces(points, exponent)
        
        # Normalize forces
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = force / force_norms
        
        # Learning rate with exponential decay
        progress = it / n_iterations
        lr = lr_init * (lr_min / lr_init) ** progress
        
        # Update points
        points = points + lr * force_unit
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Check energy periodically
        if it % 50 == 0:
            energy = _electrostatic_energy(points, exponent)
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
    
    return best_points, best_energy


def _refine_covering(points, n_iterations=3000):
    """Refine points to minimize covering radius directly."""
    points = points.copy()
    n = len(points)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(n_iterations):
        # Compute pairwise distances
        dots = points @ points.T
        np.fill_diagonal(dots, 0.0)
        dots = np.clip(dots, -0.9999, 0.9999)
        
        # Find nearest neighbors for each point
        min_dots = np.min(dots, axis=1)  # Most negative dot = farthest
        # Actually we want the closest: max dot
        max_dots = np.max(dots, axis=1)
        
        # Force: move away from nearest neighbors
        # For each point, find its 3 nearest neighbors and move away from them
        force = np.zeros_like(points)
        
        for i in range(n):
            # Find 3 nearest neighbors
            dists = 1.0 - dots[i]
            idx = np.argsort(dists)[:3]
            
            for j in idx:
                if i == j:
                    continue
                # Move away from point j
                perp = points[i] - dots[i, j] * points[j]
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-15:
                    force[i] += perp / perp_norm
        
        # Normalize force
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = force / force_norms
        
        # Learning rate
        lr = 0.02 * (1 - it / n_iterations)
        
        # Update
        points = points + lr * force_unit
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        if it % 100 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    return best_points, best_radius


def _fibonacci_spiral(n, offset=0.5):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on the unit sphere."""
    # Use normal distribution and normalize
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and keep the best
    best_points = None
    best_radius = np.inf
    
    # Initialization 1: Fibonacci spiral
    points1 = _fibonacci_spiral(N)
    points1, _ = _refine_electrostatic(points1, n_iterations=3000, exponent=2.0)
    points1, radius1 = _refine_covering(points1, n_iterations=2000)
    
    if radius1 < best_radius:
        best_radius = radius1
        best_points = points1.copy()
    
    # Initialization 2: Random with refinement
    points2 = _random_sphere(N, rng)
    points2, _ = _refine_electrostatic(points2, n_iterations=3000, exponent=2.0)
    points2, radius2 = _refine_covering(points2, n_iterations=2000)
    
    if radius2 < best_radius:
        best_radius = radius2
        best_points = points2.copy()
    
    # Initialization 3: Fibonacci with different offset
    points3 = _fibonacci_spiral(N, offset=0.3)
    points3, _ = _refine_electrostatic(points3, n_iterations=3000, exponent=2.5)
    points3, radius3 = _refine_covering(points3, n_iterations=2000)
    
    if radius3 < best_radius:
        best_radius = radius3
        best_points = points3.copy()
    
    # Final refinement on the best configuration
    best_points, final_radius = _refine_covering(best_points, n_iterations=1000)
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
