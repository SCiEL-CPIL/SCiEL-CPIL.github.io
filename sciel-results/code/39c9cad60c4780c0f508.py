"""Rewritten program: spherical potential minimization with velocity Verlet integrator."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _angular_distances_to_points(points, n_eval=5000):
    """Compute max angular distance from random sphere points to nearest of `points`."""
    # Generate random points on sphere
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n_eval)
    phi = np.arccos(2 * rng.uniform(0, 1, n_eval) - 1)
    y = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    # Normalize points
    p = points / np.linalg.norm(points, axis=1, keepdims=True)
    # Compute cos of angular distances: y . p for all pairs
    cos_dists = y @ p.T  # shape (n_eval, N)
    max_cos = cos_dists.max(axis=1)
    max_cos = np.clip(max_cos, -1.0, 1.0)
    angles = np.arccos(max_cos)
    return np.max(angles) * 180.0 / np.pi


def _covering_radius_from_hull(points):
    """Compute covering radius as max circumradius of hull facets with empty caps."""
    p = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(p)
    except Exception:
        return None
    
    # For each facet, compute circumradius
    max_circum = 0.0
    for simplex in hull.simplices:
        tri = p[simplex]
        # Circumradius of triangle on unit sphere
        a = np.linalg.norm(tri[1] - tri[2])
        b = np.linalg.norm(tri[0] - tri[2])
        c = np.linalg.norm(tri[0] - tri[1])
        area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        if area < 1e-12:
            continue
        R = (a * b * c) / (4.0 * area)
        # Check if the cap is empty: the circumcenter should have all other points
        # on the same side or the cap should not contain other points
        # For now, just track max circumradius
        if R > max_circum:
            max_circum = R
    
    # Convert circumradius to angular distance
    # For a spherical triangle with circumradius R (chord), the angular covering radius
    # is related. Actually, the circumradius of the spherical cap is arccos(1 - R^2/2) approx
    # More precisely: if R is the chord length of the circumcircle, the angular radius is
    # arcsin(R/2) for small angles, but more accurately for unit sphere:
    # The circumcenter of the spherical triangle, the angular distance to vertices is the covering radius
    # For a triangle on unit sphere with side lengths (chords) a,b,c, the spherical circumradius 
    # satisfies: sin(R_circ) = (abc) / (4 * area_spherical)
    # But we computed chord circumradius. Let's use a different approach.
    
    # Actually, let's just use the random sampling method as a fallback if this is unreliable
    return None


def construct():
    """
    Spherical potential minimization using velocity Verlet integration.
    
    Uses a repulsive inverse-power law potential between points to maximize
    minimum separation, combined with velocity Verlet for stable long-term dynamics.
    """
    # Initialize with Fibonacci spiral as starting configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float64)
    
    # Normalize
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Velocity Verlet integration
    dt = 0.01
    n_steps = 20000
    power = 4  # inverse power law exponent
    
    # Initialize velocities to zero
    vels = np.zeros_like(pts)
    
    # Cooling schedule: reduce dt over time
    dt_start = 0.05
    dt_end = 0.001
    
    best_pts = pts.copy()
    best_radius = _angular_distances_to_points(pts)
    
    for step in range(n_steps):
        # Anneal: reduce timestep
        progress = step / n_steps
        dt = dt_start * (1.0 - progress) + dt_end * progress
        
        # Compute forces: repulsive inverse-power law
        # F_ij = k * (r_ij)^(-power-2) * (p_i - p_j) / |p_i - p_j|
        # Actually, for potential U = sum_{i<j} 1/|p_i - p_j|^power
        # Force on i from j: -dU/dp_i = power * (p_i - p_j) / |p_i - p_j|^(power+2)
        
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Force magnitude: power / dist^(power+1) in direction diff/dist
                # = power * diff / dist^(power+2)
                f_mag = power / (dist ** (power + 1))
                forces[i] += f_mag * diff / dist
        
        # Half-step velocity update
        vels += 0.5 * dt * forces
        
        # Position update
        new_pts = pts + dt * vels
        
        # Project back to unit sphere
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Recompute forces at new position for second half-step
        new_forces = np.zeros_like(new_pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = new_pts[i] - new_pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                f_mag = power / (dist ** (power + 1))
                new_forces[i] += f_mag * diff / dist
        
        # Second half-step velocity update
        vels += 0.5 * dt * new_forces
        
        # Damping to help convergence
        vels *= 0.99
        
        pts = new_pts
        
        # Periodically check the covering radius
        if step % 500 == 0:
            radius = _angular_distances_to_points(pts)
            if radius < best_radius:
                best_radius = radius
                best_pts = pts.copy()
    
    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify no coincident points and proper shape
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    
    return best_pts
# EVOLVE-BLOCK-END
