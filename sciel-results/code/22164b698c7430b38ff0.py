"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _fibonacci_lattice(n, seed=0):
    """Generate n points using a Fibonacci lattice (sunflower pattern) in [-0.5, 0.5]^2."""
    rng = np.random.default_rng(seed)
    # Use golden angle for uniform distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    t = np.linspace(0, 1, n, endpoint=False)
    # Add small random perturbation to break symmetry
    t = t + rng.uniform(-0.001, 0.001, n)
    t = np.clip(t, 0, 1)
    
    r = np.sqrt(t)
    theta = 2 * np.pi * t * (1 + 0.5 * (1 - t))  # slight spiral variation
    # Actually, standard sunflower: theta = i * golden_angle
    theta = np.arange(n) * golden_angle
    r = np.sqrt(np.arange(n) / n)
    
    x = r * np.cos(theta) * 0.5
    y = r * np.sin(theta) * 0.5
    return np.column_stack([x, y])


def _compute_radius(pts):
    """Compute the maximum radius for the given point configuration."""
    if len(pts) < 2:
        return 0.0
    # Minimum distance between pairs
    from scipy.spatial.distance import pdist
    d = pdist(pts, 'euclidean')
    min_dist = d.min() if len(d) > 0 else np.inf
    # Minimum distance to boundary
    dist_boundary = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = dist_boundary.min()
    return min(min_dist / 2.0, min_boundary)


def _repulsive_forces(pts, k=10.0, eps=1e-8):
    """Compute repulsive forces between all pairs of points."""
    n = len(pts)
    forces = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < eps:
                # Random small perturbation
                diff = np.random.randn(2) * eps
                dist = eps
            # Repulsive force: F = k / dist^2 * unit_vector
            force_mag = k / (dist * dist)
            force = force_mag * diff / dist
            forces[i] += force
            forces[j] -= force
    return forces


def _boundary_forces(pts, k_b=50.0):
    """Compute forces pushing points away from boundaries."""
    forces = np.zeros_like(pts)
    margin = 0.01
    for i in range(len(pts)):
        for dim in range(2):
            x = pts[i, dim]
            if x > 0.5 - margin:
                forces[i, dim] -= k_b * (x - (0.5 - margin))
            elif x < -0.5 + margin:
                forces[i, dim] += k_b * ((-0.5 + margin) - x)
    return forces


def construct():
    start_time = time.time()
    
    # Initialize with Fibonacci lattice
    pts = _fibonacci_lattice(N, seed=42)
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Repulsive force-based optimization
    lr = 0.001
    decay = 0.9995
    iterations = 0
    
    while time.time() - start_time < DEADLINE - 5.0:
        # Compute forces
        forces = _repulsive_forces(pts, k=10.0) + _boundary_forces(pts, k_b=50.0)
        
        # Update positions
        pts = pts + lr * forces
        
        # Keep points within the square
        pts[:, 0] = np.clip(pts[:, 0], -0.4999, 0.4999)
        pts[:, 1] = np.clip(pts[:, 1], -0.4999, 0.4999)
        
        iterations += 1
        
        # Periodically check radius
        if iterations % 50 == 0:
            r = _compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_pts = pts.copy()
        
        # Decay learning rate
        lr *= decay
    
    # Final polish: local refinement using gradient ascent on the radius
    # Use a few iterations of projected gradient
    pts = best_pts.copy()
    lr2 = 0.0001
    for _ in range(2000):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        forces = _repulsive_forces(pts, k=5.0) + _boundary_forces(pts, k_b=20.0)
        pts = pts + lr2 * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.4999, 0.4999)
        pts[:, 1] = np.clip(pts[:, 1], -0.4999, 0.4999)
        lr2 *= 0.999
    
    r_final = _compute_radius(pts)
    if r_final > best_radius:
        best_radius = r_final
        best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
