"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _initial_hex_lattice(n):
    """Generate a hexagonal lattice of points inside the unit disk."""
    # Estimate spacing
    # Area of disk = pi. Area per point ~ s^2 * sqrt(3)/2.
    # n * s^2 * sqrt(3)/2 ~ pi => s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.98
    
    points = []
    k = int(2.0 / s) + 5
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback spiral
        theta = np.linspace(0, 2 * np.pi * 10, n)
        r = np.linspace(0, 1, n)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')
        points = points[idx[:n]]
    return points

def _compute_radius(points):
    """Compute the max possible radius for given centers."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _relax(points, r, max_iter=200, learning_rate=0.01, noise=0.0):
    """
    Relax points to satisfy constraints for radius r.
    Returns (success, points).
    """
    n = len(points)
    points = points.copy()
    
    for _ in range(max_iter):
        # Build KDTree
        tree = cKDTree(points)
        
        # Find neighbors within 2.5 * 2r
        neighbors = tree.query_ball_point(points, r=2.5 * 2 * r)
        
        forces = np.zeros_like(points)
        
        for i in range(n):
            for j in neighbors[i]:
                if i == j:
                    continue
                # Avoid double counting or just accumulate
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r:
                    # Repulsive force
                    # F = k * (2r - dist) * direction
                    # Direction is diff / dist
                    if dist > 1e-9:
                        forces[i] += (2 * r - dist) * (diff / dist)
        
        # Boundary forces
        radii = np.linalg.norm(points, axis=1)
        for i in range(n):
            if radii[i] > 1.0 - r:
                # Inward force
                # F = k * (radius - (1-r)) * (-direction)
                # Direction is points[i] / radii[i]
                if radii[i] > 1e-9:
                    forces[i] += (radii[i] - (1.0 - r)) * (-points[i] / radii[i])
        
        # Update
        if noise > 0:
            forces += np.random.normal(0, noise, size=points.shape)
        
        points += learning_rate * forces
        
        # Check convergence
        # Max violation
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        current_r = min(min_pair, min_boundary)
        
        if current_r >= r - 1e-6:
            return True, points
            
    return False, points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial guess
    points = _initial_hex_lattice(N)
    r = _compute_radius(points)
    
    best_r = r
    best_points = points.copy()
    
    # Incremental radius increase with relaxation
    # Start with a small step
    dr = 0.0001
    failed_attempts = 0
    
    while time.time() < deadline:
        # Try to increase r
        target_r = r + dr
        
        # Relax
        success, new_points = _relax(points, target_r, max_iter=100, learning_rate=0.05, noise=dr*0.1)
        
        if success:
            # Accept
            points = new_points
            r = target_r
            if r > best_r:
                best_r = r
                best_points = points.copy()
            # Increase step size slightly if successful
            dr = min(dr * 1.1, 0.001)
            failed_attempts = 0
        else:
            # Failed, reduce step size and add noise to escape
            dr = dr * 0.5
            failed_attempts += 1
            
            # Apply random perturbation to points to escape local minimum
            points = points + np.random.normal(0, 0.01, size=points.shape)
            
            # Re-relax at current r to stabilize
            _, points = _relax(points, r, max_iter=50, learning_rate=0.05, noise=0.01)
            
            if failed_attempts > 20:
                # Restart with new random initialization?
                # Or just break if we've been stuck too long
                break
        
        if r > 0.0531: # Close to known best
             # Reduce step size for fine tuning
             dr = min(dr, 0.00001)
             
    # Final polish: try to squeeze out a bit more with a tighter relaxation
    # Use a smaller step size and more iterations
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        target_r = best_r + 0.00001
        success, new_points = _relax(best_points, target_r, max_iter=200, learning_rate=0.01, noise=0.001)
        if success:
            best_r = target_r
            best_points = new_points
        else:
            break
            
    # Ensure validity
    radii = np.linalg.norm(best_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
        
    return best_points
# EVOLVE-BLOCK-END
