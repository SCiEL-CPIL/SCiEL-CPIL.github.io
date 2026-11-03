"""Packing 50 equal spheres in a unit cube using BCC init + L-BFGS-B optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_wall, min_dist / 2.0)


def _bcc_init(n):
    """Generate n points from a BCC lattice scaled to fit in [-0.5, 0.5]^3."""
    # BCC lattice: points at (0,0,0) and (0.5,0.5,0.5) in each unit cell
    # We'll generate a grid of BCC points and take the first n that fit
    
    # For a cube of side L, the BCC lattice has lattice constant a
    # such that the nearest neighbor distance is a*sqrt(3)/2
    # We want to fit n points
    
    # Let's try different lattice constants
    best_centers = None
    best_score = 0.0
    
    # The BCC unit cell has 2 atoms. For n points, we need n/2 unit cells.
    # If we use a cubic block of m x m x m unit cells, we get 2*m^3 points.
    # For n=50, m=3 gives 54 points, m=2 gives 16 points.
    
    # Let's generate a larger BCC grid and pick the best subset
    m = 4  # 4x4x4 BCC cells = 128 points
    a = 1.0 / m  # lattice constant so that m*a = 1 (cube side)
    
    # Generate BCC points
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner point
                x = (i + 0.5) * a - 0.5
                y = (j + 0.5) * a - 0.5
                z = (k + 0.5) * a - 0.5
                points.append([x, y, z])
                # Body center
                x2 = (i + 1.0) * a - 0.5
                y2 = (j + 1.0) * a - 0.5
                z2 = (k + 1.0) * a - 0.5
                points.append([x2, y2, z2])
    
    points = np.array(points)
    
    # We have 128 points, need 50
    # Use a greedy approach: start with all points, remove the ones that
    # are most "crowded" (have smallest nearest neighbor distance)
    # Actually, let's just take a random subset and optimize
    
    # Better: use all points but only keep 50
    # Let's use a deterministic selection: take points in order but skip some
    
    # Simple approach: take first 50 points from the BCC grid
    # But this might not be optimal. Let's use a more careful selection.
    
    # For now, let's just take a random subset with a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.choice(len(points), n, replace=False)
    centers = points[indices].copy()
    
    return centers


def _optimize(centers, max_time=120.0):
    """Optimize centers using L-BFGS-B to maximize the minimum radius."""
    start_time = time.time()
    deadline = start_time + max_time - 5.0  # 5s safety margin
    
    def objective(x):
        """Minimize negative radius (i.e., maximize radius)."""
        c = x.reshape(N, 3)
        # Check bounds
        if np.any(np.abs(c) > 0.5):
            return 1e6
        # Compute radius
        r = _score(c)
        return -r
    
    # Initial centers
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_score = _score(centers)
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                   'maxls': 50})
        
        if time.time() < deadline:
            c = result.x.reshape(N, 3)
            s = _score(c)
            if s > best_score:
                best_score = s
                best_x = result.x.copy()
    except Exception:
        pass
    
    # If we still have time, try another round with different initial conditions
    if time.time() < deadline:
        try:
            # Perturb the best solution and optimize again
            rng = np.random.RandomState(123)
            perturbed = best_x.reshape(N, 3) + rng.randn(N, 3) * 0.01
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            result = minimize(objective, perturbed.flatten(), method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                      'maxls': 50})
            
            if time.time() < deadline:
                c = result.x.reshape(N, 3)
                s = _score(c)
                if s > best_score:
                    best_score = s
                    best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 3)


def construct():
    """Construct centers of 50 equal spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 120.0 - 10.0  # 10s safety margin
    
    # Initialize with BCC lattice
    centers = _bcc_init(N)
    
    # Check time
    if time.time() > deadline:
        return centers
    
    # Optimize
    centers = _optimize(centers, max_time=120.0)
    
    # Final validation
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    # Ensure centers are within bounds
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
