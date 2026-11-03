"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def _compute_radii(centers):
    """
    Given centers, compute radii such that each circle is tangent to its nearest neighbors
    and within the unit square. We use a Delaunay-based approach:
    For each vertex, the radius is half the minimum distance to its Delaunay neighbors,
    but also bounded by distance to boundary.
    """
    n = len(centers)
    if n < 3:
        return np.full(n, 0.01)
    
    try:
        tri = Delaunay(centers)
    except Exception:
        return np.full(n, 0.01)
    
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundary
        d_boundary = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        
        # Find neighbors via Delaunay
        neighbors = []
        for simplex in tri.simplices:
            if i in simplex:
                for j in simplex:
                    if j != i:
                        neighbors.append(j)
        
        if not neighbors:
            # No neighbors, use boundary distance
            radii[i] = d_boundary
        else:
            min_dist = np.inf
            for j in neighbors:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
            # Radius is half the min distance to neighbor, but also bounded by boundary
            radii[i] = min(min_dist / 2.0, d_boundary)
        
        # Ensure positive radius
        radii[i] = max(radii[i], 1e-6)
    
    return radii

def _objective(centers):
    """
    Objective to minimize: negative sum of radii (we want to maximize sum of radii).
    We add a small penalty for any potential overlaps to guide optimization.
    """
    centers = centers.reshape(-1, 2)
    radii = _compute_radii(centers)
    
    # Check for overlaps and add penalty
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            required = radii[i] + radii[j]
            if d < required:
                penalty += (required - d) ** 2
    
    # Also check boundary violations
    for i in range(n):
        d_boundary = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        if radii[i] > d_boundary:
            penalty += (radii[i] - d_boundary) ** 2
    
    return -np.sum(radii) + 10.0 * penalty

def _initial_centers():
    """
    Generate initial center positions using a grid with some jitter.
    """
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add small jitter
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            # Clamp to [0.05, 0.95] to stay away from boundaries initially
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
    
    return np.array(centers[:N])

def construct():
    """
    Construct 26 circles in the unit square maximizing sum of radii.
    Returns array of shape (26, 3) with rows (x, y, radius).
    """
    np.random.seed(42)
    
    # Generate initial centers
    centers0 = _initial_centers()
    
    # Optimize center positions
    best_centers = centers0
    best_val = _objective(centers0)
    
    # Try multiple random restarts
    for restart in range(5):
        if restart == 0:
            c0 = centers0
        else:
            c0 = _initial_centers()
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(
            _objective,
            c0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_centers = result.x.reshape(-1, 2)
    
    # Compute final radii
    radii = _compute_radii(best_centers)
    
    # Ensure all circles are within the unit square
    for i in range(N):
        # Adjust radius to fit within boundaries
        d_boundary = min(best_centers[i][0], 1 - best_centers[i][0], 
                        best_centers[i][1], 1 - best_centers[i][1])
        radii[i] = min(radii[i], d_boundary)
        radii[i] = max(radii[i], 1e-6)
    
    # Final validation: adjust radii to ensure no overlaps
    # Simple iterative adjustment
    for iteration in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                required = radii[i] + radii[j]
                if d < required:
                    # Reduce the larger radius
                    overlap = (required - d) / 2.0
                    if radii[i] >= radii[j]:
                        radii[i] = max(radii[i] - overlap, 1e-6)
                    else:
                        radii[j] = max(radii[j] - overlap, 1e-6)
                    changed = True
        
        # Re-check boundary
        for i in range(N):
            d_boundary = min(best_centers[i][0], 1 - best_centers[i][0],
                            best_centers[i][1], 1 - best_centers[i][1])
            if radii[i] > d_boundary:
                radii[i] = d_boundary
                changed = True
        
        if not changed:
            break
    
    # Construct result array
    result = np.column_stack([best_centers, radii])
    
    return result
# EVOLVE-BLOCK-END
