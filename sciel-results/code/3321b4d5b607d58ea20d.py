"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in degrees for points on unit sphere."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    # Find the largest circumradius of facets - approximate via farthest point
    # For a set of points on sphere, covering radius = max over sphere of min angular dist
    # We approximate by checking a dense grid
    # Actually, for the convex hull approach, we need to find the largest empty cap
    # Simple approximation: sample many points and find max min-distance
    # Better: use the fact that covering radius is determined by the largest face
    # of the convex hull. For now, use a practical approximation.
    
    # Compute pairwise distances to find local structure
    # Use a grid-based approach for speed
    # Sample the sphere with a fine grid
    n_theta = 60
    n_phi = 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    gx = np.sin(TH) * np.cos(PH)
    gy = np.sin(TH) * np.sin(PH)
    gz = np.cos(TH)
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    
    # For each grid point, find min distance to any of the N points
    # Use matrix multiplication for speed
    # dist = 1 - dot product, angular dist = arccos(dot)
    dots = grid @ points.T  # shape (n_grid, N)
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    max_angle = np.arccos(min_dots)
    return np.degrees(max_angle)

def _energy(points):
    """Energy function to minimize - based on covering radius proxy."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    # Use a sum of inverse powers of nearest-neighbor distances as a smooth proxy
    # This encourages uniform distribution
    dots = points @ points.T
    np.fill_diagonal(dots, -1)
    # For each point, find its nearest neighbor (excluding self)
    min_dots = np.max(dots, axis=1)  # max dot with others = min angle
    # Energy: sum of (1 - min_dot)^(-p) to penalize close points
    # Actually, we want to MAXIMIZE min separation, so minimize max covering radius
    # Use a different approach: minimize the sum of squared distances to nearest neighbor
    # This is the "thomson problem" style energy
    return -np.sum(min_dots)  # maximize sum of nearest-neighbor cosines

def _energy_grad(points):
    """Gradient of the energy function."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    dots = points @ points.T
    np.fill_diagonal(dots, -1)
    min_dots = np.max(dots, axis=1)
    
    # Gradient: for each point i, the contribution to energy from its nearest neighbor j
    # d(-min_dot)/d(points[i]) = -points[j] / |points[j]| (approximately, since points are on sphere)
    # More precisely, if j is the nearest neighbor of i, then:
    # d/dx_i (x_i . x_j) = x_j, but we're on the sphere, so project
    grad = np.zeros_like(points)
    for i in range(N):
        j = np.argmax(dots[i])
        grad[i] = -points[j]
    # Project gradient to tangent space
    for i in range(N):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    return grad

def _make_tangent(points, v):
    """Project v onto tangent space at each point."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    v = v - (v * points).sum(axis=1, keepdims=True) * points
    return v

def _retract(points, v, step):
    """Retract: move points along tangent vectors and renormalize."""
    new_points = points + step * v
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Leave 10s margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Gradient descent on energy (maximize nearest-neighbor separation)
    alpha = 0.01
    for it in range(5000):
        if time.time() > deadline - 30.0:
            break
        
        grad = _energy_grad(points)
        points = _retract(points, grad, alpha)
        
        # Adaptive step size
        if it % 500 == 0:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
            # Reduce step size periodically
            alpha *= 0.95
    
    # Phase 2: Polish with smaller steps
    alpha = 0.001
    for it in range(3000):
        if time.time() > deadline - 10.0:
            break
        
        grad = _energy_grad(points)
        points = _retract(points, grad, alpha)
        
        if it % 300 == 0:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
            alpha *= 0.98
    
    # Final polish
    alpha = 0.0001
    for it in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        grad = _energy_grad(points)
        points = _retract(points, grad, alpha)
        
        if it % 200 == 0:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Check if we should use best_points or current points
    final_cr = _covering_radius(points)
    if final_cr < best_cr:
        best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
