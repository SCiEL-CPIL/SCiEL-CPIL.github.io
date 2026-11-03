"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(pts):
    """Compute the common radius for a set of circle centers."""
    if len(pts) < 2:
        return 1.0
    # Distance to boundary
    dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    r_boundary = np.min(dist_to_boundary)
    
    # Minimum pairwise distance
    # Use a more efficient method for large N
    from scipy.spatial.distance import pdist
    min_pair_dist = np.min(pdist(pts))
    r_pair = min_pair_dist / 2.0
    
    return min(r_boundary, r_pair)


def _hex_lattice_init(n):
    """Generate initial points using a hexagonal lattice pattern."""
    # Generate a hexagonal lattice of points
    points = []
    # Start with a reasonable spacing
    # For n points in unit circle, rough estimate of spacing
    # Area per point ~ pi/n, so spacing ~ sqrt(pi/n)
    spacing = np.sqrt(np.pi / n) * 0.95
    
    # Generate hex lattice points
    max_r = 1.5
    n_max = int(max_r / spacing) + 2
    
    for j in range(-n_max, n_max + 1):
        for i in range(-n_max, n_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more points than needed, randomly select n
    if len(points) > n:
        # Prefer points closer to center for better packing
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add more points by reducing spacing
        spacing *= 0.9
        points = _hex_lattice_init(n)
    
    return points


def _local_optimize(pts, max_iter=100, tol=1e-8):
    """Perform local optimization to maximize the minimum distance."""
    n = len(pts)
    
    def objective(x):
        """Negative of the common radius (we minimize negative radius)."""
        pts = x.reshape(-1, 2)
        r = _compute_radius(pts)
        return -r
    
    # Initial gradient estimate
    x0 = pts.flatten()
    
    # Use Nelder-Mead or BFGS with bounds
    # For better performance, use a simpler local optimization
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Simple hill climbing with random perturbations
    step_size = 0.01
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for attempt in range(10):
                # Random perturbation
                delta = np.random.randn(2) * step_size
                new_pts = best_pts.copy()
                new_pts[i] += delta
                
                # Check if still inside unit circle
                if np.linalg.norm(new_pts[i]) > 1.0:
                    continue
                
                new_r = _compute_radius(new_pts)
                if new_r > best_r + tol:
                    best_pts = new_pts
                    best_r = new_r
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_pts, best_r


def construct():
    """Construct the packing of N equal circles in the unit circle."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial solution using hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Ensure valid shape
    if pts.shape != (N, 2):
        # Fallback to a simple grid
        pts = np.random.randn(N, 2) * 0.5
        pts = pts / np.max(np.linalg.norm(pts, axis=1, keepdims=True)) * 0.9
    
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            # First restart uses the hex lattice
            init_pts = _hex_lattice_init(N)
        else:
            # Other restarts use random or perturbed initializations
            if restart == 1:
                # Slightly perturbed hex lattice
                init_pts = _hex_lattice_init(N) + np.random.randn(N, 2) * 0.01
            elif restart == 2:
                # Ring-based initialization
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                radii = np.sqrt(np.random.rand(N)) * 0.9
                init_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            elif restart == 3:
                # Grid-based initialization
                grid_size = int(np.sqrt(N))
                xs = np.linspace(-0.8, 0.8, grid_size)
                ys = np.linspace(-0.8, 0.8, grid_size)
                xx, yy = np.meshgrid(xs, ys)
                pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
                # Filter to unit circle
                mask = np.linalg.norm(pts_grid, axis=1) <= 0.95
                pts_grid = pts_grid[mask]
                if len(pts_grid) >= N:
                    init_pts = pts_grid[np.random.choice(len(pts_grid), N, replace=False)]
                else:
                    init_pts = pts_grid
                    while len(init_pts) < N:
                        extra = np.random.randn(N - len(init_pts), 2) * 0.5
                        init_pts = np.vstack([init_pts, extra])
            else:
                # Random initialization
                init_pts = np.random.randn(N, 2) * 0.7
        
        # Ensure points are within unit circle
        norms = np.linalg.norm(init_pts, axis=1)
        init_pts[norms > 0.95] *= 0.95 / norms[norms > 0.95, None]
        
        # Local optimization
        opt_pts, opt_r = _local_optimize(init_pts, max_iter=50)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
    
    # Final refinement
    if time.time() < deadline - 5:
        best_pts, best_r = _local_optimize(best_pts, max_iter=30, tol=1e-9)
    
    return best_pts

# EVOLVE-BLOCK-END
