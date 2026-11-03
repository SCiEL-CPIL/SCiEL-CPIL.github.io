import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _hex_lattice_init():
    """Generate a hexagonal lattice initialization with 50 points."""
    # Start with a reasonable spacing
    spacing = 0.25
    while True:
        # Generate hex lattice points
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= (1 - spacing/2)**2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Sort by distance from center and take closest N
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            return points[idx]
        spacing *= 0.95

def _compute_score(points):
    """Compute the score: min(min_pairwise_dist/2, min(1 - |c_i|))."""
    n = len(points)
    if n != N:
        return 0.0
    
    # Check all points are finite
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check boundary: all points must be inside unit circle
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        return 0.0
    
    # Minimum distance from boundary
    min_boundary = 1.0 - np.max(norms)
    
    # Minimum pairwise distance (vectorized)
    # Calculate all pairwise squared distances
    # points shape (N, 2)
    # diff shape (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist = np.sqrt(np.min(dist_sq))
    
    return min(min_dist / 2.0, min_boundary)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s limit
    
    # Start with hex lattice
    points = _hex_lattice_init()
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Define a fast, smooth objective for local optimization
    # We use a soft-min approximation to make the objective differentiable
    # max r = min(d_ij/2, 1-|c_i|)
    # We minimize -r. 
    # To handle the min() non-differentiability, we can use a log-sum-exp trick or just rely on L-BFGS-B's robustness to kinks if we provide gradients? 
    # L-BFGS-B doesn't accept gradients here, so it uses numerical. 
    # A common trick is to minimize the sum of "violations" or use a smooth approximation of the minimum.
    # Let's stick to the exact score but vectorize it heavily. L-BFGS-B will struggle with the kink, but it's often effective for packing if started well.
    
    def smooth_objective(x):
        pts = x.reshape(N, 2)
        if not np.all(np.isfinite(pts)):
            return 1e6
        
        # Check boundary
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm > 1.0:
            # Penalize being outside
            return 1e6 + (max_norm - 1.0)
        
        # Compute pairwise distances vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # The score is min(sqrt(min_dist_sq)/2, 1 - max_norm)
        # We want to maximize this.
        # To help the optimizer, we can return a smooth approximation of the minimum.
        # min(a, b) approx -log(exp(-beta*a) + exp(-beta*b))/beta for large beta.
        # However, simple L-BFGS-B on the exact non-smooth function often works reasonably well for packing if we do many restarts.
        # Let's try the exact value first, as it's simpler and vectorized.
        min_dist = np.sqrt(np.min(dist_sq))
        score = min(min_dist / 2.0, 1.0 - max_norm)
        return -score
    
    # Multi-restart local optimization
    # We will run many short local optimizations from different starting points.
    # 1. From hex lattice
    # 2. From random points (uniform in disk)
    # 3. From perturbations of best points
    
    rng = np.random.default_rng(42)
    
    # Helper to run local optimization
    def run_local_opt(x0, maxiter=100):
        try:
            res = minimize(
                smooth_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
            )
            if res.fun < 0:
                new_points = res.x.reshape(N, 2)
                return new_points, -res.fun
        except Exception:
            pass
        return None, -np.inf

    # 1. Optimize from hex lattice
    if time.time() - start_time < deadline - 5:
        pts, score = run_local_opt(best_points.flatten(), maxiter=200)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()

    # 2. Random restarts
    # Generate random points inside unit circle
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 5:
            break
        
        # Generate random points
        # Uniform distribution in circle
        r = rng.random(N) ** 0.5
        theta = 2 * np.pi * rng.random(N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x0 = np.column_stack((x, y)).flatten()
        
        # Optimize
        pts, score = run_local_opt(x0, maxiter=100)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
            
            # If we found a significantly better solution, do a quick polish
            if time.time() - start_time < deadline - 5:
                pts2, score2 = run_local_opt(best_points.flatten(), maxiter=50)
                if pts2 is not None and score2 > best_score:
                    best_score = score2
                    best_points = pts2.copy()

    # 3. Final polish on the best solution found
    if time.time() - start_time < deadline - 2:
        pts, score = run_local_opt(best_points.flatten(), maxiter=300)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()

    # Final validation
    final_score = _compute_score(best_points)
    if final_score <= 0:
        # Fallback to hex lattice
        return _hex_lattice_init()
    
    return best_points
