import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    
    # Minimum pairwise distance
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    
    return min(min_dist / 2.0, min_boundary)

def _neg_score(points):
    """Negative score for minimization."""
    return -_compute_score(points)

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with hex lattice
    points = _hex_lattice_init()
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Try differential evolution first (global optimization)
    # We optimize over 2*N coordinates directly
    def objective(x):
        pts = x.reshape(N, 2)
        return -_compute_score(pts)
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Run differential evolution with limited budget
    if time.time() - start_time < deadline - 30:
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                init='sobol',
                polish=True,
                x0=points.flatten(),
                workers=1
            )
            if result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B on the best points found
    if time.time() - start_time < deadline - 10:
        # Use a smooth approximation of the min function for gradient-based optimization
        # We want to maximize min(min_pairwise/2, 1 - max_norm)
        # This is equivalent to maximizing r such that all constraints are satisfied
        
        def smooth_objective(x):
            pts = x.reshape(N, 2)
            if not np.all(np.isfinite(pts)):
                return 1e6
            
            # Check boundary
            norms = np.linalg.norm(pts, axis=1)
            max_norm = np.max(norms)
            if max_norm > 1.0:
                return 1e6 + (max_norm - 1.0)
            
            # Compute pairwise distances using a smooth min approximation
            # For N=50, we have 1225 pairs
            min_dist_sq = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d2 = np.sum((pts[i] - pts[j])**2)
                    if d2 < min_dist_sq:
                        min_dist_sq = d2
            
            # The score is min(sqrt(min_dist_sq)/2, 1 - max_norm)
            # We want to maximize this, so minimize its negative
            score = min(np.sqrt(min_dist_sq) / 2.0, 1.0 - max_norm)
            return -score
        
        # Multi-restart local optimization
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - start_time > deadline - 5:
                break
            
            if restart == 0:
                x0 = best_points.flatten()
            else:
                # Perturb the best solution
                x0 = best_points.flatten() + 0.05 * np.random.randn(2 * N)
                x0 = np.clip(x0, -1, 1)
            
            try:
                res = minimize(
                    smooth_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < 0:
                    new_points = res.x.reshape(N, 2)
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass
        
        # One more round of refinement with tighter tolerance
        if time.time() - start_time < deadline - 3:
            try:
                res = minimize(
                    smooth_objective,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 0:
                    new_points = res.x.reshape(N, 2)
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass
    
    # Final validation
    final_score = _compute_score(best_points)
    if final_score <= 0:
        # Fallback to hex lattice
        return _hex_lattice_init()
    
    return best_points
