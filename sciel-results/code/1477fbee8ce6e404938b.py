"""Optimized program for Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 145.0  # safety margin before 150s kill

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = pts.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area. Higher is better."""
    hull = _hull_area(pts)
    if hull < 1e-10:
        return 0.0
    return _min_triangle_area(pts) / hull

def _negative_score(pts):
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_points = None
    
    # Strategy: parameterize points as radial + angular perturbations of a regular 13-gon
    # This maintains convexity and reduces search space
    
    def _params_to_points(params):
        """Convert (radii[13], angles[13]) to 2D points, maintaining convexity."""
        radii = params[:N]
        angles = params[N:]
        base_angles = 2.0 * np.pi * np.arange(N) / N
        actual_angles = base_angles + angles
        # Ensure angles are in order (convexity) by sorting if needed
        # Actually, small perturbations keep them ordered
        x = radii * np.cos(actual_angles)
        y = radii * np.sin(actual_angles)
        return np.column_stack([x, y])
    
    def _objective(params):
        pts = _params_to_points(params)
        hull = _hull_area(pts)
        if hull < 1e-10:
            return 1e10
        return -_min_triangle_area(pts) / hull
    
    # Initial guesses for parameters
    base_radii = np.ones(N)
    base_angles = np.zeros(N)
    
    # Run multiple optimization attempts
    attempts = 0
    max_attempts = 200
    
    while time.time() < deadline - 5.0:
        attempts += 1
        
        # Create diverse initial points
        if attempts <= 10:
            # Regular 13-gon with small random perturbations
            rng = np.random.RandomState(42 + attempts)
            radii = base_radii + 0.05 * rng.randn(N)
            angles = 0.02 * rng.randn(N)
        elif attempts <= 30:
            # More diverse: varying base radius
            rng = np.random.RandomState(100 + attempts)
            base_r = 0.8 + 0.4 * rng.random()
            radii = base_r * (1 + 0.1 * rng.randn(N))
            angles = 0.05 * rng.randn(N)
        else:
            # Perturb best solution found so far
            if best_points is not None:
                # Extract approximate radii and angles from best points
                cx, cy = np.mean(best_points, axis=0)
                dx = best_points[:, 0] - cx
                dy = best_points[:, 1] - cy
                radii = np.sqrt(dx**2 + dy**2)
                angles = np.arctan2(dy, dx)
                base_angles = 2.0 * np.pi * np.arange(N) / N
                angle_diffs = np.angle(np.exp(1j * (angles - base_angles)))
                rng = np.random.RandomState(200 + attempts)
                radii = radii * (1 + 0.05 * rng.randn(N))
                angles = angle_diffs + 0.03 * rng.randn(N)
            else:
                rng = np.random.RandomState(300 + attempts)
                radii = 1.0 + 0.1 * rng.randn(N)
                angles = 0.05 * rng.randn(N)
        
        # Ensure radii are positive
        radii = np.maximum(radii, 0.3)
        
        initial_params = np.concatenate([radii, angles])
        
        # Use L-BFGS-B for local optimization
        try:
            result = minimize(
                _objective,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e9:
                opt_points = _params_to_points(result.x)
                s = _score(opt_points)
                if s > best_score:
                    best_score = s
                    best_points = opt_points
        except Exception:
            pass
        
        # Check deadline
        if time.time() > deadline - 10.0:
            break
    
    # Final fallback: if no good solution found, use regular 13-gon
    if best_points is None:
        angles = 2.0 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
    
    # Normalize: scale so hull area = 1
    hull = _hull_area(best_points)
    if hull > 1e-10:
        best_points = best_points / np.sqrt(hull)
    
    return best_points
