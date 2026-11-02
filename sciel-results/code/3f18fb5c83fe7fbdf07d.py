"""Rewritten program: maximize minimum pairwise distance on unit sphere for N=14, D=3."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _pairwise_distances(pts):
    """Compute all pairwise squared distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # Exclude diagonal
    sq_dists = sq_dists[~np.eye(N, dtype=bool)]
    return sq_dists


def _score(pts):
    """Score = (max pairwise dist / min pairwise dist)^2. Lower is better."""
    sq_dists = _pairwise_distances(pts)
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq < 1e-15:
        return np.inf
    return (max_sq / min_sq) ** 2


def _objective(x):
    """
    Objective: minimize -(log min pairwise distance) subject to points on unit sphere.
    We parameterize points on the unit sphere using angles, then maximize the minimum
    pairwise distance. This is equivalent to minimizing the ratio when all points are
    on the sphere (which is optimal for this problem).
    
    Actually, let's use a different approach: 
    - Place all points on the unit sphere
    - Maximize the minimum pairwise distance (Tammes problem)
    - The ratio max/min will then be 1 (all on sphere, max dist = 2, min dist = optimized)
    - Wait, no. If all points are on the sphere, max pairwise distance is at most 2 (diameter),
      and min pairwise distance is what we maximize. So ratio = 2/min_dist, ratio^2 = 4/min_dist^2.
    
    Better approach: directly minimize the ratio by optimizing point positions.
    Use a log-barrier or direct optimization.
    
    Let's use: minimize f(x) = max_i,j ||p_i - p_j||^2 / min_i,j ||p_i - p_j||^2
    This is non-smooth. Instead, use the approach from the knowledge:
    Powell's method on centered, unit-norm manifold.
    
    Alternative: Since N=14 is small, use a multi-start local optimization.
    Objective: minimize (max_dist^2 / min_dist^2)
    We can rewrite: minimize log(max_dist^2) - log(min_dist^2)
    But max and min are non-differentiable.
    
    Practical approach: 
    1. Start with good initial configurations (e.g., Fibonacci sphere, random)
    2. Use scipy.optimize.minimize with L-BFGS-B on flattened coordinates
    3. Objective: sum of (1/d_ij^2) for all pairs, which pushes all distances up
       but this doesn't directly minimize the ratio.
    
    Better: Use the fact that for optimal configurations in 3D for small N,
    points lie on the sphere. So constrain to unit sphere and maximize min distance.
    Then ratio = 2/min_dist, so minimizing ratio = maximizing min_dist.
    
    Let's do: maximize min_{i<j} ||p_i - p_j|| subject to ||p_i|| = 1 for all i.
    This is the Tammes problem. For N=14, known optimal min distance exists.
    
    Parameterization: use spherical coordinates (theta, phi) for each point.
    x = [theta_1, phi_1, ..., theta_N, phi_N]
    p_i = [sin(theta_i)cos(phi_i), sin(theta_i)sin(phi_i), cos(theta_i)]
    
    Objective to minimize: -min_{i<j} ||p_i - p_j||
    But min is non-smooth. Use: -log(sum(exp(-beta * d_ij))) as a soft-min approximation,
    or just use the direct min with a smooth optimizer.
    
    Actually, let's use a simpler approach: 
    - Multiple random starts on the sphere
    - Local optimization using L-BFGS-B with the objective:
      f(x) = -min_{i<j} d_ij^2  (maximize min squared distance)
    - Since min is non-smooth, use subgradient or just let the optimizer handle it
      (L-BFGS-B can work with non-smooth objectives in practice, or use Powell)
    """
    theta = x[::2]
    phi = x[1::2]
    
    # Convert to Cartesian
    pts = np.zeros((N, 3))
    pts[:, 0] = np.sin(theta) * np.cos(phi)
    pts[:, 1] = np.sin(theta) * np.sin(phi)
    pts[:, 2] = np.cos(theta)
    
    sq_dists = _pairwise_distances(pts)
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    
    if min_sq < 1e-15:
        return 1e10
    
    ratio_sq = max_sq / min_sq
    return ratio_sq


def _random_sphere_points(n_points, rng):
    """Generate random points on the unit sphere."""
    vecs = rng.normal(size=(n_points, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _fibonacci_sphere(n_points):
    """Generate points using Fibonacci sphere for good initial spread."""
    points = np.zeros((n_points, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n_points):
        y = 1 - (i / (n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = np.sin(theta) * radius
        points[i, 2] = y
    return points


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = np.inf
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Fibonacci sphere
    initial_configs.append(_fibonacci_sphere(N))
    
    # Random sphere points
    for _ in range(20):
        initial_configs.append(_random_sphere_points(N, rng))
    
    # Optimize each configuration
    for init_pts in initial_configs:
        elapsed = time.time() - _START_TIME
        if elapsed > TIME_LIMIT - 5:
            break
        
        # Convert to spherical coordinates for optimization
        theta = np.arccos(np.clip(init_pts[:, 2], -1, 1))
        phi = np.arctan2(init_pts[:, 1], init_pts[:, 0])
        # Handle phi wrapping
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        
        x0 = np.zeros(2 * N)
        x0[::2] = theta
        x0[1::2] = phi
        
        # Use Powell's method (derivative-free, good for non-smooth objectives)
        try:
            result = minimize(
                _objective, x0, method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            
            # Convert back to Cartesian
            theta_opt = result.x[::2]
            phi_opt = result.x[1::2]
            pts_opt = np.zeros((N, 3))
            pts_opt[:, 0] = np.sin(theta_opt) * np.cos(phi_opt)
            pts_opt[:, 1] = np.sin(theta_opt) * np.sin(phi_opt)
            pts_opt[:, 2] = np.cos(theta_opt)
            
            score = _score(pts_opt)
            if score < best_score:
                best_score = score
                best_pts = pts_opt.copy()
                
        except Exception:
            continue
    
    # If no valid result, fall back to Fibonacci sphere
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
