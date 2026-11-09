"""
Heilbronn convex regions n=14: Parametric Perturbation Strategy.
Represents the 14 points as a regular 14-gon with angular and radial perturbations.
This ensures the points remain in a "near-regular" convex configuration, reducing
the search space dimensionality and avoiding degenerate clusters.
Uses Differential Evolution on the 28 perturbation parameters.
"""
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]  # 364

def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    return np.min(_all_triangle_areas(points))

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert perturbation parameters to point coordinates.
    params: shape (2*N,) = [d_angle_0, d_radius_0, ..., d_angle_13, d_radius_13]
    
    Base: Regular 14-gon.
    Perturbations:
    - d_angle_i: angular deviation from the regular position.
    - d_radius_i: radial deviation from the unit circle.
    
    Constraints:
    - Angles must remain sorted to ensure convexity (no crossing).
      We enforce this by clamping the angular deviations so that the order is preserved.
      Specifically, the change in angle between consecutive points must be positive.
      Base angle step = 2*pi/N.
      Let theta_i = 2*pi*i/N + d_angle_i.
      We require theta_{i+1} > theta_i.
      This implies d_angle_{i+1} - d_angle_i < 2*pi/N.
      To make this robust and differentiable (or at least continuous for DE),
      we can simply clamp the resulting angles to be strictly increasing?
      No, clamping breaks the smoothness.
      
      Alternative: Define the angular deviations relative to the previous point.
      Let delta_theta_i be the actual angle step.
      Base step = 2*pi/N.
      Let delta_theta_i = (2*pi/N) * (1 + d_theta_i).
      If we constrain d_theta_i to be small (e.g., in [-0.5, 0.5]), then delta_theta_i > 0.
      This guarantees convexity in terms of angles.
      
      Similarly for radius:
      r_i = 1 + d_r_i.
      To ensure the polygon is star-shaped and convex, r_i should be positive.
      We can constrain d_r_i to be in [-0.5, 0.5] so r_i in [0.5, 1.5].
      
      This parameterization guarantees:
      1. Points are in angular order (since all steps are positive).
      2. Points are at positive radii.
      3. The polygon is strictly convex if the curvature is positive.
         For a polygon defined by polar coordinates (r_i, theta_i) with sorted theta_i,
         convexity is not automatically guaranteed if r_i varies too wildly.
         However, for small perturbations around a regular polygon, it remains convex.
         We can add a penalty if the polygon is not convex (cross product check).
    """
    d_theta = params[:N]
    d_r = params[N:]
    
    # Base angles
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Perturbed angles: use multiplicative perturbation on the step
    # step_i = (2*pi/N) * (1 + d_theta_i)
    # But the sum of steps must be 2*pi.
    # If we just add d_theta_i to base_angles, the sum of angles is not necessarily 2*pi,
    # but that's fine for a set of points. The convex hull will be determined by the points.
    # However, to maintain the "regular polygon" structure, it's better to perturb the steps
    # such that they sum to 2*pi.
    
    # Let's use the step perturbation method:
    # steps = (2*pi/N) * (1 + d_theta)
    # We need sum(steps) = 2*pi.
    # This implies sum(d_theta) = 0.
    # DE doesn't easily handle sum constraints.
    
    # Simpler: Just add d_theta to the base angles.
    # theta_i = base_angles[i] + d_theta[i]
    # To ensure order, we can sort them? No, that changes the mapping.
    # If d_theta is small, the order is preserved.
    # We will bound d_theta to be small, e.g., [-pi/4, pi/4].
    # The base step is ~0.45 rad. pi/4 is ~0.78 rad.
    # If d_theta[i] = pi/4 and d_theta[i+1] = -pi/4, the difference is -pi/2, which is < 0.
    # So order might be violated.
    
    # Robust approach:
    # Define angles as cumulative sum of positive steps.
    # steps_i = exp(d_log_step_i)
    # Normalize steps to sum to 2*pi.
    # theta_i = cumsum(steps)[:-1]
    
    d_log_step = params[:N]
    raw_steps = np.exp(d_log_step)
    steps = raw_steps / np.sum(raw_steps) * 2 * np.pi
    
    angles = np.cumsum(steps)[:-1]
    
    # Radii
    # r_i = 1 + d_r_i
    # Constrain d_r_i to be small to ensure convexity.
    # Let's use a soft constraint or just bound d_r_i in DE.
    r = 1.0 + d_r
    
    # Ensure r > 0
    r = np.maximum(r, 0.1)
    
    points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    
    return points

def _negative_score_params(params):
    """Objective function for DE: minimize negative score."""
    points = _params_to_points(params)
    
    # Check convexity
    # Compute cross products of consecutive edges in the sorted order (which is already sorted by construction)
    # For a polygon defined by points in CCW order, convexity means all cross products are positive.
    # Note: The points from _params_to_points are in CCW order by construction (cumsum of positive steps).
    
    # Vectorized cross product check
    # p_i, p_{i+1}, p_{i+2}
    p_i = points
    p_j = np.roll(points, -1, axis=0)
    p_k = np.roll(points, -2, axis=0)
    
    v1 = p_j - p_i
    v2 = p_k - p_j
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # If any cross product is <= 0, the polygon is not strictly convex.
    # We add a penalty.
    penalty = np.sum(np.maximum(0, -cross) ** 2)
    
    score = _score(points)
    
    return -score + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial guess: Zero perturbations (Regular polygon)
    init_params = np.zeros(2 * N)
    
    best_score = _score(_params_to_points(init_params))
    best_params = init_params.copy()
    
    # DE Bounds
    # d_log_step: exp(d_log_step) should be around 1. So d_log_step around 0.
    # Range [-1, 1] -> steps in [0.37, 2.7]. Ratio ~7. This allows significant shape variation.
    # d_r: Range [-0.5, 0.5] -> r in [0.5, 1.5].
    
    bounds = [(-1.0, 1.0)] * N + [(-0.5, 0.5)] * N
    
    try:
        de_result = differential_evolution(
            _negative_score_params,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            mutation=(0.3, 1.2),
            recombination=0.8,
            polish=True,
            init='sobol'
        )
        
        cur_score = -de_result.fun
        # Note: de_result.fun includes penalty. We need to check the actual score.
        cur_points = _params_to_points(de_result.x)
        actual_score = _score(cur_points)
        
        if actual_score > best_score:
            best_score = actual_score
            best_params = de_result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return _params_to_points(best_params)
    
    # Polish with L-BFGS-B
    try:
        res = minimize(
            _negative_score_params,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        cur_points = _params_to_points(res.x)
        actual_score = _score(cur_points)
        
        if actual_score > best_score:
            best_score = actual_score
            best_params = res.x.copy()
    except Exception:
        pass
    
    return _params_to_points(best_params)

# EVOLVE-BLOCK-END
