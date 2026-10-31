import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

def _triangle_areas_from_angles(angles):
    """
    Compute areas of all triangles formed by points on the unit circle
    given their angles.
    """
    # Points on unit circle
    # x = cos(theta), y = sin(theta)
    # Area of triangle with vertices i, j, k on unit circle:
    # 0.5 * | (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)) |
    # This simplifies to 0.5 * | sin(theta_j - theta_i) + sin(theta_k - theta_j) + sin(theta_i - theta_k) |
    # Wait, let's use the determinant formula directly for clarity and safety with arbitrary angles.
    
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _objective_angles(angles, target_min_area=0.0):
    """
    Objective function for minimization.
    We want to maximize the minimum triangle area.
    We use a barrier approach: 
    f(angles) = sum( max(0, target - area_i)^2 ) + penalty_for_ordering
    
    However, a simpler robust approach for 'maximin' is to minimize the negative of a smooth minimum.
    But the prompt suggests a log-barrier constrained optimization where the barrier penalizes any triangle area falling below a target.
    
    Let's refine: We will try to increase a 'target' threshold.
    Actually, for a single minimization run, we can define the objective as:
    min sum( (target - area_i)^2 * I(area_i < target) )
    
    This is non-smooth. Let's use a soft-min or a barrier.
    A common technique for max-min is to minimize:
    -log(min_area) is not smooth.
    
    Let's use the strategy: Minimize a function that grows rapidly if any area is below a threshold.
    If we fix a target T, we minimize sum( max(0, T - area_i)^2 ).
    If this sum is 0, then all areas >= T.
    We can iteratively increase T.
    
    Alternatively, we can just minimize the negative of the "smooth minimum" of the areas.
    smooth_min(areas) = -1/beta * log(sum(exp(-beta * area_i)))
    Minimizing -smooth_min is equivalent to maximizing smooth_min.
    As beta -> inf, this approaches max-min.
    
    Let's use the smooth-min approach as it is differentiable and handles the "min" naturally without explicit thresholding loops.
    """
    areas = _triangle_areas_from_angles(angles)
    
    # Ensure areas are positive (they should be for distinct points on circle)
    # To avoid log(0), add small epsilon
    areas_safe = np.maximum(areas, 1e-12)
    
    # Smooth minimum parameter
    beta = 50.0
    
    # Smooth min of areas
    # log-sum-exp trick for stability
    max_area = np.max(areas_safe)
    log_sum = np.log(np.sum(np.exp(-beta * (areas_safe - max_area)))) + (-beta * max_area)
    smooth_min = -1.0 / beta * log_sum
    
    # We want to maximize smooth_min, so minimize -smooth_min
    return -smooth_min

def _optimize_angles_from_start(start_angles, time_limit=5.0):
    """
    Run L-BFGS-B minimization starting from start_angles.
    """
    start_time = time.time()
    
    # Bounds: angles in [0, 2*pi). 
    # To ensure uniqueness and convexity, we can sort them.
    # But L-BFGS-B doesn't handle sorting constraints well.
    # Instead, we can parameterize the *gaps* between angles.
    # Let g_i be the gap between point i and i+1. Sum(g_i) = 2*pi, g_i > 0.
    # This automatically ensures the points are on the circle in order and form a convex polygon.
    # This reduces the dimension to N-1 (since sum is fixed) or N with a constraint.
    # Using gaps is a structural change that enforces convexity and ordering!
    
    # Let's switch to gap representation.
    # x = [g_0, ..., g_{N-2}]. g_{N-1} = 2*pi - sum(x).
    # We need all g_i > 0.
    # We can use a softplus transform to ensure positivity: g_i = softplus(y_i).
    # But then sum constraint is hard.
    
    # Simpler: Use angles but penalize if they are not sorted? No, that's messy.
    # Let's stick to angles but initialize them sorted and hope L-BFGS keeps them close.
    # Or better: Use the gap representation properly.
    
    # Let's implement the gap representation optimization.
    # Variables: y (size N-1). 
    # g_i = exp(y_i) for i=0..N-2.
    # g_{N-1} = 2*pi - sum(exp(y_i)).
    # Constraint: g_{N-1} > 0 => sum(exp(y_i)) < 2*pi.
    
    # This is getting complex. Let's stick to angles [0, 2pi) but use a barrier for non-convexity?
    # Actually, if points are on the circle, they are ALWAYS in convex position (unless collinear, which is impossible on circle).
    # The only issue is the order. If we sort them, the area calculation is the same.
    # The score depends on the set of angles, not the order.
    # So we can just optimize angles in [0, 2pi) and the result will be a set of points.
    # The "convexity" is inherent to being on the circle.
    
    # Let's use L-BFGS-B on angles directly.
    bounds = [(0, 2*np.pi)] * N
    
    def obj_wrapper(y):
        if time.time() > start_time + time_limit:
            return 1e10
        return _objective_angles(y)
        
    try:
        res = minimize(
            obj_wrapper, 
            start_angles, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 5000}
        )
        return res.x
    except Exception:
        return start_angles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Strategy:
    # 1. Generate several initial configurations of angles.
    #    - Regular polygon
    #    - Random angles
    #    - Angles perturbed from regular polygon
    # 2. Optimize each using L-BFGS-B with smooth-min objective.
    # 3. Evaluate the exact score (min area / hull area) of the best candidates.
    # 4. If time permits, run a second phase with a higher beta (sharper min) or local polish.
    
    initial_configs = []
    
    # Config 1: Regular polygon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_configs.append(angles_reg)
    
    # Config 2: Random
    np.random.seed(42)
    angles_rand = np.sort(np.random.uniform(0, 2*np.pi, N))
    initial_configs.append(angles_rand)
    
    # Config 3: Perturbed regular
    angles_pert = angles_reg + np.random.normal(0, 0.1, N)
    angles_pert = np.sort(angles_pert)
    initial_configs.append(angles_pert)
    
    # Config 4: Another random
    np.random.seed(123)
    angles_rand2 = np.sort(np.random.uniform(0, 2*np.pi, N))
    initial_configs.append(angles_rand2)
    
    # Config 5: "Golden angle" inspired?
    # Just add a few more randoms
    for seed in [7, 8, 9]:
        np.random.seed(seed)
        angles_r = np.sort(np.random.uniform(0, 2*np.pi, N))
        initial_configs.append(angles_r)
        
    # Optimize each config
    for i, init_angles in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break
            
        # Check time per optimization
        # L-BFGS-B is fast. 13 dims. Should be < 1s each.
        opt_angles = _optimize_angles_from_start(init_angles, time_limit=5.0)
        
        # Convert angles to points
        points = np.column_stack((np.cos(opt_angles), np.sin(opt_angles)))
        
        # Calculate exact score
        score = calculate_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    # If we have time, try to refine the best points found so far
    # We can convert best_points back to angles?
    # Or just do a local coordinate-wise polish on the Cartesian points?
    # The current program had a local polish. Let's include a simple one.
    
    if best_points is not None and time.time() < deadline - 10.0:
        # Local polish on Cartesian coordinates
        current_pts = best_points.copy()
        current_score = best_score
        
        step = 0.01
        no_improve = 0
        max_no_improve = 50
        polish_start = time.time()
        
        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            improved = False
            for i in range(N):
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                    trial_pts = current_pts.copy()
                    trial_pts[i, 0] += dx
                    trial_pts[i, 1] += dy
                    trial_score = calculate_score(trial_pts)
                    
                    if trial_score > current_score:
                        current_pts = trial_pts
                        current_score = trial_score
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                step *= 0.9
                if step < 1e-6:
                    break
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)
        
    return best_points
