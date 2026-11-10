import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # Safety margin for 630s limit

def _compute_score(points):
    """
    Computes the exact score: A_min / A_hull.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Pre-compute indices for efficiency if called often, but here we just compute directly
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    
    A_min = np.min(areas)
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    return A_min / A_hull

def _param_to_points(params):
    """
    Maps a low-dimensional parameter vector to 13 points.
    
    Structure:
    - params[0]: center_x
    - params[1]: center_y
    - params[2]: r1 (radius of ring 1)
    - params[3]: theta1 (angle offset ring 1)
    - params[4]: r2 (radius of ring 2)
    - params[5]: theta2 (angle offset ring 2)
    
    Ring 1 has 6 points, Ring 2 has 6 points.
    Total 13 points: 1 center + 6 + 6.
    
    To allow for breaking symmetry and finding the true optimum (which might not be perfectly concentric),
    we add small radial and angular perturbations for each point in the rings.
    However, to keep the dimensionality low for L-BFGS-B speed, we might stick to just the global params first.
    If that's too restrictive, we can add per-point radial offsets.
    
    Let's try:
    - Global: cx, cy, r1, th1, r2, th2 (6 params)
    - Perturbations: 12 radial offsets (6 for ring 1, 6 for ring 2) and 12 angular offsets?
    That's 6 + 12 + 12 = 30 params. Slightly higher than 26, but structured.
    
    Actually, the optimal config for N=13 is often very close to a regular polygon or concentric rings.
    Let's try a hybrid:
    - 1 center point (fixed at 0,0 initially, but let's optimize cx, cy)
    - 6 points on a circle of radius R1, with individual radial perturbations dr1_i and angular perturbations dth1_i
    - 6 points on a circle of radius R2, with individual radial perturbations dr2_i and angular perturbations dth2_i
    
    Total params:
    cx, cy (2)
    R1, base_th1 (2)
    R2, base_th2 (2)
    dr1 (6), dth1 (6)
    dr2 (6), dth2 (6)
    Total = 2 + 2 + 2 + 12 + 12 = 30 parameters.
    
    This is comparable to 26 but the structure guides the search.
    """
    cx, cy = params[0], params[1]
    R1, th1_base = params[2], params[3]
    R2, th2_base = params[4], params[5]
    
    # Perturbations
    dr1 = params[6:12]
    dth1 = params[12:18]
    dr2 = params[18:24]
    dth2 = params[24:30]
    
    points = np.zeros((13, 2))
    
    # Center
    points[0] = [cx, cy]
    
    # Ring 1
    for i in range(6):
        r = R1 + dr1[i]
        # Ensure radius is positive to avoid weird flips, though optimizer should handle it
        if r < 0: r = 0.0
        th = th1_base + i * (np.pi / 3) + dth1[i]
        points[i+1] = [cx + r * np.cos(th), cy + r * np.sin(th)]
        
    # Ring 2
    for i in range(6):
        r = R2 + dr2[i]
        if r < 0: r = 0.0
        th = th2_base + i * (np.pi / 3) + dth2[i]
        points[i+7] = [cx + r * np.cos(th), cy + r * np.sin(th)]
        
    return points

def _objective(params):
    """
    Objective function for minimization.
    We want to MAXIMIZE A_min / A_hull.
    So we MINIMIZE - (A_min / A_hull).
    
    To make it differentiable, we use a soft-min approximation for A_min.
    A_min = min(areas).
    Soft-min: -T * log(sum(exp(-areas/T))).
    
    We also need to handle the hull area. The hull area is piecewise linear.
    For L-BFGS-B, we can use the exact hull area if we don't need its gradient, 
    but L-BFGS-B uses gradients. 
    If we provide a gradient for the soft-min part and approximate the hull gradient, 
    or just use numerical gradients?
    
    L-BFGS-B can use numerical gradients if 'jac' is not provided, but it's slower.
    However, 30 params is small. Numerical gradients might be fine.
    But let's try to provide an analytic gradient for the soft-min part.
    
    Let S = sum(exp(-areas/T)).
    L = -T log(S).
    dL/dx = -T * (1/S) * dS/dx = -T * (1/S) * sum(exp(-area_i/T) * (-1/T) * d(area_i)/dx)
    dL/dx = (1/S) * sum(exp(-area_i/T) * d(area_i)/dx)
    
    This is a weighted average of the gradients of the areas, weighted by exp(-area_i/T).
    As T -> 0, the weight concentrates on the minimum area.
    
    The gradient of the ratio R = L / A_hull is:
    dR/dx = (dL/dx * A_hull - L * dA_hull/dx) / A_hull^2
    
    Computing dA_hull/dx analytically is complex.
    Option 1: Use numerical gradient for the whole objective.
    Option 2: Use a different optimizer that doesn't need gradients (like Nelder-Mead) on the 30D space.
    Option 3: Use L-BFGS-B with numerical gradients.
    
    Given N=13, 30 params, numerical gradients are 60 evaluations per step.
    Each evaluation is fast (vectorized area calc + ConvexHull).
    ConvexHull for 13 points is very fast.
    Let's use L-BFGS-B with numerical gradients. It is robust and handles the structure well.
    
    Wait, the soft-min makes it smooth, but the hull area is not.
    The ratio will have kinks where the hull changes.
    Numerical gradients handle this (subgradients).
    
    Let's define the objective to return the value.
    """
    points = _param_to_points(params)
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    n = 13
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    cross_z = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross_z)
    
    # Soft min
    T = 0.05
    # To avoid overflow in exp, shift by max
    max_area = np.max(areas)
    # We want min(areas).
    # exp(-area/T). If area is large, exp is small.
    # To be stable:
    # sum(exp(-(area - min_area)/T)) is better?
    # Standard soft min: m = -T log(sum(exp(-x_i/T)))
    # If x_i are large, -x_i/T is large negative. exp is 0.
    # If x_i are small, exp is ~1.
    # Let's just use the formula directly, but clip areas to be >= 0.
    
    # Actually, for numerical stability in log-sum-exp:
    # log(sum(exp(-x/T))) = max(-x/T) + log(sum(exp(-x/T - max(-x/T))))
    # Let M = max(-x/T) = -min(x)/T
    # Then sum(exp(-x/T)) = exp(M) * sum(exp(-x/T - M))
    # log(sum(exp(-x/T))) = M + log(sum(exp(-x/T - M)))
    
    neg_areas_over_T = -areas / T
    M = np.max(neg_areas_over_T)
    if M == -np.inf:
        return 1e10
    log_sum = M + np.log(np.sum(np.exp(neg_areas_over_T - M)))
    soft_min_area = -T * log_sum
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
        
    if A_hull <= 1e-9:
        return 1e10
        
    ratio = soft_min_area / A_hull
    return -ratio

def heilbronn_convex13() -> np.ndarray:
    best_points = None
    best_score = -np.inf
    
    # Initial guesses for the 30 params
    # cx, cy, R1, th1, R2, th2, dr1(6), dth1(6), dr2(6), dth2(6)
    
    inits = []
    
    # Guess 1: Concentric rings, centered at origin
    # R1 ~ 0.4, R2 ~ 0.8
    p0 = np.zeros(30)
    p0[0] = 0.0; p0[1] = 0.0
    p0[2] = 0.4; p0[3] = 0.0
    p0[4] = 0.8; p0[5] = np.pi/6  # Offset ring 2 by 30 deg
    # Perturbations start at 0
    
    inits.append(p0)
    
    # Guess 2: Different radii
    p1 = np.zeros(30)
    p1[0] = 0.0; p1[1] = 0.0
    p1[2] = 0.3; p1[3] = 0.0
    p1[4] = 0.7; p1[5] = 0.0
    inits.append(p1)
    
    # Guess 3: Perturbed regular 13-gon mapped to this structure?
    # Hard to map directly. Let's just use random perturbations on top of Guess 1.
    rng = np.random.default_rng(42)
    for _ in range(5):
        p_rand = p0.copy()
        # Small random perturbations
        p_rand[6:] += rng.normal(0, 0.05, 24)
        p_rand[2] += rng.normal(0, 0.05)
        p_rand[4] += rng.normal(0, 0.05)
        inits.append(p_rand)
        
    # Also try a "flat" initialization where R1 is small
    p_flat = np.zeros(30)
    p_flat[2] = 0.1
    p_flat[4] = 0.9
    inits.append(p_flat)
    
    for init_params in inits:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < 1e9:
                p_opt = res.x
                points_opt = _param_to_points(p_opt)
                
                # Check validity
                if not np.all(np.isfinite(points_opt)):
                    continue
                    
                score = _compute_score(points_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
        except Exception:
            pass
        
        # Also evaluate the initial point
        points_init = _param_to_points(init_params)
        if np.all(np.isfinite(points_init)):
            score_init = _compute_score(points_init)
            if score_init > best_score:
                best_score = score_init
                best_points = points_init.copy()

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
