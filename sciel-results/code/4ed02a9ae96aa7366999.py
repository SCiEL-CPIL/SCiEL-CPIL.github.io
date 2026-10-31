"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _ratio(points):
    """Compute the score: min triangle area / convex hull area."""
    ma = _min_triangle_area(points)
    ha = _convex_hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _points_from_angles(angles, rx=1.0, ry=1.0):
    """Generate points on an ellipse from angles."""
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    return np.column_stack([rx * cos_a, ry * sin_a])

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Approach: Start from regular polygon, optimize angular perturbations + elliptical shape.
    """
    best_score = 0.0
    best_points = None
    
    start_time = time.time()
    timeout = 110.0
    
    # Base angles for regular 13-gon
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    def _objective_perturbed(params):
        """
        Objective: perturb regular polygon angles and optimize ellipse shape.
        params: [delta_1, ..., delta_12, shape] (13 params total, delta_13 fixed to maintain sum)
        We use 12 free deltas + 1 shape = 13 params
        """
        deltas = params[:N-1]
        shape = params[N-1]
        
        # Compute delta_13 to keep sum of all deltas = 0 (maintain centering)
        delta_13 = -np.sum(deltas)
        all_deltas = np.append(deltas, delta_13)
        
        # Perturbed angles
        angles = base_angles + all_deltas
        # Ensure angles are in [0, 2*pi)
        angles = angles % (2*np.pi)
        angles = np.sort(angles)
        
        points = _points_from_angles(angles, rx=1.0, ry=shape)
        return -_ratio(points)
    
    # Bounds for deltas: small perturbations around regular polygon
    delta_bounds = [(-0.3, 0.3)] * (N-1)
    shape_bounds = [(0.5, 2.0)]
    bounds = delta_bounds + shape_bounds
    
    # Initial: zero perturbations, circular
    x0 = np.zeros(N)
    
    # Strategy 1: Differential evolution on perturbation space (13D)
    try:
        if time.time() - start_time < timeout - 40:
            de_res = differential_evolution(
                _objective_perturbed,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-7,
                mutation=(0.4, 1.2),
                recombination=0.8,
                polish=True,
                init='sobol'
            )
            score_de = -de_res.fun
            if score_de > best_score:
                best_score = score_de
                deltas_opt = de_res.x[:N-1]
                shape_opt = de_res.x[N-1]
                delta_13 = -np.sum(deltas_opt)
                all_deltas = np.append(deltas_opt, delta_13)
                angles_opt = base_angles + all_deltas
                angles_opt = np.sort(angles_opt % (2*np.pi))
                best_points = _points_from_angles(angles_opt, rx=1.0, ry=shape_opt)
    except Exception:
        pass
    
    # Strategy 2: Local refinement from multiple starting points
    np.random.seed(42)
    
    # Starting configurations
    configs = []
    
    # Regular polygon, circular
    configs.append((np.zeros(N-1), 1.0))
    
    # Small random perturbations
    for _ in range(8):
        deltas_r = np.random.uniform(-0.15, 0.15, N-1)
        shape_r = np.random.uniform(0.7, 1.3)
        configs.append((deltas_r, shape_r))
    
    # Different shape ratios with regular angles
    for shape in [0.6, 0.8, 1.2, 1.5]:
        configs.append((np.zeros(N-1), shape))
    
    for deltas_init, shape_init in configs:
        if time.time() - start_time > timeout - 10:
            break
        
        x0 = np.append(deltas_init, shape_init)
        
        try:
            res = minimize(
                _objective_perturbed,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'maxiter': 5000,
                    'maxfev': 5000
                }
            )
            
            if time.time() - start_time > timeout - 5:
                break
                
            score = -res.fun
            if score > best_score:
                best_score = score
                deltas_opt = res.x[:N-1]
                shape_opt = res.x[N-1]
                delta_13 = -np.sum(deltas_opt)
                all_deltas = np.append(deltas_opt, delta_13)
                angles_opt = base_angles + all_deltas
                angles_opt = np.sort(angles_opt % (2*np.pi))
                best_points = _points_from_angles(angles_opt, rx=1.0, ry=shape_opt)
        except Exception:
            continue
    
    # Strategy 3: Direct coordinate optimization (free points in unit disk)
    if time.time() - start_time < timeout - 20:
        def _direct_objective(params):
            """Optimize free coordinates directly."""
            pts = params.reshape(-1, 2)
            return -_ratio(pts)
        
        # Start from regular polygon in unit circle
        init_pts = _points_from_angles(base_angles, rx=1.0, ry=1.0)
        x0_direct = init_pts.flatten()
        
        try:
            res_direct = minimize(
                _direct_objective,
                x0_direct,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'maxiter': 3000,
                    'maxfev': 3000
                }
            )
            score_direct = -res_direct.fun
            if score_direct > best_score:
                best_score = score_direct
                best_points = res_direct.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Fallback: regular polygon
    if best_points is None:
        best_points = _points_from_angles(base_angles)
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
