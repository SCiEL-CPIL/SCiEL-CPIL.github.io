import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) where w=1-u-v to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * H
    return np.column_stack((x, y))

def _cart_to_bary(pts):
    """Convert Cartesian coordinates to barycentric (u, v)."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = x - y / (2.0 * H)
    u = 1.0 - x - y / (2.0 * H)
    return np.column_stack((u, v))

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    bary = _cart_to_bary(points)
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return _bary_to_cart(np.column_stack((u, v)))

def _sobol_points_in_triangle(n):
    """Generate n points in the triangle using Sobol sequence."""
    # Sobol sequence in 2D
    sampler = qmc.Sobol(d=2, scramble=True)
    # Generate more points than needed to ensure we have enough inside the triangle
    # Probability of a random point being in the triangle is 1/2 (if using square [-0.5, 0.5]x[0, 0.5]?)
    # Actually, let's generate in [0,1]x[0,1] and filter.
    # The triangle is (0,0), (1,0), (0.5, H).
    # Area is H/2 * 1 = H/2? No, base 1, height H. Area = H/2.
    # Wait, standard unit area triangle?
    # Task says "unit-area triangle".
    # The triangle (0,0), (1,0), (0.5, sqrt(3)/2) has area sqrt(3)/4.
    # If the task implies the evaluator divides by the area of the triangle used,
    # then we are maximizing the ratio.
    # The prompt says: "The notebook uses an equilateral triangle and divides by its area."
    # And "construct() -> array of shape (n, 2): n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
    # So we work in this specific triangle.
    
    # Generate points in [0,1]x[0,1]
    # We need points inside (0,0), (1,0), (0.5, H).
    # Condition: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # Let's just generate random points in the bounding box and filter, or use rejection sampling on Sobol.
    
    pts = []
    # Generate a batch
    samples = sampler.random(n * 10)
    for s in samples:
        x, y = s
        # Map [0,1]x[0,1] to the triangle?
        # It's easier to generate in the bounding box [0,1]x[0,H] and check.
        y_scaled = y * H
        if y_scaled >= 0 and y_scaled <= SQRT3 * x and y_scaled <= SQRT3 * (1 - x):
            pts.append([x, y_scaled])
        if len(pts) >= n:
            break
    
    if len(pts) < n:
        # Fallback to random if Sobol fails (unlikely)
        while len(pts) < n:
            x = np.random.rand()
            y = np.random.rand() * H
            if y >= 0 and y <= SQRT3 * x and y <= SQRT3 * (1 - x):
                pts.append([x, y])
                
    return np.array(pts[:n])

def _objective_smooth(params, beta=100.0):
    """
    Smooth objective function for optimization: negative of soft min area.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points)
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - log_sum / beta
    return -soft_min

def _local_nudge_search(points, max_iters=1000, time_limit=None, start_time=None):
    """
    Fast local search that nudges points to improve the minimum area.
    Focuses on the points involved in the smallest triangles.
    """
    current_pts = points.copy()
    current_min = _min_area(current_pts)
    
    # Precompute gradients? No, just trial moves.
    # We will try moving each point in small random directions and keep if it improves.
    rng = np.random.default_rng(42)
    
    step_size = 0.05
    for it in range(max_iters):
        if time_limit and time.time() - start_time > time_limit:
            break
            
        areas = _compute_areas(current_pts)
        min_area_val = np.min(areas)
        
        # Find indices of points involved in the minimum area triangles
        min_idx = np.argmin(areas)
        p1, p2, p3 = _idx_arr[min_idx]
        
        # Try to improve by moving one of these points
        # Or move any point? Moving a point not in the min triangle might break other constraints.
        # It's safer to move points in the critical set.
        # Let's pick a random point from the critical set, or iterate through all points?
        # Iterating through all points is O(N) per step.
        
        improved = False
        # Try nudging each point in the critical triangle
        for idx in [p1, p2, p3]:
            for _ in range(10): # Multiple nudges per point
                # Random direction
                angle = rng.random() * 2 * np.pi
                dx = step_size * np.cos(angle)
                dy = step_size * np.sin(angle)
                
                new_pts = current_pts.copy()
                new_pts[idx, 0] += dx
                new_pts[idx, 1] += dy
                new_pts = _project_to_triangle(new_pts)
                
                new_min = _min_area(new_pts)
                if new_min > current_min + 1e-12:
                    current_pts = new_pts
                    current_min = new_min
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            # No improvement in critical points, try random point
            idx = rng.integers(0, N)
            for _ in range(5):
                angle = rng.random() * 2 * np.pi
                dx = step_size * np.cos(angle)
                dy = step_size * np.sin(angle)
                
                new_pts = current_pts.copy()
                new_pts[idx, 0] += dx
                new_pts[idx, 1] += dy
                new_pts = _project_to_triangle(new_pts)
                
                new_min = _min_area(new_pts)
                if new_min > current_min + 1e-12:
                    current_pts = new_pts
                    current_min = new_min
                    break
                    
            # Decay step size if stuck
            step_size *= 0.95
            if step_size < 1e-6:
                break
                
    return current_pts, current_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    def _update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # 1. Initialize with Sobol
    init_pts = _sobol_points_in_triangle(N)
    init_pts = _project_to_triangle(init_pts)
    score = _min_area(init_pts)
    _update_best(init_pts, score)
    
    # 2. Local Nudge Search
    refined_pts, refined_score = _local_nudge_search(init_pts, max_iters=5000, time_limit=20, start_time=start_time)
    _update_best(refined_pts, refined_score)
    
    # 3. Smooth Optimization Polish
    if time.time() < deadline - 30:
        x0 = best_points.flatten()
        try:
            res = minimize(
                _objective_smooth, 
                x0, 
                method='L-BFGS-B',
                args=(100.0,),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_triangle(new_pts)
            score = _min_area(new_pts)
            _update_best(new_pts, score)
        except Exception:
            pass
            
    # 4. More Local Nudge Search from polished result
    if time.time() < deadline - 10:
        refined_pts, refined_score = _local_nudge_search(best_points, max_iters=5000, time_limit=20, start_time=start_time)
        _update_best(refined_pts, refined_score)
        
    # 5. Restart with different random seeds if time permits
    restarts = 5
    for i in range(restarts):
        if time.time() > deadline - 15:
            break
        rng = np.random.default_rng(100 + i)
        # Random initialization
        pts = np.random.rand(N, 2) * np.array([1.0, H])
        # Filter/project to triangle
        # Simple rejection sampling for init
        valid_pts = []
        while len(valid_pts) < N:
            x, y = rng.random(2) * np.array([1.0, H])
            if y >= 0 and y <= SQRT3 * x and y <= SQRT3 * (1 - x):
                valid_pts.append([x, y])
        rand_pts = np.array(valid_pts)
        
        # Quick local search
        refined_pts, refined_score = _local_nudge_search(rand_pts, max_iters=2000, time_limit=10, start_time=start_time)
        _update_best(refined_pts, refined_score)
        
    if best_points is None:
        best_points = init_pts
        
    return _project_to_triangle(best_points)
