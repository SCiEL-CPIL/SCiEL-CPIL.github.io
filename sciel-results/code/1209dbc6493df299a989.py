import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_triangle_areas(points):
    """
    Compute areas of all C(n,3) triangles formed by the points.
    Returns a numpy array of shape (286,) with the areas.
    """
    n = len(points)
    # Generate all combinations of 3 indices
    indices = list(itertools.combinations(range(n), 3))
    num_triangles = len(indices)
    
    # Pre-extract coordinates for vectorization
    # points is (13, 2)
    # We need to form a matrix of shape (286, 3, 2)
    tri_points = np.array([points[i] for i in indices]).reshape(num_triangles, 3, 2)
    
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    # det(v, w) = v_x*w_y - v_y*w_x
    p1 = tri_points[:, 0, :]
    p2 = tri_points[:, 1, :]
    p3 = tri_points[:, 2, :]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas, indices

def _compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return float('inf')

def _score(points):
    """Compute A_min / A_hull."""
    if len(points) != 13:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    areas, _ = _compute_triangle_areas(points)
    a_min = np.min(areas)
    hull_area = _compute_hull_area(points)
    
    if hull_area <= 0 or np.isinf(hull_area):
        return 0.0
    if a_min <= 0:
        return 0.0
        
    return a_min / hull_area

def _subgradient_ascent(points, time_limit, rng):
    """
    Optimize the minimum triangle area using subgradient ascent.
    We maximize A_min while normalizing the hull area to 1.
    """
    current_points = points.copy()
    best_points = points.copy()
    best_score = _score(current_points)
    
    # Normalize initial points to have hull area 1
    hull_area = _compute_hull_area(current_points)
    if hull_area > 0:
        scale = np.sqrt(1.0 / hull_area)
        current_points = current_points * scale
        best_points = current_points.copy()
        best_score = _score(current_points)
    
    T_start = time.time()
    step_size = 0.05
    decay = 0.999
    active_eps = 1e-6
    
    while time.time() - T_start < time_limit:
        # 1. Compute areas and find min
        areas, indices = _compute_triangle_areas(current_points)
        a_min = np.min(areas)
        
        # 2. Identify active triangles (those close to min)
        # Using a small epsilon to handle plateaus
        active_mask = (areas <= a_min + active_eps)
        active_indices = indices[active_mask]
        
        # 3. Compute subgradient
        # The gradient of the area of triangle (i, j, k) with respect to point p_m
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
        # Let D = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # If D > 0, grad wrt p_i = -0.5 * (D_y, D_x) ? 
        # Let's derive carefully.
        # Area = 0.5 * | det(p_j - p_i, p_k - p_i) |
        # Let u = p_j - p_i, v = p_k - p_i. Area = 0.5 | u_x v_y - u_y v_x |.
        # Let f = u_x v_y - u_y v_x.
        # df/dp_i = -df/du - df/dv
        # df/du = (v_y, -v_x)
        # df/dv = (-u_x, u_y)
        # So df/dp_i = -(v_y, -v_x) - (-u_x, u_y) = (u_x - v_y, v_x - u_y)
        # df/dp_j = df/du = (v_y, -v_x)
        # df/dp_k = df/dv = (-u_x, u_y)
        
        # We want to maximize the min area. So we move in the direction of the gradient of the min area.
        # If we use the active set, we sum the gradients of the active triangles.
        
        grad = np.zeros_like(current_points)
        
        for i, j, k in active_indices:
            pi = current_points[i]
            pj = current_points[j]
            pk = current_points[k]
            
            u = pj - pi
            v = pk - pi
            
            # Check orientation to handle absolute value
            # f = u_x v_y - u_y v_x
            f = u[0]*v[1] - u[1]*v[0]
            
            # Gradient of |f| is sign(f) * gradient(f)
            # If f == 0, gradient is undefined (0 or any direction), but usually we just use 0 or skip.
            if np.abs(f) < 1e-10:
                continue
                
            sign_f = 1.0 if f > 0 else -1.0
            
            # Gradients for the determinant part (before 0.5 factor)
            # d(det)/dp_i = (u_x - v_y, v_x - u_y)
            # d(det)/dp_j = (v_y, -v_x)
            # d(det)/dp_k = (-u_x, u_y)
            
            g_i = np.array([u[0] - v[1], v[0] - u[1]])
            g_j = np.array([v[1], -v[0]])
            g_k = np.array([-u[0], u[1]])
            
            # Multiply by 0.5 and sign
            factor = 0.5 * sign_f
            
            grad[i] += factor * g_i
            grad[j] += factor * g_j
            grad[k] += factor * g_k
            
        # Normalize gradient to avoid huge steps
        norm_grad = np.linalg.norm(grad)
        if norm_grad > 1e-10:
            grad = grad / norm_grad
        
        # 4. Update points
        # We want to increase A_min. So we move in direction of grad.
        # However, we are also normalizing hull area.
        # A simple approach: Update, then normalize.
        new_points = current_points + step_size * grad
        
        # Normalize to keep hull area approx 1
        hull_area = _compute_hull_area(new_points)
        if hull_area > 0 and np.isfinite(hull_area):
            scale = np.sqrt(1.0 / hull_area)
            new_points = new_points * scale
            
        # Check score
        new_score = _score(new_points)
        
        # Accept if better, or with some probability (SA-like) to escape local minima?
        # Subgradient method is deterministic. Let's add a small random noise.
        if new_score > best_score:
            best_points = new_points.copy()
            best_score = new_score
            current_points = new_points.copy()
        else:
            # If we get stuck, add noise
            if rng.random() < 0.05:
                noise = rng.normal(0, step_size * 0.5, size=current_points.shape)
                new_points = current_points + noise
                hull_area = _compute_hull_area(new_points)
                if hull_area > 0:
                    scale = np.sqrt(1.0 / hull_area)
                    new_points = new_points * scale
                current_points = new_points
        
        # Decay step size
        step_size *= decay
        
        # Safety check
        if not np.all(np.isfinite(current_points)):
            break

    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    deadline = time.time() + 620  # Leave 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    inits = []
    
    # 1. Halton
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points_halton = np.zeros((13, 2))
    for i in range(13):
        points_halton[i, 0] = halton(i + 1, 2)
        points_halton[i, 1] = halton(i + 1, 3)
    points_halton = (points_halton - 0.5) * 2.0
    inits.append(points_halton)
    
    # 2. Regular Polygon + Center
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    points_poly = np.column_stack((np.cos(theta), np.sin(theta)))
    inits.append(points_poly)
    
    # 3. Grid-like
    points_grid = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]
    ])
    points_grid = (points_grid - 1.0)
    inits.append(points_grid)
    
    # 4. Random
    rng_init = np.random.default_rng(42)
    inits.append(rng_init.random((13, 2)) - 0.5)
    
    # Run optimization on each init
    for i, init_points in enumerate(inits):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        # Allocate time based on remaining inits
        time_per_run = time_left / (len(inits) - i)
        
        rng = np.random.default_rng(1000 + i)
        points, score = _subgradient_ascent(init_points, time_per_run, rng)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
        # If we have time, do a local polish on the best found so far
        if time.time() < deadline - 30:
             # Restart from best found with smaller step
             rng2 = np.random.default_rng(2000 + i)
             # Perturb slightly
             perturbed = best_points + rng2.normal(0, 0.01, size=(13, 2))
             time_left_polish = deadline - time.time() - 5
             if time_left_polish > 1:
                 points_polish, score_polish = _subgradient_ascent(perturbed, time_left_polish, rng2)
                 if score_polish > best_score:
                     best_score = score_polish
                     best_points = points_polish.copy()

    if best_points is None:
        # Fallback
        best_points = inits[0]
        
    return best_points
