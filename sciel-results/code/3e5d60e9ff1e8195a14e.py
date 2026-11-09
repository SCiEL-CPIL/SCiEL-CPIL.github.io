import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import time
import itertools

def _compute_all_triangle_areas(points):
    """
    Compute areas of all C(13,3) = 286 triangles.
    Returns array of shape (286,) and index pairs.
    """
    n = len(points)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                 p2[0] * (p3[1] - p1[1]) + 
                                 p3[0] * (p1[1] - p2[1]))
                areas.append(area)
                indices.append((i, j, k))
    return np.array(areas), indices

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n != 13:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    areas, _ = _compute_all_triangle_areas(points)
    min_area = np.min(areas)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
    
    return min_area / hull_area

def _active_set_qp_solver(points, time_limit=5.0):
    """
    Active-set QP solver:
    1. Identify the minimum-area triangle (active constraint)
    2. Solve a QP that maximizes this triangle's area while keeping all other triangles >= current min
    3. Iterate, adding newly-active constraints to the working set
    
    We use a projected gradient method on the QP subproblem.
    """
    start_time = time.time()
    current = points.copy().astype(float)
    n = len(current)
    
    # Precompute all triangle index triples
    all_tris = list(itertools.combinations(range(n), 3))
    n_tris = len(all_tris)  # 286
    
    # Store triangle coefficients: area = 0.5 * |a*x_i + b*x_j + c*x_k + ...|
    # For a fixed triangle (i,j,k), the signed area is:
    # 0.5 * (x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j))
    # This is linear in the 2n coordinates!
    
    # We'll work with flattened coordinates: z = [x_0, y_0, x_1, y_1, ..., x_12, y_12]
    z = current.flatten().copy()
    z_dim = 2 * n  # 26
    
    # For each triangle, precompute the linear coefficient vector for the SIGNED area
    # signed_area = 0.5 * (x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j))
    # = 0.5 * [x_i*y_j - x_i*y_k + x_j*y_k - x_j*y_i + x_k*y_i - x_k*y_j]
    # This is NOT linear in z because of products x*y!
    # 
    # However, if we FIX two points and optimize the third, it IS linear.
    # Better approach: use the fact that for small perturbations, we can linearize.
    
    # Alternative: use a different formulation.
    # Let's use a sequential approach:
    # At each iteration, find the min triangle, then move its 3 vertices to increase its area
    # while checking all other triangles don't decrease below the new target.
    
    # For now, implement a robust local search with active-set identification:
    
    best_z = z.copy()
    best_min_area, _ = _compute_all_triangle_areas(current)
    best_min_area = np.min(best_min_area)
    
    step_size = 0.1
    active_set = set()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Recompute areas and find minimum
        areas, _ = _compute_all_triangle_areas(current)
        min_area = np.min(areas)
        min_idx = np.argmin(areas)
        
        # Update best
        if min_area > best_min_area:
            best_min_area = min_area
            best_z = z.copy()
        
        # Identify active set: triangles with area within 5% of minimum
        threshold = min_area * 1.05
        active_indices = np.where(areas <= threshold)[0]
        
        # For the most critical triangle, compute gradient direction
        i, j, k = all_tris[min_idx]
        p1, p2, p3 = current[i], current[j], current[k]
        
        # Signed area
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        sign = np.sign(cross) if cross != 0 else 1.0
        
        # Gradients for increasing the area of this triangle
        def rot90(v):
            return np.array([-v[1], v[0]])
        
        grad_p1 = -0.5 * sign * rot90(p3 - p2)
        grad_p2 =  0.5 * sign * rot90(p3 - p1)
        grad_p3 =  0.5 * sign * rot90(p1 - p2)
        
        # Build gradient vector in z-space
        grad_z = np.zeros(z_dim)
        grad_z[2*i] = grad_p1[0]
        grad_z[2*i+1] = grad_p1[1]
        grad_z[2*j] = grad_p2[0]
        grad_z[2*j+1] = grad_p2[1]
        grad_z[2*k] = grad_p3[0]
        grad_z[2*k+1] = grad_p3[1]
        
        # Projected gradient step: try moving in gradient direction
        new_z = z + step_size * grad_z
        new_points = new_z.reshape(n, 2)
        
        # Check if this improves the minimum area
        new_areas, _ = _compute_all_triangle_areas(new_points)
        new_min_area = np.min(new_areas)
        
        if new_min_area > min_area:
            z = new_z
            current = new_points
            # Adaptive step size
            step_size = min(step_size * 1.1, 0.5)
        else:
            # Try smaller step
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    # Final evaluation
    final_points = best_z.reshape(n, 2)
    return final_points

def _generate_structured_configs():
    """
    Generate diverse initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack((np.cos(theta), np.sin(theta))))
    
    # 2. Two concentric rings (7 outer, 6 inner)
    theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(theta_outer), np.sin(theta_outer)))
    theta_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack((0.55 * np.cos(theta_inner), 0.55 * np.sin(theta_inner)))
    configs.append(np.vstack([outer, inner]))
    
    # 3. Three rings (5 outer, 4 middle, 4 inner)
    theta1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1 = np.column_stack((np.cos(theta1), np.sin(theta1)))
    theta2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    ring2 = np.column_stack((0.7 * np.cos(theta2), 0.7 * np.sin(theta2)))
    theta3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/8
    ring3 = np.column_stack((0.35 * np.cos(theta3), 0.35 * np.sin(theta3)))
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # 4. Halton sequence
    def halton(n, base):
        result = []
        for i in range(1, n + 1):
            f, r, index = 1.0, 0.0, i
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            result.append(r)
        return np.array(result)
    
    x = halton(13, 2)
    y = halton(13, 3)
    points = np.column_stack((x, y))
    points = (points - 0.5) * 2.0
    configs.append(points)
    
    # 5. Random with fixed seed
    rng = np.random.default_rng(12345)
    configs.append(rng.uniform(-1, 1, (13, 2)))
    
    # 6. Perturbed regular polygon
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    r = 1.0 + 0.1 * np.sin(3 * theta)
    configs.append(np.column_stack((r * np.cos(theta), r * np.sin(theta))))
    
    return configs

def heilbronn_convex13() -> np.ndarray:
    """
    Active-set QP solver for the Heilbronn problem with 13 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # 10 min budget, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Generate initial configurations
    configs = _generate_structured_configs()
    
    # Refine each with active-set solver
    for cfg in configs:
        if time.time() > deadline:
            break
        refined = _active_set_qp_solver(cfg, time_limit=8.0)
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    # Stochastic restarts with refinement
    rng = np.random.default_rng(999)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        if restart_count % 3 == 0:
            # Fully random restart
            start_pts = rng.uniform(-1, 1, (13, 2))
        elif restart_count % 3 == 1:
            # Perturb best
            noise = rng.normal(0, 0.3, (13, 2))
            start_pts = best_points + noise
        else:
            # Mix: half from best, half random
            half = 6
            start_pts = np.vstack([best_points[:half] + rng.normal(0, 0.2, (half, 2)),
                                    rng.uniform(-1, 1, (7, 2))])
        
        refined = _active_set_qp_solver(start_pts, time_limit=5.0)
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
    
    return best_points
