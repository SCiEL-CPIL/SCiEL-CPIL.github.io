import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
DEADLINE = 135.0  # Safety margin before 150s kill
_START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Returns (min_area, indices_of_min_triangle).
    """
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min_val = np.min(areas)
                if local_min_val < min_area:
                    min_area = local_min_val
                    # Find the index of the minimum in the vectorized part
                    local_min_pos = np.argmin(areas)
                    k = j + 1 + local_min_pos
                    min_idx = (i, j, k)
                    
    return min_area, min_idx

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area, _ = _min_triangle_area(points)
    return min_area / hull_area

def _generate_initial_points():
    """
    Generate diverse initial configurations.
    """
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts)
    
    # 2. Two rings (inner 4, outer 10) - often better for Heilbronn
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inits.append(np.vstack((inner_pts, outer_pts)))
    
    # 3. Random points in unit disk
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts_rand)
        
    return inits

def _local_optimize(points):
    """
    Use Nelder-Mead to polish a configuration.
    """
    def _objective_hard(p):
        pts = p.reshape(N, 2)
        return -_score(pts)
    
    try:
        res = minimize(
            _objective_hard,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < _objective_hard(points.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def _simulated_annealing(points, time_budget):
    """
    Simulated Annealing with geometric moves.
    """
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Initial temperature
    T = 0.05
    cooling_rate = 0.995
    
    # Move types:
    # 1. Random point move
    # 2. Targeted move on bottleneck triangle
    
    while time.time() - start_time < time_budget:
        if _time_left() < 20:
            break
            
        move_type = np.random.random()
        
        new_points = current_points.copy()
        
        if move_type < 0.4:
            # Random perturbation of one point
            idx = np.random.randint(N)
            noise = np.random.normal(0, T, 2)
            new_points[idx] += noise
        elif move_type < 0.8:
            # Targeted move: push out vertices of the minimum area triangle
            min_area, min_idx = _min_triangle_area(current_points)
            i, j, k = min_idx
            
            # Calculate normals to push vertices out
            # We want to increase the area of triangle (i, j, k)
            # Gradient of area w.r.t point i is 0.5 * cross(p_j - p_k, n_i) ... 
            # Simpler: move each vertex slightly away from the centroid of the other two
            p_i = current_points[i]
            p_j = current_points[j]
            p_k = current_points[k]
            
            # Centroid of the other two
            c_jk = (p_j + p_k) / 2.0
            dir_i = p_i - c_jk
            norm_i = np.linalg.norm(dir_i)
            if norm_i > 1e-8:
                dir_i /= norm_i
            else:
                dir_i = np.random.normal(0, 1, 2)
                dir_i /= np.linalg.norm(dir_i)
                
            c_ik = (p_i + p_k) / 2.0
            dir_j = p_j - c_ik
            norm_j = np.linalg.norm(dir_j)
            if norm_j > 1e-8:
                dir_j /= norm_j
            else:
                dir_j = np.random.normal(0, 1, 2)
                dir_j /= np.linalg.norm(dir_j)
                
            c_ij = (p_i + p_j) / 2.0
            dir_k = p_k - c_ij
            norm_k = np.linalg.norm(dir_k)
            if norm_k > 1e-8:
                dir_k /= norm_k
            else:
                dir_k = np.random.normal(0, 1, 2)
                dir_k /= np.linalg.norm(dir_k)
                
            step_size = T * 0.5
            new_points[i] += dir_i * step_size
            new_points[j] += dir_j * step_size
            new_points[k] += dir_k * step_size
        else:
            # Swap two points (useful for symmetry breaking)
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            new_points[idx1], new_points[idx2] = current_points[idx2], current_points[idx1]
            
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = current_score - new_score
            # Probability of accepting worse solution
            # Use a soft threshold
            if np.random.random() < np.exp(-delta / max(T, 1e-8)):
                current_points = new_points
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        
    return best_points, best_score

def construct():
    start_time = time.time()
    
    inits = _generate_initial_points()
    
    best_points = inits[0].copy()
    best_score = _score(best_points)
    
    # Phase 1: Local optimization of initial points
    for init in inits:
        if _time_left() < 30:
            break
        optimized = _local_optimize(init)
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized
            
    # Phase 2: Simulated Annealing
    # We run SA for the remaining time, with a buffer for final polish
    time_for_sa = max(0, _time_left() - 20)
    if time_for_sa > 5:
        sa_points, sa_score = _simulated_annealing(best_points, time_for_sa)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
            
    # Phase 3: Final Local Polish
    if _time_left() > 10:
        final_points = _local_optimize(best_points)
        final_score = _score(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score

    # Reconstruct and normalize
    points = best_points
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        points = points * scale
        
    return points
