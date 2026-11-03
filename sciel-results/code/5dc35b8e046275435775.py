import numpy as np
import time
from scipy.spatial import ConvexHull
import sys

N = 13

def _get_min_tri_info(pts):
    """
    Returns (min_area, hull_area, min_indices)
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0, 0.0, (0,1,2)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0, 0.0, (0,1,2)
    except Exception:
        return 0.0, 0.0, (0,1,2)

    min_area = float('inf')
    min_idx = (0, 1, 2)
    
    # Vectorized calculation of all triangle areas
    # i < j < k
    # To speed up, we can use a list of indices
    # But for N=13, loop is fine.
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                # cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)

    return min_area, hull_area, min_idx

def _compute_score(pts):
    min_area, hull_area, _ = _get_min_tri_info(pts)
    if hull_area == 0:
        return -1e9
    return min_area / hull_area

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    p0 = np.column_stack((x0, y0))
    
    best_pts = p0.copy()
    best_score = _compute_score(best_pts)
    
    # Local Search Strategy:
    # 1. Identify min triangle.
    # 2. Try to expand it.
    # 3. Use simulated annealing to escape local minima.
    
    rng = np.random.default_rng(42)
    current_pts = best_pts.copy()
    current_score = best_score
    
    # Annealing parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.99
    
    # We will loop until time limit
    while time.time() - start_time < time_limit - 2.0:
        # Get current min triangle
        min_area, hull_area, min_idx = _get_min_tri_info(current_pts)
        if hull_area == 0:
            # Reset to random
            current_pts = rng.normal(0, 1, (N, 2))
            current_score = _compute_score(current_pts)
            continue
        
        i, j, k = min_idx
        
        # Calculate gradient of area for triangle (i,j,k) wrt points
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
        # Let D = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # We want to increase |D|.
        # Gradient wrt p_i:
        # dD/dx_i = -(y_k - y_i) + (y_j - y_i)*0 ... wait.
        # D = x_j y_k - x_j y_i - x_i y_k + x_i y_i - y_j x_k + y_j x_i + y_i x_k - y_i x_i
        # dD/dx_i = -y_k + y_j
        # dD/dy_i = -x_k + x_j
        # dD/dx_j = y_k - y_i
        # dD/dy_j = -(x_k - x_i)
        # dD/dx_k = -(y_j - y_i)
        # dD/dy_k = x_j - x_i
        
        p_i = current_pts[i]
        p_j = current_pts[j]
        p_k = current_pts[k]
        
        # Determine sign to maximize area (we want to increase |D|)
        D = (p_j[0]-p_i[0])*(p_k[1]-p_i[1]) - (p_j[1]-p_i[1])*(p_k[0]-p_i[0])
        sign = 1.0 if D > 0 else -1.0
        
        # Gradients for moving i, j, k to increase D (assuming sign is fixed, we move in direction of grad * sign)
        # Actually, if we move p_i by delta, D changes by grad_i . delta.
        # We want D * sign to increase.
        # So we move in direction of grad * sign.
        
        grad_i = np.array([-(p_k[1] - p_j[1]), -(p_k[0] - p_j[0])]) * sign
        grad_j = np.array([p_k[1] - p_i[1], -(p_k[0] - p_i[0])]) * sign
        grad_k = np.array([-(p_j[1] - p_i[1]), p_j[0] - p_i[0]]) * sign
        
        # Normalize gradients to have similar magnitude
        norm_i = np.linalg.norm(grad_i)
        norm_j = np.linalg.norm(grad_j)
        norm_k = np.linalg.norm(grad_k)
        
        if norm_i > 1e-9: grad_i /= norm_i
        if norm_j > 1e-9: grad_j /= norm_j
        if norm_k > 1e-9: grad_k /= norm_k
        
        # Step size
        # Adaptive step size based on T
        step = T * 0.1 
        
        # Propose move: push the min triangle vertices
        new_pts = current_pts.copy()
        new_pts[i] += step * grad_i
        new_pts[j] += step * grad_j
        new_pts[k] += step * grad_k
        
        # Evaluate
        new_score = _compute_score(new_pts)
        
        # Acceptance criterion (Simulated Annealing)
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp((new - current) / T)
            # Note: score is small, so difference is small
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
            # else reject
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Random restart / perturbation to escape local min
            # Perturb a random subset of points
            idx = rng.choice(N, size=3, replace=False)
            current_pts[idx] += rng.normal(0, 0.1, size=(3, 2))
            current_score = _compute_score(current_pts)
            T = 0.01 # Reset temp
            
    return best_pts
