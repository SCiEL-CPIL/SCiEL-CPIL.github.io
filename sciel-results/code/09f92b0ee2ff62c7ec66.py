import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score(flat_points):
    """
    Objective function for scipy.optimize.minimize.
    Minimize negative score.
    """
    points = flat_points.reshape((N, 2))
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # Initial configurations
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two concentric heptagons (interleaved)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    r_in = 0.5
    r_out = 1.0
    inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
    outer_pts = np.column_stack([r_out * np.cos(angles_outer), r_out * np.sin(angles_outer)])
    candidates.append(np.vstack([inner_pts, outer_pts]))
    
    # 3. Three rings: 5, 5, 4
    # Just a heuristic guess to break symmetry
    a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    a2 = np.linspace(2*np.pi/5, 2*np.pi + 2*np.pi/5, 5, endpoint=False)
    a3 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
    p1 = np.column_stack([0.4 * np.cos(a1), 0.4 * np.sin(a1)])
    p2 = np.column_stack([0.8 * np.cos(a2), 0.8 * np.sin(a2)])
    p3 = np.column_stack([1.0 * np.cos(a3), 1.0 * np.sin(a3)])
    candidates.append(np.vstack([p1, p2, p3]))

    # 4. Random starts
    for _ in range(5):
        candidates.append(np.random.rand(N, 2) * 2 - 1) # In square [-1, 1]

    best_score = -1
    best_points = None
    
    # Evaluate all candidates first
    for init in candidates:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

    # Optimize each candidate
    for init in candidates:
        if time.time() > deadline:
            break
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            res = minimize(
                _neg_score, 
                init.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'maxfev': 4000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_points = res.x.reshape((N, 2))
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # If time remains, run a quick SA polish on the best point
    if time.time() < deadline - 10:
        current_points = best_points.copy()
        current_score = best_score
        T = 0.01
        alpha = 0.999
        
        while time.time() < deadline - 5:
            idx = np.random.randint(N)
            delta = np.random.normal(0, T, 2)
            new_points = current_points.copy()
            new_points[idx] += delta
            
            new_score = _score(new_points)
            
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            
            T *= alpha
            if T < 1e-8:
                break

    return best_points
