"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 115.0  # Safety margin

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    # Vectorized calculation of all triangle areas
    # Indices i, j, k
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a

def _init_regular(n, r=1.0):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _init_mixed(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    pts_outer = _init_regular(n_outer, r_outer)
    if n_inner > 0:
        # Offset inner ring by half step
        angles_inner = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner
        pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])
    return pts_outer

def construct():
    start_time = time.time()
    
    # Try a few initial configurations to see which basin we fall into
    # 1. Regular 14-gon
    # 2. 7 outer, 7 inner
    # 3. 6 outer, 8 inner (less symmetric)
    
    candidates = [
        _init_regular(14, 1.0),
        _init_mixed(7, 7, 1.0, 0.6),
        _init_mixed(6, 8, 1.0, 0.5),
        _init_mixed(5, 9, 1.0, 0.4),
    ]
    
    best_pts = None
    best_score = 0.0
    
    # Quick evaluation of inits
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()
            
    # Simulated Annealing
    if best_pts is None:
        best_pts = _init_regular(14, 1.0)
        
    current_pts = best_pts.copy()
    current_score = best_score
    
    T_start = 0.05
    T_end = 1e-6
    # Number of steps depends on time. Let's estimate.
    # Each step involves score calc.
    # Let's run as many steps as possible within deadline.
    
    steps = 0
    # Adaptive step size
    step_size = 0.05
    
    while time.time() - start_time < DEADLINE - 2:
        steps += 1
        
        # Propose move
        # Randomly perturb 1 or 2 points
        n_perturb = 1 if np.random.rand() < 0.7 else 2
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        new_pts = current_pts.copy()
        noise = np.random.normal(0, step_size, size=(n_perturb, 2))
        new_pts[indices] += noise
        
        new_score = _score(new_pts)
        
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Metropolis acceptance
            delta = current_score - new_score
            if delta < 1e-12:
                accept = True
            else:
                # T is scaled by current score to be relative?
                # Or absolute? Score is area ratio, around 0.01-0.1?
                # Let's check magnitude.
                # For regular 14-gon area 1, min tri area is small.
                # Let's use relative temperature.
                T = T_start * (T_end / T_start) ** (steps / 50000.0) # Heuristic cooling
                if T < T_end: T = T_end
                
                prob = np.exp(-delta / T)
                accept = np.random.rand() < prob
                
            if accept:
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        # Simple geometric cooling
        # We need to estimate total steps. 
        # Let's assume ~10000-50000 steps.
        # Let's just decrement T based on time or steps.
        
        # Adaptive step size: if we accept too often, increase; too few, decrease?
        # Keep it simple: constant or slowly decreasing step size.
        if steps % 1000 == 0:
             step_size *= 0.95
             
    return best_pts
# EVOLVE-BLOCK-END
