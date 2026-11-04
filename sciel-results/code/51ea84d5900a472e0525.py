import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

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
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
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

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    # Initial configurations
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(init1)
    
    # 2. Two concentric heptagons (interleaved)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    
    # Try a few different radii for the inner ring
    for r_in in [0.3, 0.5, 0.7, 0.8]:
        inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        candidates.append(np.vstack([inner_pts, outer_pts]))
        
    # 3. One center, 13 on circle
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    outer_13 = np.column_stack([np.cos(angles_13), np.sin(angles_13)])
    center = np.array([[0.0, 0.0]])
    candidates.append(np.vstack([center, outer_13]))
    
    # 4. Random points in unit disk (few restarts)
    np.random.seed(42)
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        candidates.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 5. Perturbed regular polygon
    for _ in range(3):
        pert = np.random.normal(0, 0.1, (N, 2))
        candidates.append(init1 + pert)

    best_score = 0
    best_points = None
    current_points = None
    current_score = 0
    
    # Evaluate all candidates and pick the best starting point for Nelder-Mead
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            current_points = c.copy()
            current_score = s
            
    if best_points is None:
        return init1

    # Nelder-Mead Optimization
    # We will run Nelder-Mead on the current best, and potentially restart if it gets stuck or time allows.
    
    # Helper to run Nelder-Mead with a time limit
    def run_nm(pts, time_limit):
        start = time.time()
        # Nelder-Mead doesn't have a direct time limit, so we wrap the function
        # to check time and raise an exception to stop it.
        
        def objective(x):
            if time.time() > time_limit:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return _neg_score(p)
            
        try:
            res = minimize(objective, pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < 0: # Check if we found a valid point
                new_pts = res.x.reshape(N, 2)
                return new_pts
        except TimeoutError:
            pass
        except Exception:
            pass
        return None

    # Run Nelder-Mead from the best initial candidate
    if time.time() < deadline - 5:
        time_limit = deadline - 2
        res_pts = run_nm(current_points, time_limit)
        if res_pts is not None:
            s = _score(res_pts)
            if s > best_score:
                best_score = s
                best_points = res_pts
                current_points = res_pts
                current_score = s

    # If we still have time, try a few more random restarts with short Nelder-Mead
    while time.time() < deadline - 10:
        # Generate a random candidate
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        cand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Quick local search / Nelder-Mead with tight time limit
        time_limit = deadline - 5
        res_pts = run_nm(cand, time_limit)
        
        if res_pts is not None:
            s = _score(res_pts)
            if s > best_score:
                best_score = s
                best_points = res_pts
                current_points = res_pts
                current_score = s

    return best_points

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

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
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial configurations
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(init1)
    
    # 2. Two concentric heptagons (interleaved)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    
    for r_in in [0.3, 0.5, 0.7, 0.8]:
        inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        candidates.append(np.vstack([inner_pts, outer_pts]))
        
    # 3. One center, 13 on circle
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    outer_13 = np.column_stack([np.cos(angles_13), np.sin(angles_13)])
    center = np.array([[0.0, 0.0]])
    candidates.append(np.vstack([center, outer_13]))
    
    # 4. Random points in unit disk
    np.random.seed(42)
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        candidates.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 5. Perturbed regular polygon
    for _ in range(3):
        pert = np.random.normal(0, 0.1, (N, 2))
        candidates.append(init1 + pert)

    best_score = 0
    best_points = None
    current_points = None
    current_score = 0
    
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            current_points = c.copy()
            current_score = s
            
    if best_points is None:
        return init1

    # Nelder-Mead Optimization
    def run_nm(pts, time_limit):
        start = time.time()
        def objective(x):
            if time.time() > time_limit:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return -_score(p)
            
        try:
            res = minimize(objective, pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                return new_pts
        except TimeoutError:
            pass
        except Exception:
            pass
        return None

    if time.time() < deadline - 5:
        time_limit = deadline - 2
        res_pts = run_nm(current_points, time_limit)
        if res_pts is not None:
            s = _score(res_pts)
            if s > best_score:
                best_score = s
                best_points = res_pts
                current_points = res_pts
                current_score = s

    while time.time() < deadline - 10:
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        cand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        time_limit = deadline - 5
        res_pts = run_nm(cand, time_limit)
        
        if res_pts is not None:
            s = _score(res_pts)
            if s > best_score:
                best_score = s
                best_points = res_pts
                current_points = res_pts
                current_score = s

    return best_points
# EVOLVE-BLOCK-END
