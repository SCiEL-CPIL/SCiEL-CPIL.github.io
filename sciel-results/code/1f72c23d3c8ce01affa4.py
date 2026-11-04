import numpy as np
import time
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all triples
def get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = get_triple_indices(N)

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
    # points shape (N, 2)
    # TRIPLE_INDICES shape (M, 3)
    
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D: x1*y2 - y1*x2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial configurations
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0
    init1 = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    # 2. Two concentric heptagons
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False) # Interleaved
    
    r_in = 0.5
    r_out = 1.0
    inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
    outer_pts = np.column_stack([r_out * np.cos(angles_outer), r_out * np.sin(angles_outer)])
    init2 = np.vstack([inner_pts, outer_pts])
    
    # 3. 1 center, 13 on circle? (Unlikely best, but check)
    # Skip for now, focus on 1 and 2.
    
    candidates = [init1, init2]
    
    best_score = 0
    best_points = None
    
    for init in candidates:
        # Run SA for this init
        current_points = init.copy()
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        # Simulated Annealing
        T = 0.1 # Initial temperature
        T_min = 1e-6
        alpha = 0.999 # Cooling rate
        max_iter = 50000
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Generate neighbor
            # Move one point randomly
            idx = np.random.randint(N)
            delta = np.random.normal(0, T, 2)
            new_points = current_points.copy()
            new_points[idx] += delta
            
            new_score = _score(new_points)
            
            # Acceptance criterion
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            
            T *= alpha
            if T < T_min:
                break
        
        if time.time() > deadline:
            break

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
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configurations
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0
    init1 = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    # 2. Two concentric heptagons (interleaved)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    
    r_in = 0.5
    r_out = 1.0
    inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
    outer_pts = np.column_stack([r_out * np.cos(angles_outer), r_out * np.sin(angles_outer)])
    init2 = np.vstack([inner_pts, outer_pts])
    
    # 3. 1 center, 13 on circle
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    r_out_13 = 1.0
    outer_13 = np.column_stack([r_out_13 * np.cos(angles_13), r_out_13 * np.sin(angles_13)])
    center = np.array([[0.0, 0.0]])
    init3 = np.vstack([center, outer_13])
    
    candidates = [init1, init2, init3]
    
    best_score = 0
    best_points = None
    
    # Run SA on each candidate, keeping the global best
    # To use time efficiently, we can split time among candidates or just run one long SA on the best init.
    # Let's evaluate inits first.
    for init in candidates:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
            
    current_points = best_points.copy()
    current_score = best_score
    
    # Simulated Annealing
    T = 0.05 # Start small enough to not destroy structure, large enough to explore
    T_min = 1e-8
    alpha = 0.9995 # Slow cooling
    
    # Estimate iterations possible
    # _score takes ~1ms? 110s / 1ms = 110,000 iters.
    # Let's run until deadline.
    
    while time.time() < deadline:
        # Generate neighbor
        # Move one or two points? One point is safer for local minima.
        idx = np.random.randint(N)
        # Adaptive step size?
        step_size = T * 2.0
        delta = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= alpha
        if T < T_min:
            # Reheat or stop?
            # If we stop, we might miss better solutions.
            # But if T is too low, we are stuck.
            # Let's just break and rely on the fact that we started good.
            # Or restart with a slightly perturbed best_points?
            # For simplicity and safety, break.
            break

    return best_points
# EVOLVE-BLOCK-END
