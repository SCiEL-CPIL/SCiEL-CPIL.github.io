import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14
# Precompute triple indices
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    
    # Check for degenerate cases: points too close
    # This helps avoid numerical issues
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-10:
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Extract points for all triples
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for minimize.
    Reshapes flat array to (N, 2) and returns negative score.
    """
    points = points_flat.reshape(N, 2)
    score = _calculate_score(points)
    return -score

def _generate_regular_polygon(r=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _generate_two_ring(n_out, n_in, r_in, theta_in):
    """
    Generate points in two concentric rings.
    n_out points on outer ring (radius 1.0), n_in points on inner ring (radius r_in).
    theta_in is the angular offset for the inner ring.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_in
    x_in = r_in * np.cos(angles_in)
    y_in = r_in * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _generate_random_points(seed):
    rng = np.random.default_rng(seed)
    # Generate random points in unit disk
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _optimize_from_start(start_points, time_limit):
    """
    Run Nelder-Mead optimization from a starting point.
    Returns (best_score, best_points).
    """
    start_time = time.time()
    best_score = _calculate_score(start_points)
    best_points = start_points.copy()
    
    # Flatten points for scipy.optimize
    x0 = start_points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e9
        return _negative_score(x)
        
    try:
        # Nelder-Mead is good for low-dimensional local optimization
        # We use multiple restarts with different tolerances to balance speed and quality
        res = minimize(
            _negative_score, 
            x0, 
            method='Nelder-Mead', 
            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000, 'disp': False}
        )
        
        if time.time() - start_time < time_limit:
             new_points = res.x.reshape(N, 2)
             new_score = _calculate_score(new_points)
             if new_score > best_score:
                 best_score = new_score
                 best_points = new_points
                 
    except Exception:
        pass
        
    return best_score, best_points

def _polish(points, time_limit):
    """
    A more aggressive local search using coordinate descent or small perturbations
    followed by Nelder-Mead.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _calculate_score(current_points)
    
    # Try small random perturbations
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Perturb each point slightly
        perturbed = current_points + rng.normal(0, 0.01, (N, 2))
        new_score = _calculate_score(perturbed)
        
        if new_score > current_score:
            current_points = perturbed
            current_score = new_score
    
    # Final Nelder-Mead polish
    score, pts = _optimize_from_start(current_points, max(0, time_limit - (time.time() - start_time)))
    if score > current_score:
        current_score = score
        current_points = pts
        
    return current_score, current_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Start with Regular 14-gon
    reg_points = _generate_regular_polygon(1.0)
    score, pts = _optimize_from_start(reg_points, 15.0)
    if score > best_score:
        best_score = score
        best_points = pts
    
    # Polish the regular polygon result
    if time.time() < deadline - 30:
        score, pts = _polish(best_points, 10.0)
        if score > best_score:
            best_score = score
            best_points = pts

    if time.time() > deadline - 20:
        return best_points if best_points is not None else reg_points

    # 2. Try Two-Ring configurations
    # The 7+7 configuration is known to be strong.
    # We explore a range of inner radii and offsets.
    ring_configs = [
        (7, 7, 0.5, 0.0),
        (7, 7, 0.6, 0.0),
        (7, 7, 0.7, 0.0),
        (7, 7, 0.8, 0.0),
        (7, 7, 0.7, np.pi/7), # Offset by half a step
        (7, 7, 0.6, np.pi/7),
        (7, 7, 0.5, np.pi/7),
        (8, 6, 0.7, 0.0),
        (6, 8, 0.7, 0.0),
        (9, 5, 0.8, 0.0),
        (10, 4, 0.8, 0.0),
        (11, 3, 0.9, 0.0),
        (12, 2, 0.95, 0.0),
    ]
    
    for n_out, n_in, r, theta in ring_configs:
        if time.time() > deadline - 15:
            break
        start_pts = _generate_two_ring(n_out, n_in, r, theta)
        score, pts = _optimize_from_start(start_pts, 5.0)
        if score > best_score:
            best_score = score
            best_points = pts
        else:
            # Even if not better, maybe polish this specific config?
            # Only if we have time
            if time.time() < deadline - 10:
                score, pts = _polish(start_pts, 3.0)
                if score > best_score:
                    best_score = score
                    best_points = pts

    if time.time() > deadline - 20:
        return best_points if best_points is not None else reg_points

    # 3. Random Starts with Local Optimization
    for seed in range(1, 15):
        if time.time() > deadline - 10:
            break
        start_pts = _generate_random_points(seed)
        score, pts = _optimize_from_start(start_pts, 5.0)
        if score > best_score:
            best_score = score
            best_points = pts

    # 4. Perturbed Regular Polygon
    for noise_level in [0.1, 0.2, 0.5]:
        if time.time() > deadline - 5:
            break
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.normal(0, noise_level, N)
        r = 1.0 + np.random.normal(0, noise_level/10, N)
        r = np.maximum(r, 0.1)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        start_pts = np.column_stack((x, y))
        score, pts = _optimize_from_start(start_pts, 5.0)
        if score > best_score:
            best_score = score
            best_points = pts

    # Final polish on the best found
    if time.time() < deadline - 5 and best_points is not None:
        score, pts = _polish(best_points, 10.0)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = reg_points
        
    return best_points
