import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
SOFT_BETA_INITIAL = 10.0

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
    We minimize this function.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # We want to maximize min(areas), so we minimize -soft_min(areas).
    # soft_min(areas) is an approximation of min(areas).
    # Let's return -soft_min(areas) / hull_area.
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    # Avoid overflow in exp
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return -soft_min_val / hull_area

def _polish_points(points, deadline, start_beta=10.0, max_beta=500.0, steps=5):
    """
    Iteratively polish points using L-BFGS-B with increasing beta.
    """
    best_points = points.copy()
    best_score = _calculate_score(best_points)
    
    # Sequence of betas
    betas = np.linspace(start_beta, max_beta, steps)
    
    for beta in betas:
        if time.time() > deadline - 2.0:
            break
            
        def objective(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=beta)
        
        try:
            res = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished = res.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
            
    return best_score, best_points

def _generate_initial_points(seed):
    """
    Generate diverse initial point sets.
    """
    rng = np.random.RandomState(seed)
    points_list = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    points_list.append(np.column_stack((x, y)))
    
    # 2. Two rings: 7 outer, 7 inner
    # Optimize r and offset for inner ring
    # Try a few fixed configurations first
    for r in [0.5, 0.7, 0.85]:
        angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles_in = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7 # Offset by half step
        x_out = np.cos(angles_out)
        y_out = np.sin(angles_out)
        x_in = r * np.cos(angles_in)
        y_in = r * np.sin(angles_in)
        pts = np.vstack([
            np.column_stack((x_out, y_out)),
            np.column_stack((x_in, y_in))
        ])
        points_list.append(pts)
        
    # 3. Random points in unit disk, ensuring some spread
    for _ in range(5):
        # Sample random points
        # Rejection sampling for uniform in disk
        while True:
            pts = rng.uniform(-1, 1, size=(N, 2))
            dists = np.linalg.norm(pts, axis=1)
            if np.all(dists <= 1.0):
                break
        
        # Check if convex hull is non-degenerate
        try:
            hull = ConvexHull(pts)
            if hull.volume > 0.1:
                points_list.append(pts)
        except Exception:
            pass
            
    # 4. Perturbed regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + rng.normal(0, 0.1, N)
    radii = np.clip(radii, 0.5, 1.5)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points_list.append(np.column_stack((x, y)))
    
    return points_list

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Generate multiple initial guesses
    initial_points = _generate_initial_points(seed=42)
    
    # Polish each initial guess
    for i, pts in enumerate(initial_points):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time based on remaining time
        # We have ~100s left. If 10 points, ~10s each.
        # But polish is fast. Let's just run polish with a tight deadline for each.
        # Actually, just use the global deadline. The polish function will stop early.
        
        score, polished_pts = _polish_points(pts, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = polished_pts
            
    # If we have time, do a final high-precision polish on the best point
    if best_overall_points is not None and time.time() < deadline - 5.0:
        final_score, final_pts = _polish_points(
            best_overall_points, 
            deadline, 
            start_beta=50.0, 
            max_beta=1000.0, 
            steps=3
        )
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_points = final_pts
            
    # Fallback
    if best_overall_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_overall_points = np.column_stack((x, y))
        
    return best_overall_points
