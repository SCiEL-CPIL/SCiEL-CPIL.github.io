import numpy as np
import math
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Unroll for speed if possible, but N=14 is small
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            # Vectorize k loop?
            # k > j
            if j + 1 < n:
                pk = points[j+1:]
                # Area = 0.5 * | (pj - pi) x (pk - pi) |
                # Cross product in 2D: (x1*y2 - y1*x2)
                v1 = pj - pi
                v2 = pk - pi
                areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
                min_area = min(min_area, np.min(areas))
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area

def _simulated_annealing(points_init, max_iters=50000, T_start=1.0, T_end=1e-6, alpha=0.999):
    """
    Perform Simulated Annealing to maximize the score.
    """
    current_points = points_init.copy()
    current_score = _score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = T_start
    # Estimate cooling factor to reach T_end in max_iters
    # T_end = T_start * alpha^max_iters
    # alpha = (T_end / T_start) ** (1 / max_iters)
    # But we can just use a fixed alpha or compute it.
    # Let's compute alpha to ensure we reach T_end.
    if max_iters > 0:
        alpha = (T_end / T_start) ** (1.0 / max_iters)
    
    rng = np.random.default_rng(123)
    
    for i in range(max_iters):
        # Generate neighbor
        # Pick a random point to move
        idx = rng.integers(0, N)
        step_size = 0.05 * (1 + T) # Scale step with temperature? Or just T?
        # Better: step_size proportional to T to allow large moves early
        # But T starts at 1.0 and goes to 1e-6.
        # Initial points are in [-1, 1]. A step of 0.1 is reasonable.
        # Let's use step = 0.1 * T_normalized?
        # Simpler: fixed small step for late stage, larger for early.
        # Let's just use a Gaussian step scaled by a base factor.
        step = 0.05
        
        new_points = current_points.copy()
        # Move point idx by Gaussian noise
        delta = rng.normal(0, step, size=2)
        new_points[idx] += delta
        
        # Optional: Keep points within bounds? 
        # The score is normalized by hull area, so scale doesn't matter much, 
        # but relative positions do. 
        # If points fly apart, hull area grows, min area grows, ratio might stay similar.
        # But if points cluster, min area drops.
        # Let's not bound strictly, but maybe reflect or clip to [-2, 2] to prevent drift.
        new_points = np.clip(new_points, -2.0, 2.0)
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = new_points.copy()
        else:
            # Calculate probability
            # If T is very small, exp(negative large) is 0
            if T > 1e-9:
                # Use log to avoid underflow? 
                # delta_score = new_score - current_score (negative)
                # p = exp(delta_score / T)
                delta = new_score - current_score
                # If delta is very negative and T is small, this is 0.
                # To speed up, if delta < -10*T, prob is ~0.
                if delta > -10 * T: 
                    p = math.exp(delta / T)
                    if rng.random() < p:
                        current_points = new_points
                        current_score = new_score
                # else: reject
            # else: reject
        
        # Cool
        T *= alpha
        
    return best_points, best_score

def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    
    # Initialize with a regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Run Simulated Annealing
    # 50,000 iterations should be fast enough in Python if _score is optimized.
    # _score calls _min_triangle_area (loop) and _convex_hull_area (scipy).
    # ConvexHull is the bottleneck. 
    # For N=14, ConvexHull is fast but not instant.
    # 50,000 * ~10us = 0.5s? Maybe more like 50us -> 2.5s.
    # Let's try 20,000 iterations to be safe.
    
    best_points, best_score = _simulated_annealing(points_init, max_iters=20000)
    
    # Optional: A very short local polish with Nelder-Mead to squeeze out more?
    # The directive says "do not reuse the current designs". 
    # The current design is DE + NM. 
    # Using NM as a *polish* step after SA might be seen as reusing a component.
    # However, "approach family" usually refers to the main driver.
    # To be safe and strictly distinct, I will stick to SA only.
    # Or I can use a different local method, like "Powell" or just return SA result.
    # Given the non-smoothness, SA's best found point is likely good.
    
    return best_points
