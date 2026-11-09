import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_min_triangle_area(points):
    """
    Compute the minimum triangle area among all C(13,3) triangles.
    Returns min_area and the indices of the triangle forming it.
    """
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    
    # Precompute all pairs? No, just loop. 286 is small.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
                    
    return min_area, min_idx

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    min_area, _ = _compute_min_triangle_area(points)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def _halton_sequence(index, base):
    """
    Generate the index-th Halton sequence number for a given base.
    """
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def _generate_initial_points(strategy, rng):
    """
    Generate diverse initial points based on a strategy string.
    """
    n = 13
    if strategy == "polygon":
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack((np.cos(theta), np.sin(theta)))
    elif strategy == "halton":
        # Halton sequence in 2D
        points = np.array([[_halton_sequence(i + 1, 2), _halton_sequence(i + 1, 3)] for i in range(n)])
        # Center and scale
        points -= np.mean(points, axis=0)
        points /= np.max(np.linalg.norm(points, axis=1))
    elif strategy == "grid":
        # 4x4 grid minus one corner, slightly jittered
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        points = np.array([[x, y] for x in xs for y in ys])
        points = points[:13] # Take first 13
        # Jitter to break symmetry
        points += rng.normal(0, 0.05, (13, 2))
        points -= np.mean(points, axis=0)
        points /= np.max(np.linalg.norm(points, axis=1))
    elif strategy == "rings":
        # 1 center, 4 inner, 8 outer?
        pts = []
        # Center
        pts.append([0, 0])
        # Inner ring (4 points)
        theta_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        r_inner = 0.5
        for t in theta_inner:
            pts.append([r_inner * np.cos(t), r_inner * np.sin(t)])
        # Outer ring (8 points)
        theta_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        r_outer = 1.0
        for t in theta_outer:
            pts.append([r_outer * np.cos(t), r_outer * np.sin(t)])
        points = np.array(pts)
        points += rng.normal(0, 0.05, (13, 2))
        points -= np.mean(points, axis=0)
        points /= np.max(np.linalg.norm(points, axis=1))
    else:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return points

def _local_refinement(points, max_iters=1000, step_size=0.05, rng=None):
    """
    Deterministic local refinement:
    1. Find the minimum area triangle.
    2. Move one of its vertices slightly towards the centroid of the other two? 
       Or move it away from the others to increase area?
       Actually, to increase the MIN area, we should move the vertex of the smallest triangle 
       in a direction that increases the area of that specific triangle, while hopefully 
       not decreasing others too much.
       
       A simpler heuristic: Move the vertex of the worst triangle in a random direction 
       with small steps, accepting only if the global score improves. 
       But to be deterministic-ish, we can try moving towards the "center of mass" of the 
       other points to spread out, or away from the other two vertices of the worst triangle.
       
       Let's try: For the worst triangle (i,j,k), move point i in the direction 
       opposite to the vector from i to the midpoint of jk? No, that reduces area.
       Move i in the direction that maximizes the area of triangle (i,j,k) locally?
       The area is 0.5 * |base * height|. To increase area, increase height.
       So move i perpendicular to the line segment jk, away from the line.
    """
    current_points = points.copy()
    current_score = _score(current_points)
    
    for _ in range(max_iters):
        min_area, (i, j, k) = _compute_min_triangle_area(current_points)
        
        # Get points
        pi = current_points[i]
        pj = current_points[j]
        pk = current_points[k]
        
        # Vector jk
        v_jk = pk - pj
        # Normal to jk
        norm_jk = np.array([-v_jk[1], v_jk[0]])
        norm_jk /= (np.linalg.norm(norm_jk) + 1e-9)
        
        # Determine which side of the line jk point i is on
        # Cross product of (pi - pj) and norm_jk
        # If positive, move in direction of norm_jk. If negative, move in -norm_jk.
        cross_val = np.cross(pi - pj, norm_jk)
        if cross_val > 0:
            direction = norm_jk
        else:
            direction = -norm_jk
            
        # Move point i in this direction by a small step
        # Try a few step sizes
        improved = False
        for step_mult in [1.0, 0.5, 0.1]:
            step = step_size * step_mult
            new_pi = pi + direction * step
            new_points = current_points.copy()
            new_points[i] = new_pi
            
            new_score = _score(new_points)
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                improved = True
                break
                
        if not improved:
            # If local move didn't help, try a small random jitter on the worst point
            # to escape a local plateau
            jitter = np.random.default_rng().normal(0, step_size * 0.1, 2)
            new_points = current_points.copy()
            new_points[i] += jitter
            new_score = _score(new_points)
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
            else:
                # No improvement, break to avoid infinite loop
                break
                
    return current_points, current_score

def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic geometry-aware optimization with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    strategies = ["polygon", "halton", "grid", "rings"]
    rng = np.random.default_rng(42)
    
    # Phase 1: Multiple diverse initializations with local refinement
    for strat in strategies:
        if time.time() > deadline:
            break
            
        # Try a few random seeds for each strategy
        for seed in range(3):
            if time.time() > deadline:
                break
                
            rng_local = np.random.default_rng(seed + 1000)
            init_points = _generate_initial_points(strat, rng_local)
            
            # Local refinement
            refined_points, refined_score = _local_refinement(init_points, max_iters=200, step_size=0.05)
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
                
    # Phase 2: Global search with larger steps, using the best point as a reference
    # We can use a simple hill climbing with larger steps on the best point found
    if best_points is not None:
        current_points = best_points.copy()
        current_score = best_score
        
        # Iterate with decreasing step size
        step_sizes = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
        
        for step_size in step_sizes:
            if time.time() > deadline:
                break
                
            improved = True
            while improved and time.time() < deadline:
                improved = False
                min_area, (i, j, k) = _compute_min_triangle_area(current_points)
                
                # Try moving each of the 13 points
                for idx in range(13):
                    if time.time() > deadline:
                        break
                        
                    # Try moving in several directions
                    # 1. Random directions
                    for _ in range(10):
                        direction = rng.normal(0, 1, 2)
                        direction /= np.linalg.norm(direction)
                        
                        for step_mult in [1.0, 0.5, 0.1]:
                            step = step_size * step_mult
                            new_points = current_points.copy()
                            new_points[idx] += direction * step
                            
                            new_score = _score(new_points)
                            if new_score > current_score:
                                current_points = new_points
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()

    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
