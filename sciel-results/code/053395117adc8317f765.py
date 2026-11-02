import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    This provides a decent starting point for the NLP.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        # Sample many candidates
        for _ in range(500):
            cand = np.array(_random_point_in_triangle(rng))
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """
    Structured initialization: Vertices, Midpoints, Center, and some other points.
    """
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    p_struct = np.array(p_struct)
    p_struct = _project_to_triangle(p_struct)
    return p_struct

def _active_set_search(pts, max_iter=500, time_limit=5.0):
    """
    Derivative-free active-set local search.
    1. Find the minimum area triangle (active set).
    2. Move its vertices to maximize that specific triangle's area, while keeping other points fixed.
    3. Use Nelder-Mead for this sub-problem.
    4. Repeat until no improvement or time limit reached.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_min_area = _min_triangle_area(current_pts)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find the active set: indices of the triangle with minimum area
        p_i = current_pts[_i_idx]
        p_j = current_pts[_j_idx]
        p_k = current_pts[_k_idx]
        
        dx1 = p_j - p_i
        dx2 = p_k - p_i
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # If we've improved significantly or reached a very high area, we might be close to optimal
        # But we always try to improve the current min.
        
        # The active set is the three points forming this triangle.
        i, j, k = _triplets[min_idx]
        active_indices = np.array([i, j, k])
        
        # Get current positions of active points
        active_pts = current_pts[active_indices].copy()
        
        # Define the sub-objective: maximize the area of the triangle formed by these 3 points.
        # We can do this by moving the points freely within the triangle.
        # However, we must ensure they don't collide with other points in a way that reduces OTHER areas below the current min?
        # Actually, the standard active set method for max-min problems is:
        # Maximize f(x) where f(x) is the area of the critical triangle.
        # But we must ensure that no other triangle area becomes smaller than the current min_area.
        # This is a constrained optimization.
        
        # Simpler heuristic: Just maximize the area of the critical triangle, 
        # then check if the global min area increased. If not, revert or try a different move.
        
        # Let's use Nelder-Mead to maximize the area of the critical triangle.
        # Variables: 6 coordinates (2 for each of the 3 points).
        
        def sub_obj(x_flat):
            # x_flat: 6 coordinates
            new_active = x_flat.reshape(3, 2)
            # Project to triangle to ensure validity
            new_active = _project_to_triangle(new_active)
            
            # Construct full point set
            new_pts = current_pts.copy()
            new_pts[active_indices] = new_active
            
            # Calculate area of the critical triangle
            pi, pj, pk = new_pts[i], new_pts[j], new_pts[k]
            area = 0.5 * np.abs(np.cross(pj - pi, pk - pi))
            
            # We want to maximize this, so minimize -area
            return -area
        
        x0 = active_pts.flatten()
        
        try:
            res = minimize(sub_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success:
                new_active = _project_to_triangle(res.x.reshape(3, 2))
                new_pts = current_pts.copy()
                new_pts[active_indices] = new_active
                
                new_min_area = _min_triangle_area(new_pts)
                
                if new_min_area > current_min_area:
                    current_pts = new_pts
                    current_min_area = new_min_area
                else:
                    # If it didn't improve the global min, maybe we got stuck in a local max of the sub-triangle
                    # that hurts other triangles. 
                    # In a true active set method, we would solve a QP. 
                    # For now, if it doesn't improve, we might need to break the tie by perturbing.
                    # Or we can try moving points in a random direction that increases the critical area slightly.
                    pass
        except Exception:
            pass
        
        # Convergence check: if improvement is tiny, stop
        # (Hard to check without knowing previous min, but we can track history)
        
    return current_pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Active-Set Local Search with Nelder-Mead sub-optimizations.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured
    p_struct = _structured_init()
    initializations.append(p_struct)
    
    # 4. Lattice-based (more diverse)
    # Try a few more random starts to diversify
    for _ in range(3):
        p_extra = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_extra = _project_to_triangle(p_extra)
        initializations.append(p_extra)
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Run active set search
        try:
            # Allocate more time to better initializations?
            # For now, fixed time per start.
            # We have ~145s total. 5 starts. ~25s each? 
            # But Nelder-Mead is fast. Let's give 10s per start initially, then extend if time remains.
            # Actually, the first start (greedy) is likely the best. Give it more time.
            time_alloc = 30.0 if s == 0 else 10.0
            
            if time.time() - start_time + time_alloc > DEADLINE - 5:
                time_alloc = max(1.0, DEADLINE - 5 - (time.time() - start_time))
                
            optimized_pts = _active_set_search(init_pts, max_iter=200, time_limit=time_alloc)
            
            if _check_inside(optimized_pts):
                score = _min_triangle_area(optimized_pts)
                if score > best_score:
                    best_score = score
                    best_pts = optimized_pts.copy()
        except Exception:
            pass
    
    # If we have time left, do a final polish on the best point
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        remaining_time = DEADLINE - 10 - (time.time() - start_time)
        try:
            polished_pts = _active_set_search(best_pts, max_iter=500, time_limit=remaining_time)
            if _check_inside(polished_pts):
                score = _min_triangle_area(polished_pts)
                if score > best_score:
                    best_score = score
                    best_pts = polished_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
