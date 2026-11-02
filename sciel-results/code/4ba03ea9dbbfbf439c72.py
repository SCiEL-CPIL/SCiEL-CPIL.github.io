import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

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
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # In 2D, cross product is scalar: x1*y2 - y1*x2
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Lines:
    # Bottom: y >= 0
    # Left: y <= sqrt(3) * x
    # Right: y <= sqrt(3) * (1 - x)
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    if np.any(y < -eps): return False
    if np.any(y > np.sqrt(3) * x + eps): return False
    if np.any(y > np.sqrt(3) * (1 - x) + eps): return False
    if np.any(x < -eps): return False
    if np.any(x > 1 + eps): return False
    
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Clip y
    y = np.clip(y, 0, S3H)
    
    # Clip based on left and right boundaries
    # y <= sqrt(3) * x  => x >= y / sqrt(3)
    # y <= sqrt(3) * (1 - x) => x <= 1 - y / sqrt(3)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    # Clip x
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _objective(x_flat):
    """
    Objective function for minimization.
    We want to MAXIMIZE min_area, so we MINIMIZE -min_area.
    x_flat: shape (N*2,)
    """
    pts = x_flat.reshape(N, 2)
    pts = _project_to_triangle(pts)
    area = _min_triangle_area(pts)
    return -area

def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return np.array([x, y])

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(p0)
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        
        # Sample candidates
        for _ in range(1000):
            cand = _random_point_in_triangle(rng)
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        
        if best_pos is not None:
            pts.append(best_pos)
            
    return np.array(pts)

def _largest_empty_triangle_heuristic(pts, max_iters=500, rng=None):
    """
    Iteratively identifies the current minimum-area triangle, computes the centroid 
    of the convex hull of all points excluding its three vertices, and projects the 
    centroid onto the triangle domain to generate a new candidate point that replaces 
    the point with the largest potential to increase the minimum area.
    """
    if rng is None:
        rng = random.Random()
        
    pts = np.array(pts, dtype=float)
    current_min_area = _min_triangle_area(pts)
    no_improve_count = 0
    max_no_improve = 20
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # 1. Find the minimum area triangle
        p_i = pts[_i_idx]
        p_j = pts[_j_idx]
        p_k = pts[_k_idx]
        
        dx1 = p_j[:, 0] - p_i[:, 0]
        dy1 = p_j[:, 1] - p_i[:, 1]
        dx2 = p_k[:, 0] - p_i[:, 0]
        dy2 = p_k[:, 1] - p_i[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        
        min_idx = np.argmin(areas)
        min_area_val = areas[min_idx]
        
        # Indices of the three points forming the min area triangle
        i_idx = _i_idx[min_idx]
        j_idx = _j_idx[min_idx]
        k_idx = _k_idx[min_idx]
        
        # 2. Identify the other points
        other_indices = [idx for idx in range(N) if idx not in [i_idx, j_idx, k_idx]]
        other_points = pts[other_indices]
        
        # 3. Compute centroid of convex hull of other points
        try:
            if len(other_points) < 3:
                # Fallback if not enough points (shouldn't happen for N=11)
                hull = ConvexHull(other_points)
            else:
                hull = ConvexHull(other_points)
            
            # The centroid of the convex hull vertices
            hull_vertices = other_points[hull.vertices]
            centroid = np.mean(hull_vertices, axis=0)
            
        except Exception:
            # If convex hull fails (e.g., collinear points), use mean of all other points
            centroid = np.mean(other_points, axis=0)
        
        # 4. Generate candidate points
        # The heuristic suggests replacing one of the min-area triangle's vertices
        # with a point derived from the "empty space" (centroid of the rest).
        # We will generate candidates by moving one of the i, j, k points towards the centroid.
        
        candidates = []
        for idx in [i_idx, j_idx, k_idx]:
            # Move towards centroid
            # We can try different step sizes or just replace with centroid?
            # "replaces the point ... with the largest potential"
            # Let's try replacing the point with the centroid itself, 
            # and also points slightly perturbed from the centroid.
            
            # Candidate 1: Replace with centroid
            new_pts = pts.copy()
            new_pts[idx] = centroid
            new_pts = _project_to_triangle(new_pts)
            candidates.append(new_pts)
            
            # Candidate 2: Move towards centroid by 50%
            mid_point = 0.5 * pts[idx] + 0.5 * centroid
            new_pts = pts.copy()
            new_pts[idx] = mid_point
            new_pts = _project_to_triangle(new_pts)
            candidates.append(new_pts)
            
            # Candidate 3: Random point in the "gap" (random convex combination of centroid and current point)
            t = rng.random()
            gap_point = (1 - t) * pts[idx] + t * centroid
            new_pts = pts.copy()
            new_pts[idx] = gap_point
            new_pts = _project_to_triangle(new_pts)
            candidates.append(new_pts)
            
            # Candidate 4: Random point in the triangle domain, biased towards centroid
            # Sample random point, then blend with centroid
            rand_pt = _random_point_in_triangle(rng)
            blend_t = rng.random() * 0.5
            bias_pt = (1 - blend_t) * rand_pt + blend_t * centroid
            new_pts = pts.copy()
            new_pts[idx] = bias_pt
            new_pts = _project_to_triangle(new_pts)
            candidates.append(new_pts)

        # 5. Evaluate candidates and pick the best
        best_candidate = None
        best_candidate_area = current_min_area
        
        for cand in candidates:
            if not _check_inside(cand):
                continue
            area = _min_triangle_area(cand)
            if area > best_candidate_area + 1e-10:
                best_candidate_area = area
                best_candidate = cand
        
        if best_candidate is not None:
            pts = best_candidate
            current_min_area = best_candidate_area
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Largest Empty Triangle Heuristic followed by Nelder-Mead polish.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # 1. Generate a set of diverse initializations
    initializations = []
    
    # Generate 5 distinct greedy bases
    num_bases = 5
    for _ in range(num_bases):
        try:
            base_pts = _greedy_init(rng)
            initializations.append(base_pts)
            
            # Create 2 perturbations of this base
            for _ in range(2):
                perturbed = base_pts + np.random.normal(0, 0.05, (N, 2))
                perturbed = _project_to_triangle(perturbed)
                initializations.append(perturbed)
        except Exception:
            pass
            
    # Add a few purely random ones just in case
    for _ in range(5):
        u = np.random.random((N, 2))
        mask = u[:, 0] + u[:, 1] > 1
        u[mask] = 1 - u[mask]
        x = u[:, 0] + 0.5 * u[:, 1]
        y = S3H * u[:, 1]
        rand_pts = np.column_stack((x, y))
        initializations.append(rand_pts)
        
    # Shuffle initializations to ensure variety in early runs
    rng.shuffle(initializations)
    
    # 2. Optimize each initialization
    # First apply the Largest Empty Triangle Heuristic, then Nelder-Mead
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        # Apply heuristic
        try:
            heur_pts = _largest_empty_triangle_heuristic(init_pts, max_iters=200, rng=rng)
        except Exception:
            heur_pts = init_pts
            
        # Evaluate heuristic result
        if _check_inside(heur_pts):
            score = _min_triangle_area(heur_pts)
            if score > best_score:
                best_score = score
                best_pts = heur_pts.copy()
        
        # Check time again
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Apply Nelder-Mead polish
        x0 = heur_pts.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000,  # Reduced to allow more restarts
                    'xatol': 1e-5, 
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            pts_opt = res.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception as e:
            # If optimization fails, we already have the heuristic result
            continue
            
        # Check time again
        if time.time() - start_time > DEADLINE - 5:
            break

    # 3. Fallback
    if best_pts is None:
        # Use the best from initializations without optimization
        for init_pts in initializations:
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
                    
    if best_pts is None:
        # Absolute fallback: grid points
        best_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")

