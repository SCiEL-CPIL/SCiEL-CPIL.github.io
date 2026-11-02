import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triples i < j < k
TRIPLES = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)
N_TRIPLES = TRIPLES.shape[0]

def _min_triangle_area(pts):
    """
    Compute the minimum area of triangles formed by N points.
    pts: (N, 2) array of Cartesian coordinates.
    Returns the minimum area (float).
    """
    # pts: (N, 2)
    # For each triple (i, j, k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # We can vectorize this.
    
    # Extract points for each triple
    # pi: (N_TRIPLES, 2)
    # pj: (N_TRIPLES, 2)
    # pk: (N_TRIPLES, 2)
    
    pi = pts[TRIPLES[:, 0]]
    pj = pts[TRIPLES[:, 1]]
    pk = pts[TRIPLES[:, 2]]
    
    # Vectors
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product (2D scalar)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(pts, tol=1e-8):
    """
    Check if all points are inside or on the boundary of the reference triangle.
    Reference triangle vertices: (0,0), (1,0), (0.5, S3H).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Conditions for inside triangle:
    # 1. y >= -tol
    # 2. y <= S3 * x + tol  (Left edge: y = sqrt(3) x)
    # 3. y <= S3 * (1 - x) + tol (Right edge: y = sqrt(3) (1-x))
    
    c1 = y >= -tol
    c2 = y <= S3 * x + tol
    c3 = y <= S3 * (1 - x) + tol
    
    return np.all(c1 & c2 & c3)

def _project_to_triangle(pts):
    """
    Project points to the interior of the triangle.
    Uses a simple iterative clipping method.
    """
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    
    for _ in range(10):
        # Clip to bounding box first
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, S3H)
        
        # For each y, valid x range is [y/S3, 1 - y/S3]
        x_min = y / S3
        x_max = 1 - y / S3
        
        # If x is out of range, clamp it
        # This ensures the point is inside the triangle for the given y
        x = np.clip(x, x_min, x_max)
        
        # Update y? No, x is now valid for this y.
        # But does clamping x make y invalid?
        # The condition y <= S3H is already handled by clip.
        # The conditions y <= S3*x and y <= S3*(1-x) are equivalent to x >= y/S3 and x <= 1 - y/S3.
        # So if x is in [y/S3, 1 - y/S3], the point is inside.
        
        pts[:, 0] = x
        pts[:, 1] = y
        
        # Check if all are inside
        if _is_inside_triangle(pts):
            break
            
    return pts

def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = v * S3H
    return np.array([x, y])

def _fps_initialization(rng, n_points=20):
    """
    Generate initial points using Farthest Point Sampling.
    Start with a random point, then repeatedly pick the point that maximizes
    the minimum distance to existing points.
    """
    pts = []
    # Start with a random point
    p = _random_point_in_triangle(rng)
    pts.append(p)
    
    for _ in range(n_points - 1):
        # Generate candidate points
        candidates = np.array([_random_point_in_triangle(rng) for _ in range(200)])
        
        # For each candidate, calculate min distance to existing points
        # dist[i, j] = distance between candidate i and existing point j
        # We want the candidate with the largest min distance.
        
        min_dists = np.full(len(candidates), np.inf)
        for p_existing in pts:
            d = np.linalg.norm(candidates - p_existing, axis=1)
            min_dists = np.minimum(min_dists, d)
            
        # Pick the candidate with the largest min distance
        best_idx = np.argmax(min_dists)
        pts.append(candidates[best_idx])
        
    return np.array(pts[:n_points])

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin (limit is 150s)
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    # Objective function for optimizer
    # We maximize min_area. So we minimize -min_area.
    # We add a penalty if points go outside the triangle to help the optimizer.
    def obj(flat):
        pts = flat.reshape(-1, 2)
        
        # Penalty for being outside
        # Simple penalty: distance outside squared
        # Or just a large constant if outside?
        # Let's use a smooth penalty.
        x = pts[:, 0]
        y = pts[:, 1]
        
        # g1 = y
        # g2 = S3*x - y
        # g3 = S3*(1-x) - y
        # Violation if g < 0.
        viol = np.minimum(0, y) + np.minimum(0, S3*x - y) + np.minimum(0, S3*(1-x) - y)
        pen = np.sum(viol**2) * 10.0
        
        min_a = _min_triangle_area(pts)
        return -min_a + pen

    # Generate initial configurations
    configs = []
    
    # 1. FPS based
    fps_pts = _fps_initialization(rng, N)
    configs.append(fps_pts)
    
    # 2. Random
    for _ in range(5):
        rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        configs.append(rand_pts)
        
    # 3. Grid-like / Symmetric
    # A simple grid inside the triangle
    grid_pts = []
    for i in range(4):
        for j in range(3):
            # Map (i,j) to triangle coords
            # This is a rough mapping
            u = (i + 0.5) / 4
            v = (j + 0.5) / 3
            if u + v <= 1:
                x = u + 0.5 * v
                y = v * S3H
                grid_pts.append([x, y])
    # Pad with random if not enough
    while len(grid_pts) < N:
        grid_pts.append(_random_point_in_triangle(rng))
    configs.append(np.array(grid_pts[:N]))
    
    # 4. Perturbed FPS
    for _ in range(3):
        p = fps_pts + rng.normal(0, 0.1, size=fps_pts.shape)
        configs.append(p)
        
    # Optimization Loop
    # Use Powell's method as it handles non-smoothness better than L-BFGS-B for min-area.
    # Or Nelder-Mead. Powell is often more robust.
    
    for i, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
            
        init_pts = _project_to_triangle(init_pts.copy())
        
        # Evaluate initial
        area = _min_triangle_area(init_pts)
        if _is_inside_triangle(init_pts):
            if area > best_area:
                best_area = area
                best_pts = init_pts.copy()
                
        # Optimize
        try:
            # Powell
            result = minimize(
                obj,
                init_pts.flatten(),
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            
            opt_pts = result.x.reshape(-1, 2)
            opt_pts = _project_to_triangle(opt_pts)
            
            area = _min_triangle_area(opt_pts)
            
            if _is_inside_triangle(opt_pts):
                if area > best_area:
                    best_area = area
                    best_pts = opt_pts.copy()
        except Exception:
            pass
            
    # If we have time, do more random restarts with Powell
    while time.time() < deadline:
        init_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        init_pts = _project_to_triangle(init_pts.copy())
        
        try:
            result = minimize(
                obj,
                init_pts.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            
            opt_pts = result.x.reshape(-1, 2)
            opt_pts = _project_to_triangle(opt_pts)
            
            area = _min_triangle_area(opt_pts)
            
            if _is_inside_triangle(opt_pts):
                if area > best_area:
                    best_area = area
                    best_pts = opt_pts.copy()
        except Exception:
            pass
            
    if best_pts is None:
        # Fallback
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    return best_pts
