import numpy as np
import time
import scipy.spatial as spat
import scipy.optimize
import itertools

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Efficient vectorized triangle area calculation
    # Pre-computed indices for N=13 are not strictly necessary if we generate them once,
    # but for 13 points, generating indices every time is fast enough (<0.1ms).
    # However, to be safe and fast, let's generate them efficiently.
    
    # Indices for all unique triples (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _affine_transform(params, base_points):
    """
    Apply affine transformation defined by params to base_points.
    params: [scale_x, scale_y, shear, rotation]
    Transformation matrix M:
    [ sx * cos(r)    -sy * sin(r) ]
    [ sx * shear * cos(r) - sy * shear * sin(r) + sy * cos(r) ] -> This is getting complicated.
    
    Let's define a simpler affine model:
    x' = ax * x + bx * y
    y' = ay * x + by * y
    
    We have 4 parameters: a, b, c, d corresponding to the matrix [[a, b], [c, d]].
    We can parametrize this as:
    a = sx * cos(theta)
    b = -sy * sin(theta)
    c = sx * shear * cos(theta) - sy * shear * sin(theta) ? No.
    
    Let's just use 4 free parameters for the 2x2 matrix entries, but constrain them to be non-degenerate.
    Or better, use polar-like coordinates to ensure we cover a good range:
    Let M = R * S * H
    R = rotation matrix
    S = diagonal scale matrix
    H = shear matrix
    
    M = [[cos, -sin], [sin, cos]] * [[sx, 0], [0, sy]] * [[1, sh], [0, 1]]
    
    M = [[sx*cos, sx*sh*cos - sy*sin],
         [sx*sin, sx*sh*sin + sy*cos]]
         
    params = [sx, sy, sh, theta]
    """
    sx, sy, sh, theta = params
    c, s = np.cos(theta), np.sin(theta)
    
    # M = R * S * H
    # R = [[c, -s], [s, c]]
    # S = [[sx, 0], [0, sy]]
    # H = [[1, sh], [0, 1]]
    
    # S*H = [[sx, sx*sh], [0, sy]]
    # R*(S*H) = [[c*sx, c*sx*sh - s*sy], [s*sx, s*sx*sh + c*sy]]
    
    m00 = sx * c
    m01 = sx * sh * c - sy * s
    m10 = sx * s
    m11 = sx * sh * s + sy * c
    
    # Apply to base_points
    transformed = np.empty_like(base_points)
    transformed[:, 0] = m00 * base_points[:, 0] + m01 * base_points[:, 1]
    transformed[:, 1] = m10 * base_points[:, 0] + m11 * base_points[:, 1]
    
    return transformed

def _negative_score(params, base_points):
    """
    Objective function for minimization: negative of the score.
    """
    pts = _affine_transform(params, base_points)
    score = calculate_score(pts)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Base configuration: 4x4 grid minus one corner (top-right)
    # This creates a shape close to a square but with a "bite" taken out,
    # which might allow better distribution of interior points.
    # Coordinates:
    # (0,0) (1,0) (2,0) (3,0)
    # (0,1) (1,1) (2,1) (3,1)
    # (0,2) (1,2) (2,2) (3,2)
    # (0,3) (1,3) (2,3)       <- (3,3) missing
    
    base_coords = []
    for i in range(4):
        for j in range(4):
            if not (i == 3 and j == 3): # Remove top-right
                base_coords.append([i, j])
                
    base_points = np.array(base_coords, dtype=float)
    # Center the base points to avoid bias in rotation/shear around origin
    base_points -= np.mean(base_points, axis=0)
    
    # Initial parameters: Identity-like transformation
    # sx=1, sy=1, sh=0, theta=0
    initial_params = np.array([1.0, 1.0, 0.0, 0.0])
    
    # Define bounds for parameters to prevent degenerate transformations
    # sx, sy > 0 (scale)
    # sh can be positive or negative
    # theta in [0, 2pi)
    bounds = [
        (0.1, 5.0),   # sx
        (0.1, 5.0),   # sy
        (-2.0, 2.0),  # sh
        (0.0, 2*np.pi) # theta
    ]
    
    # Use Nelder-Mead or L-BFGS-B? 
    # The objective is non-smooth (min of linear functions), so gradient-based methods might struggle.
    # Nelder-Mead is derivative-free and robust to non-smoothness.
    # However, Nelder-Mead can be slow. 
    # Let's try Nelder-Mead with multiple restarts from different initial points.
    
    # Generate multiple initial configurations
    np.random.seed(42)
    inits = [initial_params.copy()]
    
    # Add some random perturbations
    for _ in range(10):
        sx = np.random.uniform(0.5, 2.0)
        sy = np.random.uniform(0.5, 2.0)
        sh = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0, 2*np.pi)
        inits.append(np.array([sx, sy, sh, theta]))
        
    # Also try specific symmetric distortions
    inits.append(np.array([1.0, 1.0, 0.5, 0.0]))
    inits.append(np.array([1.0, 1.0, -0.5, 0.0]))
    inits.append(np.array([1.0, 1.0, 0.0, np.pi/4]))
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
            
        try:
            result = scipy.optimize.minimize(
                _negative_score, 
                init, 
                args=(base_points,), 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < 0: # fun is negative score, so < 0 means positive score
                final_params = result.x
                final_pts = _affine_transform(final_params, base_points)
                final_score = calculate_score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
                    
        except Exception as e:
            continue
            
    # If optimization didn't yield a good result, fallback to a simple regular polygon
    if best_points is None or best_score < 1e-6:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)
        
    return best_points
