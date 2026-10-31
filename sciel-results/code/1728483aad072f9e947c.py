"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _min_area_bary(points_bary):
    """
    Calculate the minimum area ratio of triangles formed by N points 
    given in barycentric coordinates (u, v) where w = 1 - u - v.
    The area of a triangle formed by points i, j, k is proportional to 
    the absolute value of the determinant of their barycentric coordinates.
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    min_area = float('inf')
    
    # Iterate over all unique triplets
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Determinant of the 3x3 matrix with rows (u, v, w)
                # | u_i v_i w_i |
                # | u_j v_j w_j |
                # | u_k v_k w_k |
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                     - v[i] * (u[j] * w[k] - w[j] * u[k])
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                area = abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """
    Objective function for minimization.
    Minimizes negative min area + penalty for invalid points.
    """
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for points outside the simplex (u,v,w >= 0)
    # Since bounds are [0,1] for u and v, u<0 and v<0 are prevented by optimizer.
    # We only need to check w < 0 (i.e., u+v > 1)
    penalty = 0.0
    if np.any(w < 0):
        # Quadratic penalty for violation
        violations = -w[w < 0]
        penalty = 100.0 * np.sum(violations ** 2)
    
    # Calculate min area
    # If points are invalid, area calculation might be weird, 
    # but penalty should dominate. 
    # For safety, if penalty is high, we can return a large number.
    if penalty > 1e-6:
         return penalty - 1e-3 # Slightly negative to encourage moving in
    
    min_a = _min_area_bary(uv)
    return -min_a + penalty

def _generate_grid_init():
    """Generate an initial set of points based on a grid in barycentric coords."""
    # Create a grid of points in the simplex
    # We want roughly 11 points.
    # A 4x4 grid in the triangle has 10 points (0,0), (0.25,0)... 
    # Let's use a finer grid and pick 11.
    
    # Barycentric grid generation
    points = []
    step = 0.2
    for i in range(int(1/step) + 1):
        u = i * step
        for j in range(int((1-u)/step) + 1):
            v = j * step
            w = 1 - u - v
            if w >= -1e-9:
                points.append([u, v])
    
    # We have 15 points in a 0.2 step grid.
    # Let's pick 11 of them.
    # To maximize spread, we should pick corners and center-ish points.
    # Corners: (1,0), (0,1), (0,0) -> in u,v: (1,0) is invalid for bary (w=-1).
    # Barycentric vertices are (1,0,0)->(1,0), (0,1,0)->(0,1), (0,0,1)->(0,0).
    # So (1,0), (0,1), (0,0) are the vertices.
    
    # Let's just take the first 11 points from the grid list generated above.
    # The list generation order:
    # i=0 (u=0): v=0, 0.2, 0.4, 0.6, 0.8, 1.0 -> 6 points
    # i=1 (u=0.2): v=0, 0.2, 0.4, 0.6 -> 4 points
    # i=2 (u=0.4): v=0, 0.2, 0.4 -> 3 points
    # i=3 (u=0.6): v=0, 0.2 -> 2 points
    # i=4 (u=0.8): v=0 -> 1 point
    # Total 16 points.
    
    # Let's select a subset that looks well-distributed.
    # Indices:
    # 0: (0,0)
    # 1: (0, 0.2)
    # 2: (0, 0.4)
    # 3: (0, 0.6)
    # 4: (0, 0.8)
    # 5: (0, 1.0)
    # 6: (0.2, 0)
    # 7: (0.2, 0.2)
    # 8: (0.2, 0.4)
    # 9: (0.2, 0.6)
    # 10: (0.4, 0)
    # 11: (0.4, 0.2)
    # 12: (0.4, 0.4)
    # 13: (0.6, 0)
    # 14: (0.6, 0.2)
    # 15: (0.8, 0)
    
    # Let's pick: Corners (0,0), (1,0) is not in list? 
    # Wait, u=1, v=0 is in the list?
    # i=5 (u=1.0): v=0. 
    # My loop: `range(int(1/step) + 1)` -> 0 to 5.
    # If u=1, `int((1-1)/step)+1` = 1. j=0. v=0. Point (1,0).
    # So points list contains (1,0) at the end.
    
    # Let's just use the first 11 points generated, or better, a specific set.
    # A good heuristic is to take points that are far apart.
    # For simplicity, let's take a regular subset.
    # Indices: 0, 5, 15 (corners), 2, 10, 7 (midpoints of edges?), 12 (center), ...
    
    selected_indices = [0, 5, 15, 2, 10, 7, 12, 1, 13, 4, 8]
    # Check if these are valid indices (len(points) is 16).
    if len(points) < 16:
         # Fallback to first 11
         selected_indices = list(range(min(11, len(points))))
         
    init = np.array([points[i] for i in selected_indices])
    return init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -float('inf')
    best_points_bary = None
    
    # Generate initializations
    inits = []
    
    # 1. Grid-based init
    try:
        grid_init = _generate_grid_init()
        inits.append(grid_init)
    except Exception:
        pass
        
    # 2. Random inits
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Sample from Dirichlet distribution to get valid barycentric coords
        # alpha=1 gives uniform distribution over the simplex
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2]) # Take u, v
        
    # 3. Perturbed grid init
    try:
        grid_perturbed = _generate_grid_init() + rng.uniform(-0.05, 0.05, size=(N, 2))
        # Clamp to valid range roughly
        grid_perturbed[:, 0] = np.clip(grid_perturbed[:, 0], 0, 1)
        grid_perturbed[:, 1] = np.clip(grid_perturbed[:, 1], 0, 1)
        # Ensure u+v <= 1
        sum_uv = grid_perturbed[:, 0] + grid_perturbed[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            factor = 1.0 / sum_uv[mask]
            grid_perturbed[mask, 0] *= factor
            grid_perturbed[mask, 1] *= factor
        inits.append(grid_perturbed)
    except Exception:
        pass
        
    # Optimization loop
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # Flatten init for optimizer
            x0 = init.flatten()
            
            # Bounds for u and v are [0, 1]
            bounds = [(0, 1)] * (2 * N)
            
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Check result
            uv_opt = res.x.reshape(N, 2)
            u_opt = uv_opt[:, 0]
            v_opt = uv_opt[:, 1]
            w_opt = 1 - u_opt - v_opt
            
            # Check validity
            if np.any(u_opt < -1e-5) or np.any(v_opt < -1e-5) or np.any(w_opt < -1e-5):
                continue
                
            # Calculate score
            score = _min_area_bary(uv_opt)
            
            if score > best_score:
                best_score = score
                best_points_bary = uv_opt.copy()
                
        except Exception:
            pass
            
    # If no valid points found, use the grid init
    if best_points_bary is None:
        best_points_bary = _generate_grid_init()
        
    # Convert barycentric to Cartesian
    # Reference triangle: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points_bary[i]
        w = 1 - u - v
        # Clamp small negatives to 0 for numerical stability
        u = max(0, u)
        v = max(0, v)
        w = max(0, w)
        # Normalize if necessary (should be 1)
        s = u + v + w
        if s > 0:
            u /= s
            v /= s
            w /= s
            
        p = u * A + v * B + w * C
        pts.append(p)
        
    return np.array(pts)

# EVOLVE-BLOCK-END
