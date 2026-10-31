import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
# Vertices of the equilateral triangle
# A = (0,0), B = (1,0), C = (0.5, sqrt(3)/2)
# Area of this triangle is sqrt(3)/4

def bary_to_cart(u, v):
    # w = 1 - u - v
    # x = u*0 + v*1 + w*0.5
    # y = u*0 + v*0 + w*(sqrt(3)/2)
    w = 1 - u - v
    x = v + 0.5 * w
    y = (3 ** 0.5) / 2 * w
    return np.array([x, y])

def compute_areas_bary(bary_points):
    """
    bary_points: (N, 2) array of (u, v). w is implicit.
    Returns array of areas for all triplets.
    """
    n = len(bary_points)
    # Construct full barycentric matrix (N, 3)
    u = bary_points[:, 0]
    v = bary_points[:, 1]
    w = 1 - u - v
    bary_full = np.column_stack((u, v, w))
    
    areas = []
    # Precompute indices
    indices = list(itertools.combinations(range(n), 3))
    for i, j, k in indices:
        # Determinant of 3x3 matrix formed by barycentric coords
        # | ui vi wi |
        # | uj vj wj |
        # | uk vk wk |
        det = np.linalg.det(bary_full[[i, j, k], :])
        area = (3 ** 0.5) / 4.0 * abs(det)
        areas.append(area)
    return np.array(areas)

def objective_soft_min(bary_flat, t):
    """
    bary_flat: (2N,) array of u, v coordinates.
    Returns negative of soft-min area to be minimized.
    Actually, we minimize the soft-min value which approximates -min_area.
    Soft-min of areas A_k is -1/t * ln(sum(exp(-t*A_k))).
    We want to MAXIMIZE min_area.
    So we MINIMIZE -min_area.
    The soft approximation of min_area is S = -1/t * ln(sum(exp(-t*A_k))).
    So we minimize -S = 1/t * ln(sum(exp(-t*A_k))).
    """
    bary_points = bary_flat.reshape(-1, 2)
    areas = compute_areas_bary(bary_points)
    
    # Avoid log(0) or overflow
    # If areas are very small, exp(-t*A) is close to 1.
    # If areas are large, exp(-t*A) is small.
    # We want to minimize 1/t * ln(sum(exp(-t*A))).
    
    # Numerical stability: subtract max?
    # We are summing exp(-t*A). The largest term is exp(-t*min(A)).
    # Let A_min = min(areas).
    # sum(exp(-t*A)) = exp(-t*A_min) * sum(exp(-t*(A - A_min)))
    # ln(sum) = -t*A_min + ln(sum(exp(-t*(A - A_min))))
    # obj = 1/t * (-t*A_min + ln(sum(...))) = -A_min + (1/t)*ln(sum(...))
    
    A_min = np.min(areas)
    # For stability in the log-sum-exp
    shifted = -t * (areas - A_min)
    # shifted <= 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    
    return -A_min + (1.0 / t) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Random inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        # Sample random points in triangle
        # u, v > 0, u+v < 1
        # Use rejection sampling or transform
        # u = r1 * sqrt(r2), v = (1-r1) * r2 ... no, standard uniform in triangle:
        # u = rng.random(), v = rng.random()
        # if u+v > 1: u, v = 1-u, 1-v
        pts = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            pts.append([u, v])
        inits.append(np.array(pts))
        
    # 2. Grid-like init (Hexagonal packing approximation)
    # 11 points. 1 in center, 6 in inner ring, 4 in outer? 
    # Or just a 3x3 grid clipped?
    # Let's try a simple lattice
    # Side length 1. 
    # Rows of points.
    # Row 0: 1 pt
    # Row 1: 2 pts
    # Row 2: 3 pts
    # Row 3: 2 pts
    # Row 4: 1 pt
    # Total 9. Need 11.
    # Maybe:
    # 1
    # 2
    # 3
    # 3
    # 2
    # Total 11.
    # This is a rough hexagonal arrangement.
    hex_pts = []
    # Define positions in barycentric
    # Center is (1/3, 1/3)
    # Let's just use a regular grid in Cartesian and project? 
    # Simpler: Use the random inits. The optimizer will fix them.
    
    # 3. One specific "good" init if known? 
    # For N=11, the optimal configuration is likely related to hexagonal packing.
    # Let's add a structured init.
    # 1 point at center (1/3, 1/3)
    # 6 points around it at distance d.
    # 4 points at corners? No, corners are bad for min area (small triangles near corners).
    # Best to keep points away from edges if possible, but 11 points is dense.
    
    # Let's just rely on multiple random restarts and annealing.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = init.copy()
        # Annealing loop
        t = 5.0
        # We minimize objective_soft_min
        # Bounds: u in [0,1], v in [0,1]
        # Constraints: u+v <= 1
        
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x.reshape(-1, 2), axis=1)}
        ]
        bounds = [(0, 1)] * (2 * N)
        
        # Optimize
        for step in range(20):
            if time.time() > deadline:
                break
                
            # Evaluate current best
            areas = compute_areas_bary(current_points)
            min_area = np.min(areas)
            
            if min_area > best_score:
                best_score = min_area
                best_points = current_points.copy()
                
            # Optimize with current t
            x0 = current_points.flatten()
            
            # Use SLSQP
            # maxiter per step should be small to allow multiple t increases
            res = minimize(
                objective_soft_min, 
                x0, 
                args=(t,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if res.success:
                current_points = res.x.reshape(-1, 2)
            
            # Increase t
            t *= 2.0
            if t > 1000:
                break

    # Final Polish
    if best_points is not None and time.time() < deadline:
        # Use very high t and tight tolerance
        t = 1000.0
        x0 = best_points.flatten()
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x.reshape(-1, 2), axis=1)}
        ]
        bounds = [(0, 1)] * (2 * N)
        
        # Try a few more steps with high t
        for _ in range(5):
            if time.time() > deadline:
                break
            res = minimize(
                objective_soft_min, 
                x0, 
                args=(t,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.success:
                x0 = res.x
        
        final_points = x0.reshape(-1, 2)
        # Convert to Cartesian
        cart_points = np.array([bary_to_cart(u, v) for u, v in final_points])
        
        # Verify and clamp
        # Ensure points are inside
        # u>=0, v>=0, u+v<=1
        # Small numerical errors might push them out.
        # Clamp barycentric coords
        u = np.clip(final_points[:, 0], 0, 1)
        v = np.clip(final_points[:, 1], 0, 1)
        # Handle u+v > 1
        mask = (u + v) > 1
        if np.any(mask):
            # Scale down u and v proportionally to satisfy u+v=1
            sum_uv = u[mask] + v[mask]
            u[mask] = u[mask] / sum_uv
            v[mask] = v[mask] / sum_uv
        
        final_points_clamped = np.column_stack((u, v))
        cart_points = np.array([bary_to_cart(u, v) for u, v in final_points_clamped])
        
        # Check score of clamped points
        # Re-compute min area to be sure
        # Note: clamping might reduce area slightly, but ensures validity.
        # The optimizer should have kept them inside, so clamping should be minimal.
        
        return cart_points

    # Fallback
    if best_points is not None:
        cart_points = np.array([bary_to_cart(u, v) for u, v in best_points])
        return cart_points
    
    # Random fallback
    pts = []
    rng = np.random.RandomState(0)
    for _ in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        pts.append([u, v])
    return np.array([bary_to_cart(u, v) for u, v in pts])

# EVOLVE-BLOCK-END
