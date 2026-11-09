import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
CENTROID = (A + B + C) / 3.0

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def uv_to_cartesian(u, v):
    """
    Convert (u, v) in [0,1]^2 to Cartesian in the triangle.
    Uses normalization if u+v > 1 to map square to triangle.
    """
    s = u + v
    if s > 1.0:
        u = u / s
        v = v / s
    # alpha = 1 - u - v, beta = u, gamma = v
    x = u * 1.0 + v * 0.5
    y = v * H
    return np.array([x, y])

def params_to_points_full(params):
    """
    Convert 22-dim params (11 points, 2 coords each) to Cartesian points.
    """
    params = np.array(params).reshape(N, 2)
    points = np.zeros((N, 2))
    for i in range(N):
        points[i] = uv_to_cartesian(params[i, 0], params[i, 1])
    return points

def params_to_points_symmetric(params):
    """
    Convert 8-dim params (4 base points, 2 coords each) to 11 Cartesian points.
    Base points: p1, p2, p3, pf
    Structure:
    0: Centroid
    1-3: p1 and rotations
    4-6: p2 and rotations
    7-9: p3 and rotations
    10: pf
    """
    params = np.array(params).reshape(4, 2)
    
    p1 = uv_to_cartesian(params[0, 0], params[0, 1])
    p2 = uv_to_cartesian(params[1, 0], params[1, 1])
    p3 = uv_to_cartesian(params[2, 0], params[2, 1])
    pf = uv_to_cartesian(params[3, 0], params[3, 1])
    
    points = np.zeros((N, 2))
    points[0] = CENTROID
    
    # Rotation matrices
    # 120 degrees
    theta = 2 * np.pi / 3
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R120 = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    # 240 degrees
    R240 = R120 @ R120
    
    def rotate(p, R):
        return CENTROID + R @ (p - CENTROID)
        
    # p1 orbit
    points[1] = p1
    points[2] = rotate(p1, R120)
    points[3] = rotate(p1, R240)
    
    # p2 orbit
    points[4] = p2
    points[5] = rotate(p2, R120)
    points[6] = rotate(p2, R240)
    
    # p3 orbit
    points[7] = p3
    points[8] = rotate(p3, R120)
    points[9] = rotate(p3, R240)
    
    # pf
    points[10] = pf
    
    return points

def objective_full(params):
    points = params_to_points_full(params)
    return -min_triangle_area_vec(points)

def objective_symmetric(params):
    points = params_to_points_symmetric(params)
    return -min_triangle_area_vec(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_params_full = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution on Symmetric 8-dim space
    # This exploits the structural symmetry to find a good basin quickly.
    bounds_sym = [(0.0, 1.0)] * 8
    
    try:
        result_de = differential_evolution(
            objective_symmetric,
            bounds_sym,
            maxiter=150,
            popsize=20,
            seed=42,
            tol=1e-6,
            polish=False
        )
        
        de_score = -result_de.fun
        if de_score > best_score:
            best_score = de_score
            # Convert 8-dim result to 22-dim for full refinement
            # We need to find the (u,v) that produced result_de.x
            # result_de.x is already (u,v) pairs.
            best_params_sym = result_de.x.copy()
            
            # Create a 22-dim parameter array that maps to the same points
            # We can just use the symmetric generator to get the points,
            # but NM needs 22 params.
            # We can initialize the 22 params by inverting the mapping?
            # No, easier: Just run NM on the 22-dim space starting from a random 
            # 22-dim point that maps to the same configuration?
            # Actually, we can just create a 22-dim param array where:
            # p0 = centroid (fixed in symmetric, but in full it's variable)
            # p1..p3 = rotations of p1_sym
            # etc.
            # We need to find (u,v) for each of the 11 points.
            # For the symmetric points, we know their (u,v) from the 8-dim params.
            # For the rotations, we need to find (u,v) that map to the rotated Cartesian coords.
            # This is an inverse mapping problem.
            
            # Alternative: Run NM on the 22-dim space, but initialize it with 
            # the Cartesian points converted back to (u,v)?
            # The mapping (u,v) -> (x,y) is not easily invertible for the normalized case.
            
            # Simpler Approach:
            # We have the best 8-dim params.
            # We can define a "bridge" function that takes 22 params, 
            # but we want to start NM from the symmetric solution.
            
            # Let's just run NM on the 22-dim space with a random start?
            # No, that loses the benefit.
            
            # Better: We can optimize the 22-dim space directly, but use the 
            # symmetric solution as an initial guess for the *points*.
            # Since NM works on params, we need params.
            
            # Let's try to invert the (u,v) -> (x,y) mapping.
            # If u+v <= 1: x = u + 0.5v, y = H*v => v = y/H, u = x - 0.5v.
            # If u+v > 1: x = u/(u+v) + 0.5 v/(u+v), y = H v/(u+v).
            # This is complex.
            
            # Alternative Strategy:
            # Run DE on 22-dim space, but seed the population with the symmetric solution?
            # We can generate the 11 points from the 8-dim symmetric solution.
            # Then, we can just run NM on the 22-dim space starting from a 
            # "random" 22-dim param set that happens to map to these points?
            # No, we can't easily find those params.
            
            # Let's change the refinement step:
            # Instead of NM on 22-dim params, we can do a "Local Search" on the 22-dim 
            # Cartesian space directly? No, NM is for params.
            
            # Okay, let's do this:
            # 1. Get best 8-dim params.
            # 2. Generate 11 points.
            # 3. Run DE on 22-dim params, but initialize the population with 
            #    random points, BUT ALSO include a "smart" initialization?
            #    DE doesn't support custom init easily in all versions, but we can 
            #    just run NM on 22-dim with a random start? That's weak.
            
            # Let's try to invert the mapping for the 11 points.
            # For the 3 base points, we have (u,v).
            # For the rotations, we have (x,y).
            # If the point is inside the triangle, and we assume it came from u+v<=1 region:
            # v = y/H
            # u = x - 0.5v
            # If u+v > 1, we are in the normalized region.
            # Given the symmetry, the points are likely well inside, so u+v < 1 is likely.
            # Let's assume u+v <= 1 for inversion.
            
            def cartesian_to_uv(x, y):
                v = y / H
                u = x - 0.5 * v
                # Check if this is valid (u,v >= 0, u+v <= 1)
                if u < 0 or v < 0 or u + v > 1:
                    # Fallback: use the normalized inversion or just return (0.5, 0.5)
                    # For now, clip
                    u = np.clip(u, 0, 1)
                    v = np.clip(v, 0, 1)
                    s = u + v
                    if s > 1:
                        u /= s
                        v /= s
                return u, v

            points_sym = params_to_points_symmetric(best_params_sym)
            best_params_full = np.zeros(22)
            for i in range(N):
                u, v = cartesian_to_uv(points_sym[i, 0], points_sym[i, 1])
                best_params_full[2*i] = u
                best_params_full[2*i+1] = v
            
    except Exception:
        best_params_full = rng.random(22)

    # Phase 2: Local Refinement with Nelder-Mead on Full 22-dim Space
    # This allows breaking symmetry if it leads to a better local optimum.
    if time.time() < deadline - 10.0:
        try:
            result_nm = minimize(
                objective_full,
                best_params_full,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            nm_score = -result_nm.fun
            if nm_score > best_score:
                best_score = nm_score
                best_params_full = result_nm.x.copy()
                
        except Exception:
            pass

    # Fallback
    if best_params_full is None:
        best_params_full = rng.random(22)
        
    points = params_to_points_full(best_params_full)
    
    return points
