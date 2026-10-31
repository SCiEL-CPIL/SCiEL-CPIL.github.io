# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5c
    # y = c * H
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _soft_min_area(pts, beta=100.0):
    """Smooth approximation of min area. Higher beta is sharper."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    # Soft min: m - (1/beta) * log(sum(exp(-beta*(area - m))))
    # This is differentiable and approximates min(areas)
    shifted = -beta * (areas - m)
    # Clip to avoid overflow in exp, though shifted <= 0 so exp <= 1
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _objective_smooth(flat_pts, beta=100.0):
    """
    Objective for local optimizers.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    # Clamp points for area calculation to avoid numerical issues
    pts_clamped = _clamp_all(pts)
    
    # Use a soft min area for smoothness
    area_val = _soft_min_area(pts_clamped, beta=beta)
    
    # Large penalty weight to keep points inside
    return -area_val + 1000.0 * penalty

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    # Center
    pts.append([1/3, H/3])
    # 3 points near vertices
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # 3 more points in a ring
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _generate_random_starts(num_starts=20):
    """Generate diverse random initial configurations."""
    starts = []
    rng = np.random.default_rng(42)
    
    # Add the symmetric start
    starts.append(_initial_placement_symmetric())
    
    # Add random starts
    for _ in range(num_starts - 1):
        # Random points in triangle
        pts = []
        for _ in range(N):
            # Generate random barycentric coordinates
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            a = 1 - r1 - r2
            b = r1
            c = r2
            p = a * V0 + b * V1 + c * V2
            pts.append(p)
        starts.append(np.array(pts))
        
        # Add some structured random starts (e.g., jittered grid)
        if len(starts) < num_starts:
            grid_pts = []
            # 3x3 grid inside triangle, jittered
            for i in range(3):
                for j in range(3-i):
                    # Barycentric coords for grid
                    a = (3-i-j-1)/3 + rng.normal(0, 0.05)
                    b = (i)/3 + rng.normal(0, 0.05)
                    c = (j)/3 + rng.normal(0, 0.05)
                    # Normalize
                    s = a+b+c
                    a, b, c = a/s, b/s, c/s
                    p = a * V0 + b * V1 + c * V2
                    grid_pts.append(p)
            # Take 11 points from the 6 grid points? No, 3x3 is 6. 
            # Let's just use random for simplicity and speed.
            pass
            
    return starts

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    best_pts = _initial_placement_symmetric()
    best_score = _min_area(best_pts)
    
    # Bounds for local optimization
    # We optimize in Cartesian coordinates but clamp in objective.
    # It's better to optimize in barycentric? No, L-BFGS-B works well with Cartesian if bounds are set.
    # But points must be inside triangle.
    # Let's use the fact that x in [0,1], y in [0,H] is a box, but triangle is a subset.
    # The penalty in _objective_smooth handles the triangle constraint.
    
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    # Generate starts
    starts = _generate_random_starts(30)
    
    # Sort starts by initial score? No, just iterate.
    # To ensure we cover diverse basins, we just run them.
    
    for i, init_pts in enumerate(starts):
        if time_left() < 2.0:
            break
            
        try:
            flat_init = init_pts.flatten()
            
            # Run L-BFGS-B
            # We use a high beta for soft min to get close to true min
            res = minimize(
                lambda x: _objective_smooth(x, beta=100.0),
                flat_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < 1e5: # Check if it didn't blow up
                candidate_pts = _clamp_all(res.x.reshape(-1, 2))
                candidate_score = _min_area(candidate_pts)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts
        except Exception:
            continue
            
    # Final polish on the best point found
    if time_left() > 5.0:
        try:
            # Use Nelder-Mead on the exact objective for final precision
            # NM doesn't use bounds, but our points are already inside.
            # We rely on the clamping in the objective? No, NM objective should be clean.
            # We will use a wrapper that clamps and computes exact min area.
            def _exact_obj(flat_pts):
                pts = _clamp_all(flat_pts.reshape(-1, 2))
                return -_min_area(pts)
            
            res_nm = minimize(
                _exact_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            
            if res_nm.fun < 1e5:
                final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
                final_score = _min_area(final_pts)
                if final_score > best_score:
                    best_pts = final_pts
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
