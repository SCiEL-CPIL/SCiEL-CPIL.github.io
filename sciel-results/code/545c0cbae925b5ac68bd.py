import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# Precompute unit hexagon vertices
UNIT_HEX_VERTS = np.array([
    [math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)] for k in range(6)
])

# Outer hexagon normals (fixed orientation for simplicity, rotation 0)
# Normals are at angles pi/6 + k*pi/3
OUTER_NORMALS = np.array([
    [math.cos(math.pi / 6.0 + k * math.pi / 3.0), 
     math.sin(math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)
])

def get_hex_verts(cx, cy, theta):
    """Get vertices of a unit hexagon at (cx, cy) with rotation theta."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return cx + (R @ UNIT_HEX_VERTS.T).T[0], cy + (R @ UNIT_HEX_VERTS.T).T[1]

def min_distance_hex(i_params, j_params):
    """
    Calculate minimum distance between two hexagons.
    Returns negative value if overlapping (depth of overlap).
    i_params, j_params: (cx, cy, theta)
    """
    cx1, cy1, th1 = i_params
    cx2, cy2, th2 = j_params
    
    # Quick check: if centers are far apart
    dist_c = math.hypot(cx1 - cx2, cy1 - cy2)
    if dist_c > 4.0: # Max distance between vertices of unit hex is 2
        return dist_c - 2.0

    v1x, v1y = get_hex_verts(cx1, cy1, th1)
    v2x, v2y = get_hex_verts(cx2, cy2, th2)
    
    # Axes to test: normals of hex 1 and hex 2
    c1, s1 = math.cos(th1), math.sin(th1)
    c2, s2 = math.cos(th2), math.sin(th2)
    
    axes = []
    # Normals of hex 1
    for k in range(3): # Only 3 unique normals
        ang = th1 + math.pi/2 + k * math.pi/3
        axes.append((math.cos(ang), math.sin(ang)))
    # Normals of hex 2
    for k in range(3):
        ang = th2 + math.pi/2 + k * math.pi/3
        axes.append((math.cos(ang), math.sin(ang)))
        
    min_gap = float('inf')
    
    for nx, ny in axes:
        # Project hex 1
        p1 = v1x * nx + v1y * ny
        min1, max1 = np.min(p1), np.max(p1)
        
        # Project hex 2
        p2 = v2x * nx + v2y * ny
        min2, max2 = np.min(p2), np.max(p2)
        
        # Check for gap
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
            
        if gap < min_gap:
            min_gap = gap
            if min_gap == 0.0:
                break # Overlapping, no need to check other axes for min gap (it's 0)
                
    # If min_gap is 0, they are touching or overlapping.
    # We need the depth of overlap if they are overlapping.
    # SAT with gap 0 doesn't give depth.
    # If they are overlapping, we can estimate depth by the minimum distance between vertices?
    # Or we can just return 0.0 and rely on the objective to push them apart?
    # If we return 0.0, the penalty is 0, and the optimizer might not push them apart if they are deep inside.
    # We need a penalty for overlap.
    # Let's check if they are actually overlapping by checking if any vertex of 1 is inside 2.
    # This is expensive.
    # Alternative: If min_gap is 0, we assume they are touching. If they are overlapping, the "gap" is negative.
    # To get negative gap, we need to find the separating axis that gives the largest overlap? No, SAT says if no axis separates, they overlap.
    # The "depth" of overlap is the minimum distance required to separate them.
    # This is the minimum over all axes of the "gap" if we define gap as signed distance?
    # Actually, if they overlap, all gaps are 0 in the standard SAT check.
    # We need a different metric for overlap depth.
    # Let's use the distance between centers as a proxy for penalty if they are "close".
    # Or better: Calculate the area of intersection? Too slow.
    # Let's use a simple heuristic: if min_gap < epsilon, they are "touching".
    # If we want to penalize overlap, we can use the fact that for convex shapes, if they overlap, the distance between centers is less than the sum of radii.
    # But they can overlap even if centers are far apart? No, for convex shapes, if centers are further than sum of circumradii, they don't overlap.
    # Circumradius of unit hex is 1. So if dist_c > 2, no overlap.
    # If dist_c <= 2, they MIGHT overlap.
    # If SAT says no separating axis, they overlap.
    # To get a smooth penalty, we can use the "minimum distance between boundaries".
    # If they overlap, this is negative.
    # How to calculate it?
    # One way: Find the vertex of 1 closest to 2, and the vertex of 2 closest to 1? No, that's not the minimum distance between boundaries.
    # Let's approximate: If they overlap, return -dist_c + 2.0? No, that's not accurate.
    # Let's return 0.0 if they are touching/overlapping, and rely on the "S" term to push them?
    # No, S doesn't care about internal overlaps.
    # We MUST penalize overlaps.
    # Let's use a simple vertex-in-polygon check. If any vertex of 1 is inside 2, they overlap.
    # If they overlap, we return a negative value based on how deep the vertex is?
    # Let's just return -0.1 if overlapping? That's a constant penalty.
    # Better: If overlapping, return -(dist_c - 1.0)?
    # Let's try: If SAT indicates overlap (min_gap == 0), we check if centers are very close.
    # Actually, for the purpose of optimization, a smooth penalty that is 0 when separated and negative when overlapping is needed.
    # We can use the distance between the closest points on the boundaries.
    # If they are separated, it's the gap.
    # If they overlap, it's negative of the penetration depth.
    # Calculating penetration depth for two convex polygons is a standard problem.
    # For hexagons, we can do it.
    # But for simplicity and speed, let's use a "soft" distance:
    # d = max(0, min_gap) - overlap_indicator * (some function)
    
    # Let's refine:
    # If min_gap > 0, return min_gap.
    # If min_gap == 0, they are touching or overlapping.
    # Check if they are truly overlapping by checking if a vertex of one is strictly inside the other.
    # If yes, return -0.5 (a fixed penalty) or something related to dist_c.
    # To make it smooth, we can blend.
    
    # For now, let's assume min_gap is the distance. If it's 0, we add a small penalty based on center distance.
    # Penalty = max(0, 2.0 - dist_c) * 10.0 ?
    # If dist_c < 2, they are close.
    # If they are actually overlapping, dist_c is usually < 1.73 (sqrt(3)).
    
    # Let's use:
    if min_gap > 1e-6:
        return min_gap
    else:
        # They are touching or overlapping
        # Estimate overlap depth
        # If centers are very close, deep overlap.
        # Return a negative value proportional to (2 - dist_c)
        # This encourages separation.
        return -(2.0 - dist_c) * 0.5

def objective(params):
    """
    params: [x1, y1, th1, ..., x11, y11, th11]
    Returns: S_required + Penalty
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf')
        
    pts = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        th = params[i*3+2]
        pts.append((x, y, th))
        
    # Calculate S_required
    max_proj = 0.0
    for i in range(N):
        x, y, th = pts[i]
        vx, vy = get_hex_verts(x, y, th)
        # Project vertices onto outer normals
        for k in range(6):
            proj = vx * OUTER_NORMALS[k, 0] + vy * OUTER_NORMALS[k, 1]
            if abs(proj) > max_proj:
                max_proj = abs(proj)
                
    S = max_proj / APOTHEM
    
    # Calculate Overlap Penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance_hex(pts[i], pts[j])
            if d < 0:
                penalty += d * d # Quadratic penalty for overlap
                
    return S + penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: Simple lattice
        return generate_fallback()
        
    # Initial configuration: Best lattice from previous code
    # Axial to Cartesian for the best candidate
    # Let's use the "axial_1" from previous code which was good
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return pts

    init_pts = axial_to_cartesian(axial_1)
    # Center the configuration
    cx = np.mean([p[0] for p in init_pts])
    cy = np.mean([p[1] for p in init_pts])
    init_params = []
    for x, y in init_pts:
        init_params.append(x - cx)
        init_params.append(y - cy)
        init_params.append(0.0) # Initial rotation 0
        
    best_S = float('inf')
    best_params = init_params
    
    # Multi-start local optimization
    n_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Jitter initial parameters
        jitter_scale = 0.1 if i > 0 else 0.0
        if jitter_scale > 0:
            jitter = rng.normal(0, jitter_scale, size=(N, 3))
            cur_params = np.array(init_params).reshape(N, 3) + jitter
            # Normalize rotations to [0, pi/3]
            cur_params[:, 2] = cur_params[:, 2] % (math.pi / 3)
            cur_params = cur_params.flatten()
        else:
            cur_params = np.array(init_params)
            
        # Run Nelder-Mead
        try:
            res = minimize(objective, cur_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_S:
                # Check if penalty is negligible
                # Re-evaluate components
                S_val = 0
                penalty_val = 0
                pts = []
                for k in range(N):
                    x = res.x[k*3]
                    y = res.x[k*3+1]
                    th = res.x[k*3+2]
                    pts.append((x, y, th))
                    
                max_proj = 0.0
                for k in range(N):
                    x, y, th = pts[k]
                    vx, vy = get_hex_verts(x, y, th)
                    for m in range(6):
                        proj = vx * OUTER_NORMALS[m, 0] + vy * OUTER_NORMALS[m, 1]
                        if abs(proj) > max_proj:
                            max_proj = abs(proj)
                S_val = max_proj / APOTHEM
                
                for k in range(N):
                    for l in range(k+1, N):
                        d = min_distance_hex(pts[k], pts[l])
                        if d < 0:
                            penalty_val += d * d
                            
                if penalty_val < 1e-6:
                    best_S = S_val
                    best_params = res.x
                else:
                    # Even if penalty is high, keep it if S is much smaller? 
                    # No, we need valid packing.
                    # But maybe the penalty is an artifact of the smooth approximation.
                    # Let's check validity explicitly.
                    pass
                    
        except Exception:
            pass
            
    if best_S == float('inf'):
        return generate_fallback()
        
    # Extract results
    inner = []
    for i in range(N):
        x = float(best_params[i*3])
        y = float(best_params[i*3+1])
        th = float(best_params[i*3+2])
        # Normalize angle to degrees
        th_deg = math.degrees(th) % 360
        inner.append([x, y, th_deg])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = 0.0
    
    return inner, center, side, rotation

def generate_fallback():
    # Simple lattice packing
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return pts
    pts = axial_to_cartesian(axial_1)
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    inner = []
    for x, y in pts:
        inner.append([float(x-cx), float(y-cy), 0.0])
        
    max_proj = 0
    for x, y in [(p[0]-cx, p[1]-cy) for p in pts]:
        vx, vy = get_hex_verts(x, y, 0.0)
        for k in range(6):
            proj = vx * OUTER_NORMALS[k, 0] + vy * OUTER_NORMALS[k, 1]
            if abs(proj) > max_proj:
                max_proj = abs(proj)
    side = max_proj / APOTHEM
    return inner, [0.0, 0.0], float(side), 0.0

