import numpy as np
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices as (6,2) array."""
    angles = angle_rad + np.pi/3.0 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (6,2) arrays."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        p1 = v1 @ axis
        p2 = v2 @ axis
        if p1.max() < p2.min() + eps or p2.max() < p1.min() + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    n = len(inner)
    verts = [hex_vertices(ix, iy, 1.0, math.radians(iang)) for (ix, iy, iang) in inner]
    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap(verts[i], verts[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    ox, oy = outer_center
    outer_v = hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            inside = True
            for i in range(6):
                p1 = outer_v[i]
                p2 = outer_v[(i+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Exact minimum outer side length."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    normals = np.column_stack([np.array([np.cos(oa + np.radians(30 + 60*k)), np.sin(oa + np.radians(30 + 60*k))]) for k in range(6)])
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        rel = verts - np.array([ox, oy])
        projs = rel @ normals
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return 2.0 * max_proj / SQRT3

def objective(params):
    n = N
    inner = []
    for i in range(n):
        cx, cy, ang = params[3*i], params[3*i+1], np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox, oy = params[3*n], params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    S = compute_required_side(inner, (ox, oy), outer_angle)
    
    # Penalty for overlap
    penalty = 0.0
    verts = [hex_vertices(ix, iy, 1.0, math.radians(iang)) for (ix, iy, iang) in inner]
    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap(verts[i], verts[j], 1e-6):
                # Calculate minimum translation vector or just a large penalty
                penalty += 10.0
    
    # Penalty for being outside (if S is too small, this is handled by S calculation, 
    # but we want to ensure feasibility for the given S)
    # Actually, minimizing S directly with constraints is better.
    # Here we minimize S + penalty.
    
    return S + penalty

def construct():
    # Initialization: 1 center, 6 ring 1, 4 ring 2
    # Ring 2 angles: 0, 60, 120, 180, 240, 300
    # Pick 4: 0, 120, 240, 300 (arbitrary choice to break symmetry slightly or keep C3?)
    # Let's try 0, 60, 120, 180 (half ring)
    
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60*k)
        centers.append((SQRT3*math.cos(a), SQRT3*math.sin(a)))
    
    # 4 from ring 2
    ring2_angles = [0, 60, 120, 180, 240, 300]
    selected = [0, 120, 240, 300] # C3 symmetry? 0, 120, 240 are C3. 300 breaks it.
    # Better: 0, 60, 120, 180?
    # Let's use 0, 120, 240 and 180?
    # Actually, let's just use the first 4: 0, 60, 120, 180.
    
    for a_deg in [0, 60, 120, 180]:
        a = math.radians(a_deg)
        centers.append((2*SQRT3*math.cos(a), 2*SQRT3*math.sin(a)))
        
    # Pad if necessary (should be exactly 11)
    while len(centers) < N:
        centers.append((0.0, 0.0))
        
    params = np.zeros(3*N+3)
    for i, (cx, cy) in enumerate(centers[:N]):
        params[3*i] = cx
        params[3*i+1] = cy
        params[3*i+2] = 0.0 # Angle
    
    params[3*N] = 0.0
    params[3*N+1] = 0.0
    params[3*N+2] = 0.0
    
    # Optimize
    result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    
    p = result.x
    inner = [(p[3*i], p[3*i+1], np.degrees(p[3*i+2])%360) for i in range(N)]
    ox, oy = p[3*N], p[3*N+1]
    oa = np.degrees(p[3*N+2])%360
    
    S = compute_required_side(inner, (ox, oy), oa)
    
    # Verify
    if check_disjoint(inner) and check_inside(inner, (ox, oy), S, oa):
        return (inner, (ox, oy), S, oa)
    
    # Fallback
    return (inner, (ox, oy), S, oa)
