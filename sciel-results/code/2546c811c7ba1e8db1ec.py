import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 145.0  # Safety margin from 150s

def _hex_vertices(cx, cy, s, rot):
    """
    Compute vertices of a regular hexagon with side s, center (cx, cy), rotation rot (degrees).
    Returns list of (x, y) tuples.
    """
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """
    Compute vertices of a unit regular hexagon (side 1) with center (cx, cy), rotation ang_deg.
    Returns list of (x, y) tuples.
    """
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    """
    Compute the minimum side length S of a regular hexagon centered at outer_c
    with rotation rot (degrees) that contains all inner hexagons.
    """
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    # Apothem h = S * sqrt(3) / 2
    # Max projection P = h
    # S = 2 * P / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    """
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA + 1e-6 < minB or maxB + 1e-6 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    """
    Strictly check if the packing is valid.
    """
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    
    # Check containment
    for p in polys:
        if not _in_outer(p, outer_pts):
            return False
            
    # Check overlaps
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _generate_initial_config(rng):
    """
    Generate a plausible initial configuration of 11 hexagons.
    Uses a flower-like structure: 1 center, 6 neighbors, 4 outer.
    """
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 neighbors at distance sqrt(3)
    d1 = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        x = d1 * math.cos(a)
        y = d1 * math.sin(a)
        # Add small random perturbation
        x += rng.gauss(0, 0.01)
        y += rng.gauss(0, 0.01)
        inner.append([x, y, 0.0])
        
    # 4 outer at distance 2*sqrt(3) ? No, distance 3 is better for hex grid.
    # In a hex grid of side 1, centers are at distance sqrt(3).
    # A "flower" of 7 has radius sqrt(3).
    # Adding 4 more...
    # Let's try a 4-3-3-4-3-3-4-3-3-4-1 structure? No, 11 is small.
    # Standard 11 packing often uses a distorted flower.
    # Let's place 4 more at distance 2*sqrt(3) * cos(30) = 3?
    # Distance between centers in hex lattice is sqrt(3).
    # Second shell distance is 2*sqrt(3) * cos(30) = 3? No.
    # Vector addition: (1,0) + (1,0) in axial coords?
    # Let's just use a grid-like expansion.
    d2 = 2.0 * d1 # ~3.46
    # Actually, for 11 hexagons, a common optimal arrangement is a central one,
    # 6 surrounding it, and 4 more in the "corners" of the outer hexagon.
    # The distance to the 4 outer ones in a tight packing is roughly 3.0.
    for k in range(4):
        a = math.radians(30 + k * 60)
        x = 3.0 * math.cos(a)
        y = 3.0 * math.sin(a)
        x += rng.gauss(0, 0.01)
        y += rng.gauss(0, 0.01)
        inner.append([x, y, 0.0])
        
    return inner

def _optimize_config(inner, outer_c, rot, rng, budget):
    """
    Optimize the configuration using Scipy Minimize.
    Variables: 
    - x, y for each of the 11 hexagons (22 vars)
    - rotation angle for each of the 11 hexagons (11 vars)
    - outer center x, y (2 vars)
    - outer rotation (1 var)
    Total: 36 vars.
    
    Objective: Minimize outer side length S.
    Penalty: Add large penalty if invalid (overlap or out of bounds).
    """
    
    def objective(params):
        if time.time() > budget:
            return 1000.0
            
        # Unpack params
        # 0-21: x, y for 11 hexes
        # 22-32: rotations for 11 hexes
        # 33: outer cx
        # 34: outer cy
        # 35: outer rot
        
        inr = []
        for i in range(N):
            x = params[i*2]
            y = params[i*2+1]
            a = params[22+i]
            inr.append([x, y, a])
            
        oc = (params[33], params[34])
        rot_out = params[35]
        
        # Compute required outer side
        s_req = _min_outer_side(inr, oc, rot_out)
        
        # Check validity
        # To make it faster, we can do a quick check.
        # Full validity check is expensive.
        # We will use a penalty approach.
        
        # 1. Containment check (approximate via projection)
        # If s_req is calculated, the hexagon defined by s_req contains the projections.
        # But we need to ensure the *actual* geometry is valid.
        
        # Let's do a fast validity check.
        # If it's invalid, return a high penalty.
        
        # Fast check: 
        # Generate outer hex
        outer_pts = _hex_vertices(oc[0], oc[1], s_req, rot_out)
        polys = [_inner_hex(x, y, a) for (x, y, a) in inr]
        
        # Containment
        for p in polys:
            # Check if any vertex is outside
            # For convex polygons, if all vertices are inside, it's inside.
            for pt in p:
                if not _point_in_poly(pt, outer_pts):
                    # Estimate violation
                    return s_req + 10.0
                    
        # Overlap
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                if _poly_overlap(polys[i], polys[j]):
                    return s_req + 10.0
                    
        return s_req

    # Initial parameters
    params = np.zeros(36)
    for i in range(N):
        params[i*2] = inner[i][0]
        params[i*2+1] = inner[i][1]
        params[22+i] = inner[i][2]
    params[33] = outer_c[0]
    params[34] = outer_c[1]
    params[35] = rot
    
    # Bounds
    # Positions: [-10, 10]
    # Rotations: [0, 360]
    # Outer center: [-10, 10]
    # Outer rot: [0, 360]
    bounds = [(-10, 10)] * 22 + [(0, 360)] * 11 + [(-10, 10)] * 2 + [(0, 360)]
    
    # Optimization
    best_x = params
    best_f = objective(params)
    
    # Use Powell for robustness with bounds and non-smooth penalties
    try:
        res = minimize(objective, params, method='Powell', bounds=bounds, 
                       options={'maxiter': 50, 'disp': False, 'maxfev': 200})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
        
    # Try Nelder-Mead on the best result for finer tuning
    if time.time() < budget - 10:
        try:
            # Nelder-Mead doesn't support bounds well, so we rely on Powell's result
            # or just run it if we are confident.
            # Let's stick with Powell's result for safety.
            pass
        except Exception:
            pass
            
    # Extract best
    inr_best = []
    for i in range(N):
        inr_best.append([best_x[i*2], best_x[i*2+1], best_x[22+i]])
    oc_best = (best_x[33], best_x[34])
    rot_best = best_x[35]
    
    s_best = _min_outer_side(inr_best, oc_best, rot_best)
    
    # Final validation
    if _valid(inr_best, oc_best, s_best, rot_best):
        return inr_best, oc_best, s_best, rot_best
    else:
        # Return original if optimization failed validity
        s_orig = _min_outer_side(inner, outer_c, rot)
        if _valid(inner, outer_c, s_orig, rot):
            return inner, outer_c, s_orig, rot
        return None

def construct():
    best_inner = None
    best_oc = None
    best_s = float('inf')
    best_rot = 0.0
    
    budget = START + DEADLINE
    
    rng = random.Random(42)
    
    # Strategy:
    # 1. Generate several initial "flower" configs with perturbations.
    # 2. Optimize each using Scipy.
    # 3. Keep the best valid result.
    
    n_starts = 10
    
    for i in range(n_starts):
        if time.time() > budget - 5:
            break
            
        # Perturb initial config
        # Start with a standard flower
        inner = _generate_initial_config(rng)
        
        # Add more random perturbation for later starts
        if i > 0:
            for h in inner:
                h[0] += rng.gauss(0, 0.2)
                h[1] += rng.gauss(0, 0.2)
                h[2] += rng.gauss(0, 10)
                
        oc = (0.0, 0.0)
        rot = 0.0
        
        # Calculate initial tight side
        s_init = _min_outer_side(inner, oc, rot)
        
        # Optimize
        res = _optimize_config(inner, oc, rot, rng, budget)
        
        if res:
            inr, oc_res, s_res, rot_res = res
            if s_res < best_s:
                best_s = s_res
                best_inner = inr
                best_oc = oc_res
                best_rot = rot_res
                
        # If we have a good result, we can try to refine it further with a local search
        # but the budget is tight.
        
    if best_inner is None:
        # Fallback to a known valid configuration (hexagonal seed)
        inner = []
        inner.append([0.0, 0.0, 0.0])
        d = math.sqrt(3.0)
        for k in range(6):
            a = math.radians(30 + k * 60)
            inner.append([d * math.cos(a), d * math.sin(a), 0.0])
        d2 = 2 * d
        for k in range(4):
            a = math.radians(30 + k * 60)
            inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
            
        oc = (0.0, 0.0)
        rot = 0.0
        s = _min_outer_side(inner, oc, rot)
        if _valid(inner, oc, s, rot):
            return inner, (oc[0], oc[1]), s, rot
        else:
            # Should not happen, but increase s
            s += 0.1
            return inner, (oc[0], oc[1]), s, rot
            
    # Final tightness check
    s_tight = _min_outer_side(best_inner, best_oc, best_rot)
    if _valid(best_inner, best_oc, s_tight, best_rot):
        return best_inner, (best_oc[0], best_oc[1]), s_tight, best_rot
    else:
        # If tightness check fails (due to numerical precision or slight overlap),
        # expand slightly
        s_safe = best_s
        for _ in range(10):
            s_safe += 0.001
            if _valid(best_inner, best_oc, s_safe, best_rot):
                return best_inner, (best_oc[0], best_oc[1]), s_safe, best_rot
        return best_inner, (best_oc[0], best_oc[1]), s_safe, best_rot

if __name__ == "__main__":
    construct()
