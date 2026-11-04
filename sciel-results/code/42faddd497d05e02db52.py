import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
EPS = 1e-9
START = time.time()
DEADLINE = 150.0

def _hex_vertices_vec(centers, angles, s=1.0):
    """
    Vectorized computation of hexagon vertices.
    centers: (N, 2)
    angles: (N,) in radians
    returns: (N, 6, 2)
    """
    # Base angles for hexagon vertices
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    # (N, 6)
    all_angles = angles[:, np.newaxis] + base_angles[np.newaxis, :]
    cos_a = np.cos(all_angles)
    sin_a = np.sin(all_angles)
    
    # (N, 6, 2)
    verts = np.zeros((len(centers), 6, 2))
    verts[:, :, 0] = centers[:, 0:1] + s * cos_a
    verts[:, :, 1] = centers[:, 1:2] + s * sin_a
    return verts

def _min_outer_side_vec(centers, angles, outer_rot_rad):
    """
    Compute min outer side length for given inner hexagons.
    centers: (N, 2)
    angles: (N,)
    outer_rot_rad: float
    """
    verts = _hex_vertices_vec(centers, angles)
    # Flattened vertices: (N*6, 2)
    flat_verts = verts.reshape(-1, 2)
    
    # Normals for outer hexagon
    # Outer hexagon vertices are at angles rot + 0, 60, ...
    # Normals are at rot + 30, 90, ...
    normal_angles = outer_rot_rad + np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
    
    # Projection of all vertices onto normals: (6, N*6)
    projs = normals @ flat_verts.T
    
    # Max projection for each normal
    max_projs = np.max(projs, axis=1)
    
    # The outer hexagon must contain these projections.
    # The distance from center to side is max_proj.
    # Apothem A = max_proj.
    # Side S = 2 * A / sqrt(3).
    max_proj = np.max(max_projs)
    return 2.0 * max_proj / math.sqrt(3.0)

def _overlap_penalty_vec(centers, angles, weight=100.0):
    """
    Compute penalty for overlapping inner hexagons.
    Uses separating axis theorem on the 12 normals (6 from each hex).
    """
    n = len(centers)
    penalty = 0.0
    if n < 2:
        return 0.0
    
    # Precompute vertices for all hexagons
    verts = _hex_vertices_vec(centers, angles) # (N, 6, 2)
    
    for i in range(n):
        v_i = verts[i]
        # Normals for hex i
        # Edge 0 is between vertex 0 and 1.
        # Normal is perpendicular to edge.
        # We can compute normals for each hexagon.
        # Or just use the 6 normals relative to the hexagon's rotation.
        # The normals of a hexagon rotated by theta are at theta + 30, 90, ...
        normals_i = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + angles[i]
        n_i = np.column_stack([np.cos(normals_i), np.sin(normals_i)]) # (6, 2)
        
        for j in range(i + 1, n):
            v_j = verts[j]
            normals_j = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + angles[j]
            n_j = np.column_stack([np.cos(normals_j), np.sin(normals_j)]) # (6, 2)
            
            # All 12 normals
            all_normals = np.vstack([n_i, n_j]) # (12, 2)
            
            # Project v_i and v_j onto all normals
            # v_i is (6, 2), all_normals is (12, 2)
            # proj_i is (12, 6)
            proj_i = all_normals @ v_i.T
            proj_j = all_normals @ v_j.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            # Gap for each axis
            # If max_i < min_j, gap is min_j - max_i
            # If max_j < min_i, gap is min_i - max_j
            # Else gap is 0 (overlap)
            
            gaps = np.zeros(12)
            # Case 1: i is to the "left" of j
            gap1 = min_j - max_i
            # Case 2: j is to the "left" of i
            gap2 = min_i - max_j
            
            # If gap1 > 0, they are separated in that direction.
            # We want the minimum separation.
            # If any gap is > 0, they are separated.
            # The minimum distance is the minimum of positive gaps.
            # If all gaps are <= 0, they overlap.
            
            # We want a penalty for overlap.
            # If they overlap, the "penetration depth" is the minimum amount to move to separate.
            # This is -min(gaps) if min(gaps) < 0?
            # Actually, if they overlap, all gaps are <= 0.
            # The penetration depth is -min(gaps) (where min is the most negative).
            # Wait, if gap is -0.1, it means they overlap by 0.1 in that direction.
            # The minimum separation distance is 0.
            # The penalty should be based on the overlap.
            
            # Let's use the sum of squared negative gaps as a penalty?
            # Or just the max overlap?
            # A common approach: Penalty = sum( max(0, -gap)^2 )
            # But if they overlap, multiple axes will have negative gaps.
            
            # Simpler: Check if they are separated.
            # If separated, penalty 0.
            # If not, penalty = (min_gap)^2 ? No, min_gap is the most negative.
            # Let's use the sum of squared negative gaps.
            
            neg_gaps = np.minimum(0, gap1) + np.minimum(0, gap2) # This is wrong logic
            
            # Correct logic:
            # For each axis, the separation is max(0, min_j - max_i, min_i - max_j)
            # If separation is 0 for all axes, they overlap.
            # We want a penalty proportional to the overlap.
            # Let's compute the minimum separation.
            # sep = min( max(0, min_j - max_i), max(0, min_i - max_j) ) for each axis?
            # No, sep_axis = max(0, min_j - max_i, min_i - max_j)
            # If sep_axis > 0 for any axis, they are separated.
            # The minimum distance is min(sep_axis).
            # If min(sep_axis) == 0, they overlap.
            # Penalty = (overlap_depth)^2.
            # Overlap depth is not directly given by SAT gaps in a simple way if they are deeply embedded.
            # However, for convex polygons, if they overlap, the minimum distance is 0.
            # We can use a penalty that is non-zero when they overlap.
            # A simple penalty: sum of squared distances of vertices of i to edges of j?
            # Too slow.
            
            # Let's use the "penetration depth" approximation.
            # If they overlap, the most negative gap is a measure of penetration.
            # Let pen = min(gap1, gap2) over all axes.
            # If pen < 0, they overlap.
            # Penalty += pen^2.
            
            all_gaps = np.concatenate([gap1, gap2])
            min_gap = np.min(all_gaps)
            if min_gap < 0:
                penalty += (min_gap) ** 2
                
    return weight * penalty

def _objective(params, outer_rot_rad):
    """
    Objective: Outer Side + Overlap Penalty
    params: (3*N,) [x1, y1, a1, x2, y2, a2, ...]
    """
    n = N
    centers = params[0:2*n:3].reshape(n, 2)
    angles = params[2:3*n:3]
    
    side = _min_outer_side_vec(centers, angles, outer_rot_rad)
    pen = _overlap_penalty_vec(centers, angles, weight=50.0)
    
    return side + pen

def _hexagonal_seed():
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
    return inner, (0.0, 0.0), 0.0, 0.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _valid(inner, outer_c, s, rot):
    # Reuse the strict validity check from the current program
    # Convert to lists of tuples for the existing functions
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _hex_vertices(cx, cy, s, rot):
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

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
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _min_outer_side(inner, outer_c, rot):
    # Non-vectorized version for final check
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
    return 2.0 * max_proj / math.sqrt(3.0)

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    rng = random.Random(42)
    
    # Seed 1: Hexagonal
    seed_inner, seed_oc, seed_s, seed_rot = _hexagonal_seed()
    seed_s = _min_outer_side(seed_inner, seed_oc, seed_rot)
    
    # Initialize parameters
    # We fix outer center at (0,0) and optimize outer rotation?
    # Or fix outer rotation at 0 and optimize inner?
    # Let's optimize inner positions and angles, and outer rotation.
    # Params: x1..x11, y1..y11, a1..a11, outer_rot
    # Total 34 params.
    
    init_params = np.zeros(3 * N + 1)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
    init_params[-1] = 0.0 # outer rot
    
    best_inner = seed_inner
    best_side = seed_s
    best_rot = 0.0
    
    # Optimization 1: Nelder-Mead
    try:
        if time.time() < budget:
            # We need to pass outer_rot to objective.
            # Since outer_rot is part of params, we can't pass it as arg easily if it changes.
            # We can wrap the objective.
            
            def obj_wrapper(params):
                outer_rot = params[-1]
                inner_params = params[:-1]
                return _objective(inner_params, outer_rot)
                
            res = minimize(obj_wrapper, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_side:
                best_rot = res.x[-1]
                inner_params = res.x[:-1]
                best_inner = []
                for i in range(N):
                    best_inner.append([inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2]])
                best_side = _min_outer_side(best_inner, (0,0), best_rot)
                
    except Exception as e:
        pass
        
    # Optimization 2: L-BFGS-B
    try:
        if time.time() < budget - 10:
            def obj_wrapper_lbfgs(params):
                outer_rot = params[-1]
                inner_params = params[:-1]
                return _objective(inner_params, outer_rot)
                
            res2 = minimize(obj_wrapper_lbfgs, init_params, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            
            if res2.fun < best_side:
                best_rot = res2.x[-1]
                inner_params = res2.x[:-1]
                best_inner = []
                for i in range(N):
                    best_inner.append([inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2]])
                best_side = _min_outer_side(best_inner, (0,0), best_rot)
                
    except Exception as e:
        pass
        
    # Final validation and tightening
    if best_inner:
        if _valid(best_inner, (0,0), best_side, best_rot):
            # Try to tighten side
            s_tight = _min_outer_side(best_inner, (0,0), best_rot)
            if s_tight < best_side:
                best_side = s_tight
            best = (best_inner, (0,0), best_side, best_rot)
        else:
            # If invalid, try to repair or fall back to seed
            # Simple repair: scale down? No, just use seed if valid.
            if _valid(seed_inner, seed_oc, seed_s, seed_rot):
                best = (seed_inner, seed_oc, seed_s, seed_rot)
            else:
                best = (seed_inner, seed_oc, seed_s, seed_rot) # Should be valid by construction

    if best is None:
        best = (seed_inner, seed_oc, seed_s, seed_rot)
        
    return best[0], (best[1][0], best[1][1]), best[2], best[3]

