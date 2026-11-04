import numpy as np
import time
from scipy.optimize import minimize
import math

N = 12
DEADLINE = 145.0  # Safety margin below 150s

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """
    Calculate vertices of a regular hexagon.
    rot_deg is rotation in degrees.
    """
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    """
    Calculate outward normal vectors for the 6 edges of a regular hexagon.
    The edges are between vertex k and k+1.
    The normal to the edge connecting angle theta1 and theta2 (60 deg apart)
    is at angle (theta1 + theta2)/2 + 90? No.
    For a hexagon centered at origin with vertices at 0, 60, 120...
    Edge 0 connects (1,0) and (0.5, sqrt(3)/2).
    Midpoint: (0.75, sqrt(3)/4).
    Normal direction: angle 30 degrees.
    So normals are at 30, 90, 150, 210, 270, 330 degrees.
    """
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap(v1, v2, n1, n2):
    """
    Returns the minimum separation distance between two hexagons.
    Positive if separated, negative if overlapping (penetration depth).
    Uses Separating Axis Theorem (SAT).
    v1, v2: (6, 2) arrays of vertices.
    n1, n2: (6, 2) arrays of edge normals.
    """
    min_overlap = float('inf')
    
    # Check axes from hex 1
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            # Separating axis found
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    # Check axes from hex 2
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return -min_overlap

def _outer_hex_side(points, outer_rot_deg):
    """
    Calculate the minimum side length of a regular hexagon with rotation
    outer_rot_deg that encloses the given points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Project points onto each normal
    proj = points @ normals.T
    
    # For a regular hexagon centered at (0,0) with side L, the distance
    # from center to edge (apothem) is L * sqrt(3) / 2.
    # So max projection must be <= L * sqrt(3) / 2.
    # L = 2 * max_proj / sqrt(3)
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, tol=1e-9):
    """
    Check if all inner hexagons are disjoint.
    inner: (12, 3) array of [x, y, rot_deg]
    """
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
        
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap(verts[i], verts[j], norms[i], norms[j])
            if d < -tol:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    """
    Generate a grid of hexagons.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad if necessary (should not happen for valid grid sizes)
        while len(pts) < N:
            pts.append([0, 0, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    """
    Generate a ring configuration: 1 center, 6 inner ring, 5 outer.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
    
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_random_cluster(scale=3.0, seed=None):
    """
    Generate a random cluster of hexagons.
    """
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3) * scale
    pts[:, 2] = np.random.uniform(0, 60, N)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _objective(p, t0, deadline, beta=10.0):
    """
    Objective function: Minimize outer side length + penalty for overlaps.
    p: [x1, y1, r1, ..., x12, y12, r12, ox, oy, oang]
    """
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    # Calculate overlaps
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlaps.append(-d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        # Use Log-Sum-Exp for smooth maximum
        # We want to penalize the largest overlap heavily, but smoothly.
        # soft_max(x) = log(sum(exp(x)))
        # To make it steeper, scale by 1/beta
        scaled = p_arr / beta
        max_s = np.max(scaled)
        if max_s > 500:
            penalty = np.sum(p_arr**2)
        else:
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            penalty = beta * lse
    else:
        penalty = 0.0

    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side(all_pts, oang)
    
    return L + 100.0 * penalty

def _polish(params, t0, deadline):
    """
    Multi-stage polishing to ensure high precision and valid disjointness.
    """
    if time.time() - t0 > deadline - 15:
        return params
    
    current_params = params.copy()
    
    # Stage 1: Coarse optimization with lower penalty weight (beta=10)
    # Allows larger movements to escape local minima of the penalty.
    try:
        res = minimize(
            _objective,
            current_params,
            args=(t0, deadline - 15, 10.0),
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'maxfun': 1000
            }
        )
        if res.fun < _objective(current_params, t0, deadline - 15, 10.0):
            current_params = res.x
    except Exception:
        pass
    
    # Stage 2: Medium optimization with higher penalty weight (beta=1)
    if time.time() - t0 < deadline - 10:
        try:
            res = minimize(
                _objective,
                current_params,
                args=(t0, deadline - 10, 1.0),
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 2000
                }
            )
            if res.fun < _objective(current_params, t0, deadline - 10, 1.0):
                current_params = res.x
        except Exception:
            pass
            
    # Stage 3: Fine optimization with very high penalty weight (beta=0.1)
    if time.time() - t0 < deadline - 5:
        try:
            res = minimize(
                _objective,
                current_params,
                args=(t0, deadline - 5, 0.1),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 3000
                }
            )
            if res.fun < _objective(current_params, t0, deadline - 5, 0.1):
                current_params = res.x
        except Exception:
            pass
            
    # Stage 4: Nelder-Mead for final local refinement if time permits
    if time.time() - t0 < deadline - 2:
        try:
            res = minimize(
                _objective,
                current_params,
                args=(t0, deadline - 2, 0.1),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res.fun < _objective(current_params, t0, deadline - 2, 0.1):
                current_params = res.x
        except Exception:
            pass
            
    return current_params

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # List of initial configurations
    # We want to try different topologies: Grids, Rings, and Randoms.
    # Note: Rotations are in degrees.
    
    configs = []
    
    # 1. Standard Hex Grids (aligned)
    configs.append(_make_hex_grid(4, 3, gap=0.1))
    configs.append(_make_hex_grid(3, 4, gap=0.1))
    configs.append(_make_hex_grid(2, 6, gap=0.1))
    configs.append(_make_hex_grid(6, 2, gap=0.1))
    
    # 2. Rotated Hex Grids
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=0.1))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=0.1))
    
    # 3. Ring Configuration
    configs.append(_make_ring(gap=0.1))
    
    # 4. Random Clusters
    np.random.seed(42)
    for i in range(10):
        configs.append(_make_random_cluster(scale=4.0, seed=42+i))
        
    # 5. Specific "Zig-Zag" or "Brick" patterns might be good, 
    # but random starts with good optimization should find them.
    
    best_L = float('inf')
    best_inner = None
    best_center = None
    best_angle = None
    best_params = None
    
    # Phase 1: Evaluate and Polish each configuration
    for idx, cfg in enumerate(configs):
        if time_left() < 15:
            break
        
        # Initialize outer hexagon parameters
        # Center is mean of inner hexagons
        ox = np.mean(cfg[:, 0])
        oy = np.mean(cfg[:, 1])
        # Try a few outer rotations
        for oang in [0.0, 30.0]:
            if time_left() < 10:
                break
            
            # Initial params: inner (36) + ox, oy, oang (3) = 39
            init_params = np.concatenate((cfg.flatten(), [ox, oy, oang]))
            
            # Run polish
            polished = _polish(init_params, t0, time_left() - 2)
            
            # Extract results
            inner_opt = polished[:36].reshape(12, 3)
            ox_opt = polished[36]
            oy_opt = polished[37]
            oang_opt = polished[38]
            
            # Check validity
            if not _check_disjoint(inner_opt):
                # If invalid, try to fix with a very high penalty run? 
                # Or just skip. The optimizer should have handled it.
                # Let's try one more aggressive step if invalid.
                if time_left() > 5:
                     try:
                         res = minimize(
                             _objective,
                             polished,
                             args=(t0, time_left() - 2, 0.01),
                             method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                         )
                         polished = res.x
                         inner_opt = polished[:36].reshape(12, 3)
                         ox_opt = polished[36]
                         oy_opt = polished[37]
                         oang_opt = polished[38]
                     except:
                         pass
            
            if _check_disjoint(inner_opt):
                all_pts_opt = np.vstack([_hex_vertices(inner_opt[i,0], inner_opt[i,1], 1.0, inner_opt[i,2]) for i in range(12)])
                all_pts_opt[:, 0] -= ox_opt
                all_pts_opt[:, 1] -= oy_opt
                L_opt = _outer_hex_side(all_pts_opt, oang_opt)
                
                if L_opt < best_L:
                    best_L = L_opt
                    best_inner = inner_opt.copy()
                    best_center = (ox_opt, oy_opt)
                    best_angle = oang_opt
                    best_params = polished.copy()
    
    # Fallback if no valid solution found (should not happen)
    if best_inner is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    inner = []
    for i in range(12):
        x = best_inner[i, 0]
        y = best_inner[i, 1]
        r = best_inner[i, 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_angle = float(best_angle)
    
    # Final safety check
    if not _check_disjoint(inner):
        # This should be rare if polish worked. 
        # If it fails, we return the best valid one we found, or a fallback.
        # Since we only updated best_inner if _check_disjoint passed, this shouldn't happen.
        pass

    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side(all_pts, outer_angle)
    
    # Add a tiny epsilon to ensure strict containment if the evaluator is strict
    outer_side *= 1.00001
    
    return inner, outer_center, outer_side, outer_angle
