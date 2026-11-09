"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses SAT-based collision detection and L-BFGS-B optimization with strict time budget.
"""
N = 12

import numpy as np
import time
from scipy.optimize import minimize
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # 20s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i+1)%n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0]*rel[1] - edge[1]*rel[0]
        if cross < -1e-9:
            return False
    return True

def _containment_penalty(centers, angles, S, outer_angle_deg=0.0):
    """Penalty for inner hexagons extending outside the outer hexagon."""
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    penalty = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        for pt in iv:
            if not _point_in_polygon(pt, outer_verts):
                # Compute distance to outside
                min_dist = 1e9
                for j in range(6):
                    v1 = outer_verts[j]
                    v2 = outer_verts[(j+1)%6]
                    edge = v2 - v1
                    edge_len = np.linalg.norm(edge)
                    if edge_len < 1e-12:
                        continue
                    edge_unit = edge / edge_len
                    t = np.dot(pt - v1, edge_unit)
                    t = max(0, min(edge_len, t))
                    closest = v1 + t * edge_unit
                    d = np.linalg.norm(pt - closest)
                    min_dist = min(min_dist, d)
                penalty += min_dist ** 2 * 100.0
    return penalty

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons
    using the Separating Axis Theorem. Returns 0 if already separated.
    """
    def project(poly, axis):
        dots = poly @ axis
        return np.min(dots), np.max(dots)
    
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            edge = poly[(i+1)%len(poly)] - poly[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    min_overlap = 1e9
    for axis in axes:
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0  # Separating axis found
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return min_overlap

def _total_penalty(params, S, outer_angle_deg=0.0):
    """
    Total penalty function: containment + overlap penalties.
    params: 3*N array [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    # Containment penalty
    pen = _containment_penalty(centers, angles, S, outer_angle_deg)
    
    # Overlap penalty between inner hexagons - only check nearby pairs
    for i in range(N):
        vi = _hex_verts(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i+1, N):
            vj = _hex_verts(centers[j, 0], centers[j, 1], 1.0, angles[j])
            # Quick bounding box check before SAT
            if (np.max(vi[:,0]) < np.min(vj[:,0]) or np.min(vi[:,0]) > np.max(vj[:,0]) or
                np.max(vi[:,1]) < np.min(vj[:,1]) or np.min(vi[:,1]) > np.max(vj[:,1])):
                continue
            depth = _sat_overlap_depth(vi, vj)
            if depth > 0:
                pen += depth ** 2 * 1000.0
    
    return pen

def _min_enclosing_hex_side(centers, angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            a = np.radians(ang)
            vs_angles = a + np.radians(60 * np.arange(6))
            vs = np.array([np.cos(vs_angles), np.sin(vs_angles)]).T
            dots = vs @ np.array([dx, dy])
            h = np.max(dots)  # side=1
            needed = cx * dx + cy * dy + h
            if needed > max_ext:
                max_ext = needed
    
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_valid(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """Check if configuration is valid: all inside and disjoint."""
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    
    # Check containment
    for i in range(N):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        for pt in iv:
            if not _point_in_polygon(pt, outer_verts):
                return False
    
    # Check disjointness
    for i in range(N):
        vi = _hex_verts(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i+1, N):
            vj = _hex_verts(centers[j, 0], centers[j, 1], 1.0, angles[j])
            if _sat_overlap_depth(vi, vj) > tol:
                return False
    
    return True

def _objective(params, S_target, outer_angle_deg=0.0):
    """
    Objective: minimize S + penalty.
    Fast version with bounding box pre-checks.
    """
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    # Compute actual enclosing side
    S_actual = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    
    # Penalty for containment and overlap using S_actual
    pen = _containment_penalty(centers, angles, S_actual, outer_angle_deg)
    
    # Overlap penalty with bounding box pre-check
    for i in range(N):
        vi = _hex_verts(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i+1, N):
            vj = _hex_verts(centers[j, 0], centers[j, 1], 1.0, angles[j])
            # Quick bounding box check
            if (np.max(vi[:,0]) < np.min(vj[:,0]) or np.min(vi[:,0]) > np.max(vj[:,0]) or
                np.max(vi[:,1]) < np.min(vj[:,1]) or np.min(vi[:,1]) > np.max(vj[:,1])):
                continue
            depth = _sat_overlap_depth(vi, vj)
            if depth > 0:
                pen += depth ** 2 * 1000.0
    
    return S_actual + pen

def _make_initial_configs():
    """Generate a small number of initial configurations."""
    configs = []
    
    # Config 1: 4x3 grid with small gap
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    scale = 1.01
    inner = []
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            inner.append([float(pos[0]), float(pos[1]), 0.0])
    cx_mean = np.mean([p[0] for p in inner])
    cy_mean = np.mean([p[1] for p in inner])
    inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
    configs.append(np.array([p for pt in inner for p in pt]).flatten())
    
    # Config 2: Slightly different scale
    scale2 = 1.02
    inner2 = []
    for i in range(4):
        for j in range(3):
            pos = scale2 * (i * u + j * v)
            inner2.append([float(pos[0]), float(pos[1]), 0.0])
    cx_mean2 = np.mean([p[0] for p in inner2])
    cy_mean2 = np.mean([p[1] for p in inner2])
    inner2 = [[p[0]-cx_mean2, p[1]-cy_mean2, p[2]] for p in inner2]
    configs.append(np.array([p for pt in inner2 for p in pt]).flatten())
    
    return configs

def _fallback_grid():
    """Return a valid grid configuration as fallback."""
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    scale = 1.02
    inner = []
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            inner.append([float(pos[0]), float(pos[1]), 0.0])
    cx_mean = np.mean([p[0] for p in inner])
    cy_mean = np.mean([p[1] for p in inner])
    inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
    centers = np.array([[p[0], p[1]] for p in inner])
    angles = np.array([0.0]*N)
    S_val = _min_enclosing_hex_side(centers, angles, 0.0)
    return inner, (0.0, 0.0), float(S_val), 0.0

def construct():
    best_S = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    configs = _make_initial_configs()
    
    # Only try 1 outer angle to save time
    outer_angles = [0.0]
    
    for outer_angle in outer_angles:
        if _time_left() < 20.0:
            break
        
        for cfg_idx, init_params in enumerate(configs):
            if _time_left() < 10.0:
                break
            
            # Check initial config score
            S_init = _min_enclosing_hex_side(
                np.array([[init_params[3*i], init_params[3*i+1]] for i in range(N)]),
                np.array([init_params[3*i+2] for i in range(N)]),
                outer_angle
            )
            pen_init = _total_penalty(init_params, S_init, outer_angle)
            total_init = S_init + pen_init
            
            if total_init < best_S:
                best_S = total_init
                best_params = init_params.copy()
                best_outer_angle = outer_angle
            
            # Only optimize if we have enough time
            if _time_left() < 15.0:
                continue
            
            # Optimize positions and angles with reduced iterations
            def obj_func(p):
                return _objective(p, None, outer_angle)
            
            bounds = []
            for i in range(N):
                bounds.append((-5.0, 5.0))
                bounds.append((-5.0, 5.0))
                bounds.append((0.0, 60.0))
            
            try:
                result = minimize(
                    obj_func,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4, 'maxls': 10}
                )
                
                if result.success or result.fun < best_S:
                    S_val = _min_enclosing_hex_side(
                        np.array([[result.x[3*i], result.x[3*i+1]] for i in range(N)]),
                        np.array([result.x[3*i+2] for i in range(N)]),
                        outer_angle
                    )
                    pen = _total_penalty(result.x, S_val, outer_angle)
                    total = S_val + pen
                    
                    if total < best_S:
                        best_S = total
                        best_params = result.x.copy()
                        best_outer_angle = outer_angle
                        
            except Exception:
                continue
    
    # If optimization didn't find anything good, fall back to simple grid
    if best_params is None:
        return _fallback_grid()
    
    # Extract final result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    centers = np.array([[p[0], p[1]] for p in inner])
    angles = np.array([p[2] for p in inner])
    S_val = _min_enclosing_hex_side(centers, angles, best_outer_angle)
    
    # Verify validity with tolerance
    if not _check_valid(centers, angles, S_val * 1.001, best_outer_angle, tol=1e-5):
        return _fallback_grid()
    
    return inner, (0.0, 0.0), float(S_val), float(best_outer_angle)
