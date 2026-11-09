"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
import random

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _point_in_polygon(pt, poly):
    """Check if point is inside a convex polygon using cross products."""
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -1e-9:
            return False
    return True

def _containment_penalty(inner_verts, outer_verts):
    """Penalty for inner hexagon not being fully inside outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _check_valid(params, outer_side, outer_center, outer_angle, tol=1e-6):
    """Check if configuration is valid (no overlaps, all contained)."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen < tol and disj_pen < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions

    def make_params_from_positions(positions, angles=None):
        """Create parameter vector from positions and optional angles."""
        if angles is None:
            angles = [0.0] * len(positions)
        params = []
        for i, (x, y) in enumerate(positions):
            params.extend([float(x), float(y), float(angles[i])])
        return np.array(params)

    # Generate a few high-quality initial configurations
    configs = []
    
    # 3 rows x 4 cols hex grid
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 4 rows x 3 cols hex grid
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 2 rows x 6 cols
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    best_score = float('inf')
    best_result = None

    outer_angles = [0.0, 30.0]

    for outer_angle in outer_angles:
        outer_center = (0.0, 0.0)
        
        for config in configs:
            if time.time() > deadline - 10.0:
                break
            
            init_params = make_params_from_positions(config)
            
            # Objective function: Minimize the enclosing hexagon side length
            # We use a smooth approximation of the max projection (soft-max)
            # plus a penalty for overlaps.
            
            def objective(p):
                # 1. Calculate max projection for enclosing side
                a = np.radians(outer_angle)
                out_norms = np.array([[np.cos(a + np.radians(60*i + 30)), np.sin(a + np.radians(60*i + 30))] for i in range(6)])
                
                max_proj = 0.0
                inner_verts_list = []
                for i in range(N):
                    cx, cy, ang = p[3*i], p[3*i+1], p[3*i+2]
                    iv = _hex_verts(cx, cy, 1.0, ang)
                    inner_verts_list.append(iv)
                    for pt in iv:
                        rel = pt - np.array(outer_center)
                        for on in out_norms:
                            proj = np.dot(rel, on)
                            if proj > max_proj:
                                max_proj = proj
                
                # Side length corresponding to max_proj
                # Apothem = side * sqrt(3)/2 => side = 2 * apothem / sqrt(3)
                side_est = 2.0 * max_proj / np.sqrt(3.0)
                
                # 2. Calculate overlap penalty
                disj_pen = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
                        if depth > 0:
                            # Quadratic penalty to encourage separation
                            disj_pen += depth ** 2
                
                # Weight the overlap penalty heavily to ensure validity
                # If overlap is non-zero, the cost increases significantly
                return side_est + 10.0 * disj_pen

            try:
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_params = result.x
                opt_val = result.fun
                
                # Validate the result
                # Compute actual enclosing side
                a = np.radians(outer_angle)
                out_norms = np.array([[np.cos(a + np.radians(60*i + 30)), np.sin(a + np.radians(60*i + 30))] for i in range(6)])
                max_proj = 0.0
                inner_verts_list = []
                for i in range(N):
                    cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
                    iv = _hex_verts(cx, cy, 1.0, ang)
                    inner_verts_list.append(iv)
                    for pt in iv:
                        rel = pt - np.array(outer_center)
                        for on in out_norms:
                            proj = np.dot(rel, on)
                            if proj > max_proj:
                                max_proj = proj
                actual_side = 2.0 * max_proj / np.sqrt(3.0)
                
                # Check disjointness
                disj_pen = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
                        if depth > 0:
                            disj_pen += depth
                
                # Check containment (should be implied by side calculation, but verify)
                outer_verts = _hex_verts(outer_center[0], outer_center[1], actual_side, outer_angle)
                cont_pen = sum(_containment_penalty(iv, outer_verts) for iv in inner_verts_list)
                
                if disj_pen < 1e-5 and cont_pen < 1e-5:
                    if actual_side < best_score:
                        best_score = actual_side
                        best_result = (opt_params.copy(), outer_center, actual_side, outer_angle)
            except Exception:
                continue

    # Fallback to a valid configuration if optimization failed or didn't find a valid one
    if best_result is None:
        # Use the 3x4 lattice, which is known to be valid
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        inner = []
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        
        # Calculate the side length for this configuration
        outer_angle = 0.0
        outer_center = (0.0, 0.0)
        a = np.radians(outer_angle)
        out_norms = np.array([[np.cos(a + np.radians(60*i + 30)), np.sin(a + np.radians(60*i + 30))] for i in range(6)])
        max_proj = 0.0
        for (x, y) in (arr - centroid):
            iv = _hex_verts(x, y, 1.0, 0.0)
            for pt in iv:
                rel = pt - np.array(outer_center)
                for on in out_norms:
                    proj = np.dot(rel, on)
                    if proj > max_proj:
                        max_proj = proj
        side = 2.0 * max_proj / np.sqrt(3.0)
        
        return inner, (0.0, 0.0), float(side), float(outer_angle)

    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
