import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon with given center, angle, and side length."""
    angle = np.radians(angle_deg)
    # Vertices of a unit regular hexagon (flat-top orientation at angle=0)
    # For a hexagon with side s, vertices are at distance s from center
    # at angles 0, 60, 120, 180, 240, 300 degrees (for pointy-top)
    # Let's use pointy-top: vertices at 30, 90, 150, 210, 270, 330 + rotation
    angles = np.radians(30 + np.arange(6) * 60) + angle
    verts = np.column_stack([np.cos(angles), np.sin(angles)]) * side
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def polygon_area_signed(verts):
    """Signed area of polygon using shoelace formula."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return 0.5 * area

def point_in_convex_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = None
    for i in range(n):
        j = (i + 1) % n
        edge = poly[j] - poly[i]
        cross = edge[0] * (pt[1] - poly[i][1]) - edge[1] * (pt[0] - poly[i][0])
        if sign is None:
            sign = cross >= 0
        elif (cross >= 0) != sign:
            return False
    return True

def polygons_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    polys = [verts1, verts2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Check for gap
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def min_separating_axis_distance(verts1, verts2):
    """Find minimum distance between two convex polygons (negative if overlapping)."""
    min_dist = np.inf
    polys = [verts1, verts2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            normal = np.array([-edge[1], edge[0]]) / edge_len
            # Project both polygons
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Distance along this axis
            if max1 < min2:
                dist = min2 - max1
            elif max2 < min1:
                dist = min1 - max2
            else:
                dist = 0.0  # Overlapping on this axis
            min_dist = min(min_dist, dist)
    return -min_dist  # Negative if overlapping

def compute_objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, outer_side, outer_angle]
    Total: 12*3 + 4 = 40 parameters
    """
    inner_params = params[:36]
    cx, cy, outer_side, outer_angle = params[36], params[37], params[38], params[39]
    
    # Ensure outer_side is positive
    outer_side = max(outer_side, 0.5)
    
    # Get inner hexagon vertices
    inner_verts = []
    for i in range(N):
        x = inner_params[i*3]
        y = inner_params[i*3+1]
        a = inner_params[i*3+2]
        v = hex_vertices(x, y, a, 1.0)
        inner_verts.append(v)
    
    # Get outer hexagon vertices
    outer_v = hex_vertices(cx, cy, outer_angle, outer_side)
    
    # Penalty for inner hexagons not contained in outer
    contain_penalty = 0.0
    for v in inner_verts:
        for pt in v:
            if not point_in_convex_polygon(pt, outer_v):
                # Compute distance to polygon boundary
                # Simplified: just penalize
                contain_penalty += 1.0
    
    # Penalty for overlapping inner hexagons
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = min_separating_axis_distance(inner_verts[i], inner_verts[j])
            if dist < 0:
                overlap_penalty += (-dist) ** 2
    
    # Objective: minimize outer_side + penalties
    obj = outer_side + 100.0 * contain_penalty + 100.0 * overlap_penalty
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: hexagonal arrangement
    # Center at origin, 6 in first ring, 5 in second ring
    init_inner = []
    # Center
    init_inner.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        init_inner.append([x, y, 0.0])
    # Second ring: 5 more hexagons
    # Place them at the "gaps" of the second ring
    for k in range(5):
        angle = np.radians(60 * k + 30)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        init_inner.append([x, y, 0.0])
    
    # Initial outer hexagon: large enough to contain all
    init_params = []
    for x, y, a in init_inner:
        init_params.extend([x, y, a])
    init_params.extend([0.0, 0.0, 10.0, 0.0])
    
    best_params = np.array(init_params)
    best_obj = compute_objective(best_params)
    
    # Try COBYLA optimization
    try:
        result = minimize(
            compute_objective,
            best_params,
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 1.0}
        )
        if time.time() > deadline - 5.0:
            pass  # Use best so far
        else:
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
    except:
        pass
    
    if time.time() > deadline - 5.0:
        # Return best found
        inner = []
        for i in range(N):
            inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
        return inner, (best_params[36], best_params[37]), best_params[38], best_params[39]
    
    # Try multiple restarts with different initial arrangements
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial solution
        np.random.seed(restart)
        perturbed = best_params.copy()
        perturbed[:36] += np.random.normal(0, 0.5, 36)
        perturbed[38] = max(perturbed[38], 5.0)  # Ensure outer side is reasonable
        
        try:
            result = minimize(
                compute_objective,
                perturbed,
                method='COBYLA',
                options={'maxiter': 300, 'rhobeg': 0.5}
            )
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
        except:
            pass
    
    # Extract final solution
    inner = []
    for i in range(N):
        inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    
    cx = best_params[36]
    cy = best_params[37]
    outer_side = best_params[38]
    outer_angle = best_params[39]
    
    # Final polish: try to reduce outer_side slightly
    if time.time() < deadline - 5.0:
        for _ in range(10):
            new_side = outer_side * 0.99
            test_params = best_params.copy()
            test_params[38] = new_side
            test_obj = compute_objective(test_params)
            if test_obj < best_obj:
                best_params = test_params
                outer_side = new_side
                best_obj = test_obj
            else:
                break
    
    inner = []
    for i in range(N):
        inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    
    return inner, (best_params[36], best_params[37]), best_params[38], best_params[39]

