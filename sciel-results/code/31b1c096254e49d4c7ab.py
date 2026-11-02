"""Improved program: place 11 unit hexagons in a regular hexagon, minimizing outer side."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # Base vertices for side=1, angle=0
    # Vertices are at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Actually, standard hexagon with vertex at 0 deg:
    # (1, 0)
    # (0.5, sqrt(3)/2)
    # (-0.5, sqrt(3)/2)
    # (-1, 0)
    # (-0.5, -sqrt(3)/2)
    # (0.5, -sqrt(3)/2)
    base_x = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    base_y = np.array([0.0, math.sqrt(3)/2, math.sqrt(3)/2, 0.0, -math.sqrt(3)/2, -math.sqrt(3)/2])
    
    # Rotate
    rx = base_x * cos_a - base_y * sin_a
    ry = base_x * sin_a + base_y * cos_a
    
    # Scale and Translate
    rx = rx * side + cx
    ry = ry * side + cy
    
    return np.column_stack((rx, ry))

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side and rotation."""
    # Transform point to hexagon's local frame
    angle = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local frame, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
    # A regular hexagon with side s has vertices at distance s from center
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with vertex at angle 0, the half-plane constraints are:
    # x * cos(k*60) + y * sin(k*60) <= s * cos(30) = s * sqrt(3)/2 for k=0..5
    # Actually, for a regular hexagon with circumradius = side length s:
    # The apothem (inradius) = s * sqrt(3)/2
    # The constraint is: for each edge, the distance from center to edge is apothem
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    apothem = side * math.sqrt(3.0) / 2.0
    for k in range(6):
        normal_angle = math.radians(30 + k * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > apothem + 1e-10:
            return False
    return True

def hexagon_distance(p1, p2):
    """Minimum distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def hexagons_disjoint(v1, v2):
    """Check if two hexagons (as arrays of 6 vertices) are disjoint."""
    # Fast AABB check
    min_x1, max_x1 = v1[:, 0].min(), v1[:, 0].max()
    min_y1, max_y1 = v1[:, 1].min(), v1[:, 1].max()
    min_x2, max_x2 = v2[:, 0].min(), v2[:, 0].max()
    min_y2, max_y2 = v2[:, 1].min(), v2[:, 1].max()
    
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return True
    
    # Helper: point in convex polygon (using cross products)
    def point_in_convex_polygon(px, py, poly):
        # poly is Nx2
        # Check if point is to the same side of all edges
        sign = 0
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % 6]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True
    
    # Check vertices
    for i in range(6):
        if point_in_convex_polygon(v1[i, 0], v1[i, 1], v2):
            return False
        if point_in_convex_polygon(v2[i, 0], v2[i, 1], v1):
            return False
    
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def orient(a, b, c):
            val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else -1
        
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    
    return True

def compute_outer_side(inner_positions):
    """Given inner hexagon positions [(x,y,angle)], compute minimum outer hexagon side.
    The outer hexagon is centered at origin with 0 rotation for simplicity.
    We need to find the minimum side S such that all inner hexagons fit inside."""
    
    # For each inner hexagon, compute the maximum "extent" in each of the 6 directions
    # The outer hexagon (centered at origin, side S, angle 0) has the constraint:
    # For each edge normal direction k (at angles 30, 90, 150, 210, 270, 330),
    # the maximum projection of any inner hexagon vertex onto that normal must be <= S * sqrt(3)/2
    
    max_projections = [0.0] * 6
    
    for (x, y, angle_deg) in inner_positions:
        verts = hexagon_vertices(x, y, 1.0, angle_deg)
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for v in verts:
                proj = v[0] * nx + v[1] * ny
                if proj > max_projections[k]:
                    max_projections[k] = proj
    
    # The required side S must satisfy: S * sqrt(3)/2 >= max(max_projections)
    # So S >= 2 * max(max_projections) / sqrt(3)
    max_proj = max(max_projections)
    if max_proj <= 0:
        return 0.0
    return 2.0 * max_proj / math.sqrt(3.0)

def objective(params):
    """Objective function: outer side length. params = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN]"""
    n = len(params) // 3
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(n)]
    
    # Precompute vertices for all hexagons
    verts = [hexagon_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    
    # Check disjointness - if any two overlap, return large penalty
    for i in range(n):
        for j in range(i+1, n):
            if not hexagons_disjoint(verts[i], verts[j]):
                # Simple penalty: return a large value
                # To make it smooth-ish, we could estimate overlap depth, but for now,
                # a large constant is safer for DE to avoid getting stuck in infeasible regions
                # if the penalty is too small.
                return 1000.0
    
    return compute_outer_side(inner)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal packing
    hex_positions = []
    hex_positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = math.radians(30 + k * 60)
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        hex_positions.append((x, y, 0.0))
    for k in range(5):
        angle = math.radians(0 + k * 72)
        x = 4 * math.sqrt(3) * math.cos(angle)
        y = 4 * math.sqrt(3) * math.sin(angle)
        hex_positions.append((x, y, 0.0))
    
    x0 = []
    for (x, y, a) in hex_positions:
        x0.extend([x, y, a])
    
    best_params = x0[:]
    best_val = objective(x0)
    
    # Try differential evolution first for global search (Reduced budget)
    bounds = [(-10, 10), (-10, 10), (0, 360)] * N
    
    try:
        # Reduced maxiter and popsize to ensure it finishes quickly
        result = differential_evolution(
            objective, bounds, maxiter=10, popsize=5, 
            tol=0.01, seed=42, polish=False,
            init='sobol'
        )
        elapsed = time.time() - start_time
        if elapsed < deadline:
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x[:]
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x[:]
        except Exception:
            pass
    
    # Final polish if time remains
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x[:]
        except Exception:
            pass
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2] % 360.0
        inner.append([x, y, a])
    
    # Compute final outer side
    outer_side = compute_outer_side(inner)
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-END
