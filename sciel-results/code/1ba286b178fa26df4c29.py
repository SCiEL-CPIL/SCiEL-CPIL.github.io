"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon with center (cx, cy), side length, and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def polygon_area_signed(verts):
    """Signed area of a polygon given as array of vertices."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i, 0] * verts[j, 1] - verts[j, 0] * verts[i, 1]
    return area / 2.0

def point_in_polygon(pt, polygon):
    """Check if a point is inside a convex polygon using cross products."""
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        edge = polygon[j] - polygon[i]
        to_pt = pt - polygon[i]
        cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
        if cross < -1e-9:
            return False
    return True

def min_distance_point_to_polygon(pt, polygon):
    """Minimum distance from point to convex polygon. Returns 0 if inside."""
    if point_in_polygon(pt, polygon):
        return 0.0
    
    min_dist = float('inf')
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        # Distance to segment ab
        ab = b - a
        ap = pt - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        dist = np.linalg.norm(pt - closest)
        min_dist = min(min_dist, dist)
    return min_dist

def min_clearance(inner_verts, outer_verts):
    """Minimum clearance between inner hexagon vertices and outer hexagon boundary."""
    # For each inner vertex, find min distance to outer polygon
    min_clr = float('inf')
    for v in inner_verts:
        d = min_distance_point_to_polygon(v, outer_verts)
        min_clr = min(min_clr, d)
    return min_clr

def check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside outer hexagon and don't overlap each other."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check each inner hexagon is inside outer
    for pos in inner_positions:
        cx, cy, ang = pos
        inner_verts = hex_vertices(cx, cy, 1.0, ang)
        for v in inner_verts:
            if not point_in_polygon(v, outer_verts):
                return False
    
    # Check no overlaps between inner hexagons
    # For simplicity, check center-to-center distance
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            xi, yi, ai = inner_positions[i]
            xj, yj, aj = inner_positions[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            # For unit hexagons, minimum distance to avoid overlap is 2*sqrt(3)/2 = sqrt(3) ~ 1.732
            # Actually for hexagons, the inradius is sqrt(3)/2, so center distance must be > 2*inradius = sqrt(3)
            if dist < math.sqrt(3) - 1e-6:
                return False
    
    return True

def objective(params, outer_center, outer_angle):
    """Objective function to minimize outer side length."""
    # params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_side]
    n_params = len(params)
    outer_side = params[-1]
    if outer_side < 1.0:
        return 1e6
    
    # Extract inner positions
    inner_positions = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        inner_positions.append((x, y, a))
    
    # Check if all inner hexagons are inside outer
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    penalty = 0.0
    for pos in inner_positions:
        cx, cy, ang = pos
        inner_verts = hex_vertices(cx, cy, 1.0, ang)
        for v in inner_verts:
            if not point_in_polygon(v, outer_verts):
                # Penalty for being outside
                d = min_distance_point_to_polygon(v, outer_verts)
                penalty += d * d
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ai = inner_positions[i]
            xj, yj, aj = inner_positions[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            min_dist = math.sqrt(3)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return outer_side + 100.0 * penalty

def construct():
    """Construct packing of N unit hexagons in a regular hexagon, minimizing outer side."""
    # Initial guess: hexagonal grid pattern
    # For N=11, use a compact arrangement
    
    # Start with a known good arrangement: hexagonal lattice
    # Center + 6 around + 4 more
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # 6 neighbors at distance sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3.0
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        positions.append((x, y, 0.0))
    # 4 more at distance 2*sqrt(3) in some directions
    for i in range(4):
        angle = i * math.pi / 2.0
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        positions.append((x, y, 0.0))
    
    # Trim to N
    positions = positions[:N]
    
    # Compute initial outer side
    max_r = max(math.sqrt(x**2 + y**2) for x, y, a in positions)
    outer_side = max_r + 1.0  # rough estimate
    
    # Set up optimization
    x0 = []
    for x, y, a in positions:
        x0.extend([x, y, a])
    x0.append(outer_side)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    bounds.append((1.0, 20.0))
    
    # Try multiple starts
    best_result = None
    best_obj = float('inf')
    
    np.random.seed(42)
    
    for start_idx in range(5):
        if start_idx == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            for i in range(len(x_start) - 1):
                x_start[i] += np.random.uniform(-1, 1)
            x_start[-1] = outer_side * (1 + np.random.uniform(-0.1, 0.1))
        
        try:
            res = minimize(
                objective,
                x_start,
                args=(outer_center, outer_angle),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_result = res
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to initial guess
        final_positions = positions
        final_outer_side = outer_side
    else:
        params = best_result.x
        final_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            final_positions.append([x, y, a])
        final_outer_side = params[-1]
    
    # Verify and adjust if needed
    if not check_validity(final_positions, outer_center, final_outer_side, outer_angle):
        # Increase outer side until valid
        for _ in range(100):
            final_outer_side *= 1.05
            if check_validity(final_positions, outer_center, final_outer_side, outer_angle):
                break
    
    return final_positions, outer_center, final_outer_side, outer_angle

# EVOLVE-BLOCK-END
