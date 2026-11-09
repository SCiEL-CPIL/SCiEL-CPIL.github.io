"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Check if two hexagons overlap using Separating Axis Theorem (SAT)
    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def is_disjoint(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
        # Outer hexagon is regular. Its boundary is defined by 6 lines.
        # For a regular hexagon of side S, the apothem (distance from center to midpoint of side) is A = S * SQRT3 / 2.
        # The condition for a point (x,y) to be inside a hexagon centered at origin with rotation theta is:
        # max over 6 directions of dot((x,y), n_i) <= A
        # where n_i are unit normals to the sides.
        
        # We want to find minimum S such that all vertices of all inner hexagons are inside.
        # This is equivalent to finding the maximum projection of all vertices onto the 6 normal directions,
        # and setting A = max_proj. Then S = 2 * A / SQRT3.
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for k in range(6):
                    rad = math.radians(outer_angle_deg + k * 60)
                    # Normal to side k. If side is at angle alpha, normal is alpha.
                    # Wait, standard hexagon with flat top?
                    # Let's define outer hexagon normals.
                    # If outer hexagon has a vertex at angle 0, its side normals are at 30, 90, 150...
                    # Let's just project onto 6 directions separated by 60 deg.
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # max_proj is the required apothem A.
        # S = 2 * A / SQRT3
        return 2.0 * max_proj / SQRT3

    # Base configuration: 1 center + 6 ring + 4 more? 
    # Let's use a compact cluster of 11 hexagons.
    # A regular hexagon of side 2 has 1 + 6 + 12 = 19 hexagons.
    # A regular hexagon of side 1 has 7 hexagons.
    # We need 11.
    # Let's take the 7 hexagons (side 1 cluster) and add 4 more attached to the sides.
    # Axial coordinates (q, r)
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # These are 7.
    # Add 4 more: (2,0), (2,-1), (1,1), (-1,2) ? Let's check distances.
    # Distance between (1,0) and (2,0) is 1. OK.
    # Distance between (0,1) and (1,1) is 1. OK.
    # Distance between (-1,0) and (-1,2)? No, (-1,0) to (-1,1) is 1. (-1,1) to (-1,2) is 1.
    # Let's pick a shape that is roughly hexagonal.
    # Coords:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (2,-1), (1,1), (-1,2) -> Wait, (-1,2) is far.
    # Let's try:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (0,2), (-2,2), (-2,0) ?
    # Let's just use a list of known valid close packings or generate one.
    # Actually, the "hexagon of side 2 minus 8 corners" is 19-8=11.
    # Or simply: Center + 6 neighbors + 4 of the 12 second-ring neighbors.
    # Let's use axial coords:
    coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2) # Adding 4 from second ring
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        # Standard hex grid to cartesian
        # x = SQRT3 * (q + r/2)
        # y = 1.5 * r
        x = SQRT3 * scale * (q + r / 2.0)
        y = 1.5 * scale * r
        return x, y

    # We will optimize:
    # 1. Global rotation of the cluster (theta)
    # 2. Global rotation of the outer hexagon (phi)
    # 3. Scale factor of the lattice (s) - allows stretching to fit better?
    #    Actually, if we stretch the lattice, hexagons might overlap.
    #    Minimum distance between centers in hex lattice is SQRT3 * s * something?
    #    Distance between (0,0) and (1,0) is SQRT3 * s.
    #    Hexagons of side 1 have "radius" 1 (vertex to center).
    #    So centers must be at least 2 apart? No, they can interlock.
    #    In a perfect hex lattice with spacing 1 (center-to-center), they touch.
    #    Here, axial (1,0) distance is SQRT3 * scale.
    #    For unit hexagons (side 1), the distance between centers in a standard packing is SQRT3?
    #    Let's check. Vertices at distance 1.
    #    If two hexagons touch side-to-side, center distance is 2 * apothem = SQRT3.
    #    So if scale=1, distance is SQRT3. They touch.
    #    If scale < 1, they overlap.
    #    So we must have scale >= 1.
    #    However, by rotating the cluster relative to the outer hexagon, we can often reduce the bounding box.
    
    # Objective function for Nelder-Mead
    def objective(params):
        theta = params[0]  # Cluster rotation
        phi = params[1]    # Outer hex rotation
        scale = params[2]  # Lattice scale (>= 1.0)
        
        if scale < 1.0:
            scale = 1.0
            
        # Generate positions
        positions = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, scale)
            # Rotate around origin
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            positions.append((rx, ry))
            
        # Center the cluster
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(p[0]-cx, p[1]-cy) for p in positions]
        
        # Get vertices
        hexagons = []
        for i in range(N):
            px, py = positions[i]
            # Inner hexagons have the same rotation as the cluster? 
            # The task says "each inner hexagon may translate and rotate".
            # In a lattice packing, they usually share the same rotation.
            # If we allow individual rotations, the search space is huge.
            # Let's assume they share the cluster rotation for now.
            # This is a valid subset of the solution space.
            angle_deg = math.degrees(theta)
            verts = get_hex_vertices(px, py, angle_deg)
            hexagons.append(verts)
            
        # Check overlaps (Hard constraint)
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j]):
                    return 1e6 # Penalty
            
        # Calculate outer side
        side = calc_outer_side(hexagons, math.degrees(phi))
        return side

    # Initial guess
    # Try a few starting points
    best_side = float('inf')
    best_params = None
    
    # Starting points:
    # 1. Standard orientation
    # 2. Rotated 30 deg
    # 3. Random perturbations
    
    starts = [
        [0.0, 0.0, 1.0],
        [math.pi/6, 0.0, 1.0],
        [math.pi/3, 0.0, 1.0],
        [0.1, 0.1, 1.0],
        [0.2, 0.2, 1.0]
    ]
    
    for s in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception as e:
            pass

    # If Nelder-Mead fails or is slow, we might have a fallback.
    # Let's also try a simple grid search over angles to be safe, or just use the best NM result.
    
    if best_params is None:
        # Fallback to initial lattice
        best_params = [0.0, 0.0, 1.0]
        best_side = objective(best_params)

    # Extract final configuration
    theta, phi, scale = best_params
    
    # Generate final positions
    positions = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, scale)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        positions.append((rx, ry))
        
    # Center
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    positions = [(p[0]-cx, p[1]-cy) for p in positions]
    
    # Build inner list
    inner_list = []
    hexagons = []
    for i in range(N):
        px, py = positions[i]
        angle_deg = math.degrees(theta)
        inner_list.append([px, py, angle_deg])
        verts = get_hex_vertices(px, py, angle_deg)
        hexagons.append(verts)
        
    # Final validation
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # This shouldn't happen if penalty worked, but just in case
        # Return a valid but potentially suboptimal configuration (standard lattice)
        positions = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, 1.0)
            positions.append((x, y))
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(p[0]-cx, p[1]-cy) for p in positions]
        inner_list = []
        hexagons = []
        for i in range(N):
            px, py = positions[i]
            inner_list.append([px, py, 0.0])
            verts = get_hex_vertices(px, py, 0.0)
            hexagons.append(verts)
        theta = 0.0
        phi = 0.0
        side = calc_outer_side(hexagons, 0.0)
    else:
        side = best_side
        # Refine phi slightly? The NM optimized it.
        # The returned side is the score.
        
    # The interface requires: (inner, center, side, angle)
    # inner: rows (x, y, angle_degrees)
    # center: (x, y) of outer hexagon
    # side: outer side length
    # angle: outer rotation
    
    return (inner_list, (0.0, 0.0), side, math.degrees(phi))

# EVOLVE-BLOCK-END
