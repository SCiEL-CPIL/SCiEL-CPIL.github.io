"""Improved program: place unit regular hexagons inside a regular hexagon.
Strategy:
1. Generate candidate positions on a honeycomb-like grid inside a candidate outer hexagon.
2. Use a greedy / local search to find a feasible packing of N unit hexagons.
3. Optimize the outer side length via binary search on the side length, checking feasibility.
4. Include a deadline guard to return the best result before the 150s limit.
"""

import math
import time
import random
import numpy as np

N = 11
DEADLINE_SECONDS = 145.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return vertices

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside the regular hexagon centered at (cx, cy) with given side and rotation."""
    # Transform point to local coordinates of the hexagon
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    # Rotate point by -angle to align hexagon with axes
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, the vertices are at distance s from center
    # The hexagon is the intersection of 6 half-planes
    # In the aligned frame, the hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The half-planes are defined by the normals pointing outward
    # For a hexagon with "flat top" (vertices at 0, 60, ...), the supporting lines are at distance s*cos(30) = s*sqrt(3)/2 from center
    # Actually, let's use the standard: a regular hexagon with side length s has inradius r = s * sqrt(3)/2
    # The hexagon is the set of points where |x| <= r and |y| <= r in a rotated coordinate system? No.
    
    # Let me use a more robust method: check against each edge
    verts = hexagon_vertices(0, 0, side, 0)  # aligned hexagon at origin
    # For each edge, check if point is on the correct side
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Edge vector
        ex = x2 - x1
        ey = y2 - y1
        # Normal pointing outward (for CCW polygon, right normal is outward)
        # For a regular hexagon with vertices in CCW order, the outward normal can be computed
        # Let's compute the cross product to check side
        # Vector from v1 to point
        px_local = lx - x1
        py_local = ly - y1
        # Cross product of edge vector and vector to point
        cross = ex * py_local - ey * px_local
        # For a CCW polygon, interior is to the left of each edge, so cross should be >= 0 for interior
        # But we need to be careful about the orientation
        # Let's just check if the point is on the same side as the center (0,0)
        # Vector from v1 to center
        cx_local = 0 - x1
        cy_local = 0 - y1
        cross_center = ex * cy_local - ey * cx_local
        # If cross and cross_center have different signs, point is outside
        if cross * cross_center < 0:
            return False
    return True

def hexagon_overlap(h1_center, h1_angle, h2_center, h2_angle, side=1.0, eps=1e-9):
    """Check if two unit hexagons overlap. Using a simple distance-based heuristic first, then more precise if needed."""
    dx = h1_center[0] - h2_center[1] if False else h1_center[0] - h2_center[0]
    dy = h1_center[1] - h2_center[1]
    dist = math.sqrt(dx*dx + dy*dy)
    
    # Minimum distance between centers of two non-overlapping unit hexagons depends on orientation
    # The minimum distance is 2 * inradius = sqrt(3) ≈ 1.732 for aligned hexagons
    # But with rotation, it can be less. The minimum possible distance is 2 * apothem = sqrt(3)
    # Actually, the inradius (apothem) of a unit hexagon is sqrt(3)/2, so two hexagons can have centers as close as sqrt(3) apart without overlapping.
    # This is a necessary but not sufficient condition. For a quick check:
    if dist < math.sqrt(3) - eps:
        return True
    if dist > 2.0 + eps:  # definitely no overlap if centers are more than 2*radius apart (radius = 1)
        return False
    
    # For a more precise check, we would need to do polygon intersection
    # For now, use a conservative check: if dist < sqrt(3), assume overlap
    # This is conservative but may reject some valid configurations
    # Let's use a slightly better bound: the minimum distance for non-overlap is sqrt(3)
    # Actually, let me implement a proper polygon intersection check
    return _hexagons_intersect(h1_center, h1_angle, h2_center, h2_angle, side)

def _hexagons_intersect(c1, a1, c2, a2, side=1.0):
    """Check if two regular hexagons intersect using separating axis theorem."""
    verts1 = hexagon_vertices(c1[0], c1[1], side, a1)
    verts2 = hexagon_vertices(c2[0], c2[1], side, a2)
    
    # Get all edges from both polygons
    edges = []
    for i in range(6):
        x1, y1 = verts1[i]
        x2, y2 = verts1[(i+1)%6]
        edges.append((x1, y1, x2, y2))
    for i in range(6):
        x1, y1 = verts2[i]
        x2, y2 = verts2[(i+1)%6]
        edges.append((x1, y1, x2, y2))
    
    # For each edge, compute the normal and project both polygons onto it
    for x1, y1, x2, y2 in edges:
        ex = x2 - x1
        ey = y2 - y1
        # Normal (perpendicular to edge)
        nx = -ey
        ny = ex
        # Normalize
        length = math.sqrt(nx*nx + ny*ny)
        if length < 1e-12:
            continue
        nx /= length
        ny /= length
        
        # Project vertices of hex1
        proj1 = [vx*nx + vy*ny for vx, vy in verts1]
        # Project vertices of hex2
        proj2 = [vx*nx + vy*ny for vx, vy in verts2]
        
        min1 = min(proj1)
        max1 = max(proj1)
        min2 = min(proj2)
        max2 = max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def all_hexagons_disjoint(positions, angles, side=1.0):
    """Check if all hexagons are pairwise disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if hexagon_overlap(positions[i], angles[i], positions[j], angles[j], side):
                return False
    return True

def all_hexagons_inside(positions, angles, outer_center, outer_side, outer_angle, side=1.0):
    """Check if all inner hexagons are inside the outer hexagon."""
    for i in range(len(positions)):
        # Check if all vertices of inner hexagon are inside outer hexagon
        verts = hexagon_vertices(positions[i][0], positions[i][1], side, angles[i])
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    return True

def is_feasible(positions, angles, outer_center, outer_side, outer_angle, side=1.0):
    """Check if the configuration is feasible: all inside and disjoint."""
    if not all_hexagons_inside(positions, angles, outer_center, outer_side, outer_angle, side):
        return False
    if not all_hexagons_disjoint(positions, angles, side):
        return False
    return True

def greedy_pack(outer_center, outer_side, outer_angle, num_hex, max_iterations=10000):
    """Try to greedily pack num_hex unit hexagons inside the outer hexagon."""
    positions = []
    angles = [0.0] * num_hex
    
    # Generate candidate positions on a grid
    # The outer hexagon has inradius = outer_side * sqrt(3)/2
    inradius = outer_side * math.sqrt(3) / 2
    
    # Create a grid of candidate positions
    step = 0.5  # fine grid
    candidates = []
    for x in np.arange(-inradius, inradius + step, step):
        for y in np.arange(-inradius, inradius + step, step):
            px = outer_center[0] + x
            py = outer_center[1] + y
            # Check if this position is valid (center of a unit hexagon can be here)
            # The hexagon of side 1 has circumradius 1, so the center must be at least 1 away from the boundary
            # More precisely, all vertices must be inside
            verts = hexagon_vertices(px, py, 1.0, 0.0)
            valid = True
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    valid = False
                    break
            if valid:
                candidates.append((px, py))
    
    # Sort candidates by distance from center (prefer central positions)
    candidates.sort(key=lambda p: (p[0]-outer_center[0])**2 + (p[1]-outer_center[1])**2)
    
    # Greedily add hexagons
    for pos in candidates:
        if len(positions) >= num_hex:
            break
        # Check if this position is disjoint from all existing
        conflict = False
        for existing_pos in positions:
            if hexagon_overlap(pos, 0.0, existing_pos, 0.0, 1.0):
                conflict = True
                break
        if not conflict:
            positions.append(pos)
    
    if len(positions) < num_hex:
        return None, None
    
    # Verify
    if is_feasible(positions, angles, outer_center, outer_side, outer_angle):
        return positions, angles
    
    return None, None

def local_search(positions, angles, outer_center, outer_side, outer_angle, max_iters=1000):
    """Improve the packing via local search: try small perturbations."""
    positions = [p[:] for p in positions]
    angles = angles[:]
    
    current_cost = 0.0
    # Cost: sum of overlaps (should be 0 for feasible)
    # We want to minimize overlap
    
    def compute_overlap_penalty():
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = math.sqrt(3)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        # Also penalize going outside
        for i in range(n):
            verts = hexagon_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    penalty += 10.0
        return penalty
    
    current_penalty = compute_overlap_penalty()
    
    for it in range(max_iters):
        if current_penalty < 1e-6:
            break
        
        # Try perturbing a random hexagon
        idx = random.randint(0, len(positions)-1)
        old_pos = positions[idx][:]
        old_angle = angles[idx]
        
        # Random perturbation
        positions[idx][0] += random.gauss(0, 0.1)
        positions[idx][1] += random.gauss(0, 0.1)
        angles[idx] = (old_angle + random.gauss(0, 5.0)) % 360
        
        new_penalty = compute_overlap_penalty()
        
        if new_penalty < current_penalty:
            current_penalty = new_penalty
        else:
            # Revert
            positions[idx] = old_pos
            angles[idx] = old_angle
    
    if current_penalty < 1e-6:
        return positions, angles, True
    
    return positions, angles, False

def construct():
    start_time = time.time()
    
    best_result = None
    best_side = float('inf')
    
    # We'll try different outer side lengths and find the minimum feasible one
    # The theoretical minimum for N=11 unit hexagons is around 3-4
    # Let's search from 2.5 to 10
    
    low = 2.0
    high = 10.0
    
    # First, find a feasible side length
    # Try a range of side lengths
    sides_to_try = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]
    
    feasible_side = None
    for side in sides_to_try:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 5:
            break
        
        positions, angles = greedy_pack((0.0, 0.0), side, 0.0, N)
        if positions is not None:
            feasible_side = side
            break
    
    if feasible_side is None:
        # If no feasible side found, return a large side
        # Use a simple grid layout
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Now binary search for the minimum feasible side
    low = feasible_side - 0.5
    high = feasible_side
    
    # Ensure low is positive
    if low < 1.5:
        low = 1.5
    
    best_positions = None
    best_angles = None
    
    for iteration in range(20):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10:
            break
        
        mid = (low + high) / 2
        positions, angles = greedy_pack((0.0, 0.0), mid, 0.0, N)
        
        if positions is not None:
            # Try to improve with local search
            positions, angles, success = local_search(positions, angles, (0.0, 0.0), mid, 0.0, max_iters=500)
            if is_feasible(positions, angles, (0.0, 0.0), mid, 0.0):
                high = mid
                best_positions = positions
                best_angles = angles
                best_side = mid
            else:
                low = mid
        else:
            low = mid
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        # Return best found
        if best_positions is not None:
            inner = [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)]
            return inner, (0.0, 0.0), best_side, 0.0
        else:
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Try to further optimize with more local search iterations
    if best_positions is not None:
        positions, angles, success = local_search(best_positions, best_angles, (0.0, 0.0), best_side, 0.0, max_iters=2000)
        if is_feasible(positions, angles, (0.0, 0.0), best_side, 0.0):
            best_positions = positions
            best_angles = angles
    
    if best_positions is not None:
        inner = [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)]
        return inner, (0.0, 0.0), best_side, 0.0
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
