import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of [x, y, angle_deg]
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    # Outer hexagon vertices
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    # For each inner hexagon, check if all vertices are inside the outer hexagon
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            # Check if point is inside convex polygon using cross products
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                # Vector from p1 to p2
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                # Vector from p1 to point
                px = vx - p1[0]
                py = vy - p1[1]
                # Cross product
                cross = ex * py - ey * px
                # For a convex polygon with vertices in counter-clockwise order,
                # the point is inside if all cross products have the same sign
                # We assume the hexagon is oriented such that interior is to the left
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice - use tighter, more symmetric arrangement
    # Center hexagon + 6 around it + 4 more in a compact cluster
    # Axial coordinates for hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    scale = math.sqrt(3)
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Center the configuration
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    for p in inner:
        p[0] -= cx
        p[1] -= cy
    
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False
    
    def update_best(inner_cfg, S, ocx, ocy, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S:
            best_S = S
            best_inner = [list(x) for x in inner_cfg]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True

    # 3. Local Search / Optimization
    # We will try different outer angles and refine the inner positions.
    # The objective is to minimize S.
    
    # Try a few outer angles. 0 and 30 are the most likely candidates for optimal packing.
    outer_angles = [0.0, 30.0, 60.0]
    
    for oangle in outer_angles:
        if time.time() > deadline:
            break
            
        cur_inner = [list(x) for x in inner]
        cur_S = compute_required_side(cur_inner, (0,0), oangle)
        
        # Ensure it fits (it should, since we just computed S to fit it)
        # But we need to ensure no overlaps. The initial lattice config has no overlaps.
        if not check_all_disjoint(cur_inner):
            continue
            
        if cur_S < best_S:
            update_best(cur_inner, cur_S, 0.0, 0.0, oangle)
            
        # Refine: Local search to reduce S further
        # Strategy: Move hexes slightly towards the center or rotate the whole cluster?
        # Actually, since S is determined by the max projection, we want to pull the "extreme" hexes in.
        # A simple gradient-free approach: random perturbations that reduce S.
        
        rng = random.Random(42)
        steps = 0
        max_steps = 5000
        step_size = 0.05
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            # Pick a random hex
            idx = rng.randrange(N)
            old_x, old_y, old_a = cur_inner[idx]
            
            # Perturb position
            new_x = old_x + rng.gauss(0, step_size)
            new_y = old_y + rng.gauss(0, step_size)
            new_a = old_a # Keep angle fixed for now, or randomize slightly?
            # For hex packing, orientation matters. Let's allow small angle changes.
            if rng.random() < 0.1:
                new_a = old_a + rng.gauss(0, 5.0)
            
            cur_inner[idx] = [new_x, new_y, new_a]
            
            # Check disjointness
            if not check_all_disjoint(cur_inner):
                cur_inner[idx] = [old_x, old_y, old_a]
                continue
            
            new_S = compute_required_side(cur_inner, (0,0), oangle)
            
            if new_S < cur_S:
                cur_S = new_S
                if cur_S < best_S:
                    update_best(cur_inner, cur_S, 0.0, 0.0, oangle)
            else:
                # Revert
                cur_inner[idx] = [old_x, old_y, old_a]
                
            # Occasionally try to shift the whole cluster to center of mass?
            # The compute_required_side function already centers on (0,0) implicitly?
            # No, it uses outer_center (0,0). So we should keep the cluster centered at 0,0.
            # Let's re-center after every successful move to prevent drift.
            if new_S <= cur_S: # If we accepted
                 # Actually, we reverted if it was worse. So if we are here, we accepted.
                 # Wait, the logic above reverts if new_S >= cur_S.
                 # So if we reach here, new_S < cur_S.
                 # Re-center
                 cx = sum(p[0] for p in cur_inner) / N
                 cy = sum(p[1] for p in cur_inner) / N
                 for p in cur_inner:
                     p[0] -= cx
                     p[1] -= cy
                 # Recalculate S after re-centering? 
                 # Re-centering doesn't change relative positions, so S relative to centroid is invariant.
                 # But compute_required_side uses (0,0) as outer center.
                 # So yes, S is invariant to translation of the whole cluster if outer center is fixed at 0,0?
                 # No! If we move the cluster, the projections change.
                 # But we want the cluster to be centered at 0,0 to minimize the max projection?
                 # Yes, for a symmetric shape, centering minimizes the bounding box.
                 # So re-centering is a good move.
                 cur_S = compute_required_side(cur_inner, (0,0), oangle)
                 if cur_S < best_S:
                     update_best(cur_inner, cur_S, 0.0, 0.0, oangle)

    # Fallback if no valid configuration was found (should not happen with the lattice start)
    if not best_valid or best_inner is None:
        # Simple fallback: just return the initial lattice config
        S = compute_required_side(inner, (0,0), 0.0)
        return inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
