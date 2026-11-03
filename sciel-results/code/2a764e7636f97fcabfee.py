"""
Unit regular hexagons inside a regular hexagon n=12
Optimized packing using honeycomb arrangements and local refinement.
"""
import numpy as np
import time
import math

N = 12

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def polygons_overlap_sat(poly1, poly2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    polys = [poly1, poly2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def check_no_overlap(inner_positions, tol=1e-6):
    """Check that no two inner hexagons overlap."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner_positions[i]
            xj, yj, aj = inner_positions[j]
            # Quick distance check
            dx = xi - xj
            dy = yi - yj
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 2.5:  # Maximum possible overlap distance for unit hexagons
                continue
            vi = hexagon_vertices(xi, yi, 1.0, ai)
            vj = hexagon_vertices(xj, yj, 1.0, aj)
            if polygons_overlap_sat(vi, vj, tol):
                return False
    return True

def compute_min_outer_side(inner_positions, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    cx, cy = outer_center
    max_extent = 0.0
    for (ix, iy, iangle) in inner_positions:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            dx = vx - cx
            dy = vy - cy
            angle_rad = np.radians(outer_angle_deg)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for i in range(6):
                normal_angle = np.radians(60 * i)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                extent = lx * nx + ly * ny
                if extent > max_extent:
                    max_extent = extent
    
    side = 2.0 * max_extent / np.sqrt(3)
    return side

def generate_honeycomb_ring(n_center, n_ring1, n_ring2, d1, d2, ring1_offset=0.0, ring2_offset=0.0):
    """Generate a honeycomb-like arrangement with concentric rings."""
    positions = []
    
    # Center
    if n_center > 0:
        positions.append((0.0, 0.0, 0.0))
    
    # Ring 1
    for i in range(n_ring1):
        angle = np.radians(60 * i + ring1_offset)
        positions.append((d1 * np.cos(angle), d1 * np.sin(angle), 0.0))
    
    # Ring 2
    for i in range(n_ring2):
        angle = np.radians(360.0 / n_ring2 * i + ring2_offset)
        positions.append((d2 * np.cos(angle), d2 * np.sin(angle), 0.0))
    
    return positions

def generate_honeycomb_rows(rows, col_positions, row_spacing, col_spacing):
    """Generate a honeycomb arrangement with rows."""
    positions = []
    for r, cols in enumerate(rows):
        y = r * row_spacing
        offset = (r % 2) * col_spacing / 2
        for c in cols:
            x = (c - (len(cols)-1)/2.0) * col_spacing + offset
            positions.append((x, y, 0.0))
    return positions

def refine_positions(positions, outer_center, outer_angle, max_iters=50, deadline=None):
    """Local refinement of positions to minimize outer hexagon side length."""
    positions = list(positions)
    cx, cy = outer_center
    best_side = compute_min_outer_side(positions, outer_center, outer_angle)
    
    step_size = 0.1
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        for i in range(len(positions)):
            xi, yi, ai = positions[i]
            best_local = (xi, yi, ai)
            best_local_side = best_side
            
            # Try small perturbations
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    new_pos = list(positions)
                    new_pos[i] = (xi + dx, yi + dy, ai)
                    
                    if not check_no_overlap(new_pos):
                        continue
                    
                    side = compute_min_outer_side(new_pos, outer_center, outer_angle)
                    if side < best_local_side - 1e-8:
                        best_local = (xi + dx, yi + dy, ai)
                        best_local_side = side
                        improved = True
            
            positions[i] = best_local
        
        new_side = compute_min_outer_side(positions, outer_center, outer_angle)
        if new_side < best_side:
            best_side = new_side
        
        if not improved:
            break
        step_size *= 0.8
    
    return positions, best_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_result = None
    
    d1 = np.sqrt(3)  # Distance for edge-to-edge contact
    d2 = 2 * np.sqrt(3)
    
    # Generate multiple candidate arrangements
    candidates = []
    
    # 1+6+5 arrangement with various offsets
    for ring1_off in [0.0, 30.0]:
        for ring2_off in [0.0, 30.0, 60.0]:
            pos = generate_honeycomb_ring(1, 6, 5, d1, d2, ring1_off, ring2_off)
            candidates.append(pos)
    
    # 1+6+5 with slightly different d2
    for d2_factor in [1.8, 1.9, 2.0, 2.1]:
        pos = generate_honeycomb_ring(1, 6, 5, d1, d2_factor * d1, 0.0, 30.0)
        candidates.append(pos)
    
    # Row-based arrangements
    # 2-3-3-3-1
    rows1 = [2, 3, 3, 3, 1]
    col_spacing = np.sqrt(3)
    row_spacing = np.sqrt(3) / 2
    pos = generate_honeycomb_rows(rows1, range(max(rows1)), row_spacing, col_spacing)
    candidates.append(pos)
    
    # 3-3-3-3
    rows2 = [3, 3, 3, 3]
    pos = generate_honeycomb_rows(rows2, range(3), row_spacing, col_spacing)
    candidates.append(pos)
    
    # 2-3-3-2-2
    rows3 = [2, 3, 3, 2, 2]
    pos = generate_honeycomb_rows(rows3, range(3), row_spacing, col_spacing)
    candidates.append(pos)
    
    # 4-4-4
    rows4 = [4, 4, 4]
    pos = generate_honeycomb_rows(rows4, range(4), row_spacing, col_spacing)
    candidates.append(pos)
    
    # 3-4-5
    rows5 = [3, 4, 5]
    pos = generate_honeycomb_rows(rows5, range(5), row_spacing, col_spacing)
    candidates.append(pos)
    
    # 5-4-3
    rows6 = [5, 4, 3]
    pos = generate_honeycomb_rows(rows6, range(5), row_spacing, col_spacing)
    candidates.append(pos)
    
    # Evaluate each candidate
    for idx, pos in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Check for overlaps
        if not check_no_overlap(pos):
            continue
        
        # Compute centroid
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # Try several outer rotations
        for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_min_outer_side(pos, (cx, cy), outer_angle)
            if side < best_side:
                best_side = side
                best_result = (list(pos), (cx, cy), side, outer_angle)
        
        # Try local refinement
        if time.time() > deadline:
            break
        
        for outer_angle in [0.0, 30.0]:
            refined_pos, refined_side = refine_positions(
                pos, (cx, cy), outer_angle, 
                max_iters=30, deadline=deadline
            )
            if refined_side < best_side:
                best_side = refined_side
                best_result = (refined_pos, (cx, cy), refined_side, outer_angle)
    
    # If no valid candidate found, return a default
    if best_result is None:
        pos = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            pos.append((5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0))
        return pos, (0.0, 0.0), 15.0, 0.0
    
    pos, center, side, angle = best_result
    return pos, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function above is the main logic)
# EVOLVE-BLOCK-END
