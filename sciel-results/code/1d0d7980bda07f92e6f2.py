"""
Strategy: Discrete Triangular Lattice Search with Greedy Best-Fit Packing.

Instead of continuous optimization, we:
1. Generate a set of candidate positions on a triangular lattice (spacing = 2.0 for side=1 hexagons).
2. Use a greedy algorithm to place the 12 hexagons, always choosing the position that minimizes the bounding box expansion or is closest to the center, while maintaining non-overlap.
3. Since all inner hexagons are aligned with the lattice (angle 0), we only need to optimize the outer hexagon's rotation and center.
4. We sweep the outer hexagon angle to find the minimal side length containing the fixed inner cluster.
5. We also try a few different lattice orientations (rotating the lattice itself) to capture different packing symmetries.
"""

import numpy as np
import time
import itertools

N = 12
SIDE = 1.0
LATTICE_SPACING = 2.0 * SIDE # Minimum center distance for non-overlapping unit hexagons on a triangular lattice

def _hex_vertices(cx, cy, angle_deg, side=SIDE):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    for verts in [v1, v2]:
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 + tol or max2 < min1 + tol:
                return False
    return True

def _dist_to_center(pos, center):
    return np.linalg.norm(pos - center)

def _generate_lattice_positions(max_radius, angle_offset=0.0):
    """
    Generate positions on a triangular lattice.
    The lattice is rotated by angle_offset.
    """
    positions = []
    # Triangular lattice basis vectors
    # Standard: (1, 0) and (0.5, sqrt(3)/2)
    # Scaled by LATTICE_SPACING
    a1 = np.array([LATTICE_SPACING, 0.0])
    a2 = np.array([LATTICE_SPACING * 0.5, LATTICE_SPACING * np.sqrt(3) / 2])
    
    # Rotation matrix for the lattice
    theta = np.radians(angle_offset)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    a1_rot = R @ a1
    a2_rot = R @ a2
    
    # Iterate over grid indices
    # max_radius determines how many lattice steps we go out
    # For 12 hexagons, we don't need a huge radius.
    # Bounding box of 12 hexagons is roughly 5-6 units wide.
    # Lattice spacing is 2. So radius ~ 3-4 steps is enough.
    max_steps = 4
    
    for i in range(-max_steps, max_steps + 1):
        for j in range(-max_steps, max_steps + 1):
            pos = i * a1_rot + j * a2_rot
            # Keep only positions within a reasonable bounding box
            if np.linalg.norm(pos) <= max_radius:
                positions.append(pos)
    
    return np.array(positions)

def _greedy_pack(lattice_positions, n_hexes=N):
    """
    Greedily place n_hexes from lattice_positions.
    Strategy: Sort positions by distance from origin.
    Place a hexagon if it doesn't overlap with already placed ones.
    This tends to create a compact cluster around the center.
    """
    # Sort by distance from origin
    dists = np.linalg.norm(lattice_positions, axis=1)
    sorted_indices = np.argsort(dists)
    
    placed = []
    vertices_cache = []
    
    for idx in sorted_indices:
        if len(placed) == n_hexes:
            break
        
        pos = lattice_positions[idx]
        v_new = _hex_vertices(pos[0], pos[1], 0.0)
        
        # Check overlap with all placed
        valid = True
        for v_old in vertices_cache:
            if _polygons_overlap(v_new, v_old, tol=-1e-6):
                valid = False
                break
        
        if valid:
            placed.append(pos)
            vertices_cache.append(v_new)
            
    if len(placed) < n_hexes:
        return None
        
    return np.array(placed)

def _min_outer_side(inner_positions, outer_angle_deg):
    """
    Calculate the minimum side length of an outer hexagon with given angle
    that contains all inner positions.
    We assume the outer hexagon is centered at the centroid of the inner positions
    for this quick estimate, but we can refine the center later if needed.
    Actually, the optimal center is the one that minimizes the max projection.
    For a fixed set of points and a fixed orientation, the minimal enclosing hexagon
    center can be found, but for a quick sweep, centering at centroid is a good heuristic.
    """
    if len(inner_positions) == 0:
        return 0.0
        
    # Centroid of inner positions
    center = np.mean(inner_positions, axis=0)
    
    angle_rad = np.radians(outer_angle_deg)
    max_proj = 0.0
    
    # Face normals of the outer hexagon
    for k in range(6):
        theta = angle_rad + k * np.pi / 3 + np.pi / 6
        nx = np.cos(theta)
        ny = np.sin(theta)
        
        # Project all inner vertices onto this normal
        # Note: We must check all vertices of all inner hexagons, not just centers
        # To be safe and fast, we approximate by checking the "radius" of the inner hexagon
        # in that direction. The max distance from center of a unit hexagon in any direction
        # is the circumradius = 1.0.
        # So for each inner center, the max projection is dot(center - outer_center, normal) + 1.0
        # Wait, this is an approximation. Let's be more precise but still fast.
        # The max projection of a convex polygon onto a normal is max(dot(v, normal)).
        # We can precompute the support function of a unit hexagon.
        # For a unit hexagon (side 1), the support function h(u) for unit vector u is:
        # It varies between inradius (sqrt(3)/2) and circumradius (1).
        # h(u) = 1.0 * max_{vertex} dot(v_hat, u)
        # Since the inner hexagons are all oriented at 0 degrees, we can precompute the 6 vertices.
        
        pass # We will do the exact vertex check in the main loop for accuracy
    
    # For the sweep, let's use the exact method but vectorized if possible.
    # However, N=12 is small.
    
    # Precompute vertices for all inner hexagons (all angle 0)
    all_verts = []
    for pos in inner_positions:
        v = _hex_vertices(pos[0], pos[1], 0.0)
        all_verts.append(v)
        
    max_proj = 0.0
    for k in range(6):
        theta = angle_rad + k * np.pi / 3 + np.pi / 6
        n_vec = np.array([np.cos(theta), np.sin(theta)])
        
        for v in all_verts:
            rel = v - center
            proj = np.dot(rel, n_vec)
            if proj > max_proj:
                max_proj = proj
                
    inradius = max_proj
    side = 2.0 * inradius / np.sqrt(3)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try different lattice orientations
    # The lattice orientation affects the "shape" of the cluster
    lattice_angles = [0, 30, 60, 90, 120, 150]
    
    for lat_angle in lattice_angles:
        if time.time() > deadline:
            break
            
        # Generate lattice
        # Radius needs to be large enough to hold 12 hexagons
        # 12 hexagons in a hex cluster (1+6+9... no, 1+6=7, 7+12=19)
        # A compact cluster of 12 is roughly 4x4 area.
        # Radius 10 should be safe.
        lattice_pos = _generate_lattice_positions(max_radius=12.0, angle_offset=lat_angle)
        
        # Greedy pack
        packed = _greedy_pack(lattice_pos, N)
        
        if packed is None:
            continue
            
        # The packed positions are relative to the lattice origin (0,0).
        # We should center the cluster at the origin for the outer hexagon calculation
        # or let the outer hexagon center adjust.
        # Let's center the inner cluster at (0,0) to simplify.
        centroid = np.mean(packed, axis=0)
        centered_packed = packed - centroid
        
        # Now find the best outer hexagon angle
        # Sweep angle from 0 to 60 degrees (symmetry)
        # Fine sweep
        angles = np.arange(0, 60, 1.0)
        
        for oa in angles:
            if time.time() > deadline:
                break
                
            side = _min_outer_side(centered_packed, oa)
            
            if side < best_side:
                best_side = side
                best_inner = centered_packed
                best_outer_center = (0.0, 0.0)
                best_outer_angle = oa
        
        # Refine the best angle found in the coarse sweep
        if best_inner is not None and best_outer_angle > 0:
            # Fine sweep around the best coarse angle
            ref_angles = np.arange(best_outer_angle - 2, best_outer_angle + 2, 0.1)
            for oa in ref_angles:
                if time.time() > deadline:
                    break
                side = _min_outer_side(centered_packed, oa)
                if side < best_side:
                    best_side = side
                    best_inner = centered_packed
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = oa

    # If no valid packing was found (should not happen), return a fallback
    if best_inner is None:
        # Fallback: simple grid
        init_x = []
        init_y = []
        init_angles = []
        # 4x3 grid
        for i in range(4):
            for j in range(3):
                x = i * 2.0
                y = j * np.sqrt(3)
                if i % 2 == 1:
                    y += np.sqrt(3)/2
                init_x.append(x)
                init_y.append(y)
                init_angles.append(0.0)
        best_inner = np.array(list(zip(init_x, init_y)))
        centroid = np.mean(best_inner, axis=0)
        best_inner = best_inner - centroid
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = _min_outer_side(best_inner, best_outer_angle)

    # Construct the return values
    inner_list = []
    for i in range(N):
        x = float(best_inner[i, 0])
        y = float(best_inner[i, 1])
        angle = 0.0 # All inner hexagons are aligned with the lattice
        inner_list.append([x, y, angle])
        
    outer_cx, outer_cy = best_outer_center
    outer_side = float(best_side)
    outer_angle = float(best_outer_angle)
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
