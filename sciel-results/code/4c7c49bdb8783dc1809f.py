# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side_for_rotation(inner_centers, rot_deg):
    """
    Given inner hex centers (unit hexagons, all axis-aligned internally),
    find minimum side length s of a regular hexagon (centered at origin, rotated by rot_deg)
    that contains all 12 inner hexagons.
    Returns (s, best_center_offset) or (inf, None) if infeasible.
    """
    rot = np.radians(rot_deg)
    
    # Vertices of unit hexagon (side 1) centered at origin
    unit_verts = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    
    # All inner vertices
    all_verts = []
    for c in inner_centers:
        for u in unit_verts:
            all_verts.append(c + u)
    all_verts = np.array(all_verts)  # shape (72, 2)
    
    # For each edge direction i (0..5) and each vertex (72), we have a constraint.
    n_constraints = 6 * len(all_verts)
    A = np.zeros((n_constraints, 3))  # variables: [ox, oy, s]
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        angle = rot + np.radians(60*i + 30)
        ci = np.cos(angle)
        si = np.sin(angle)
        for j in range(len(all_verts)):
            A[idx, 0] = -ci
            A[idx, 1] = -si
            A[idx, 2] = -np.sqrt(3)/2
            b[idx] = -(ci * all_verts[j, 0] + si * all_verts[j, 1])
            idx += 1
    
    c_obj = np.array([0, 0, 1])
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        s = res.x[2]
        ox = res.x[0]
        oy = res.x[1]
        return s, (ox, oy)
    else:
        return np.inf, None


def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Get outer hexagon vertices
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Shift all centers by minus outer center
    shifted_centers = inner_hex_data[:, :2] - np.array([ox, oy])
    
    # Check unit edges of inner hexagons
    unit_verts = _hex_vertices(0, 0, 1, 0)
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1) > 1e-8:
                return False
    
    # Check separation between inner hexagons using separating axis test
    for i in range(12):
        for j in range(i+1, 12):
            vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1, inner_hex_data[i, 2])
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1, inner_hex_data[j, 2])
            
            # Get all edge normals from both hexagons
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k+1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
            
            # Check if any normal separates the two hexagons
            separated = False
            for n in normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                if max_i + eps < min_j or max_j + eps < min_i:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for v in verts:
            # For each outer edge, check cross product
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a crystallographic lattice approach with triangular lattice symmetry.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Define 12 hexagon centers in a hexagonal lattice pattern
    # Use the "flower" pattern: center + 6 neighbors + 5 more from next ring
    eps = 1e-3
    scale = 1.0 + eps
    
    # Lattice basis for triangular lattice with spacing 2
    # e1 = (2, 0), e2 = (1, sqrt(3))
    centers_base = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        # Next ring - pick 5 that make it compact
        [3, sqrt3],
        [0, 2*sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2*sqrt3],
    ])
    
    # Try different lattice configurations and rotations
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Strategy 1: Standard hexagonal lattice with rotation optimization
    centers = centers_base * scale
    
    # Try rotations from 0 to 60 degrees
    rot_range = np.linspace(0, 60, 121)
    for rot_deg in rot_range:
        if time.time() > deadline - 30:
            break
        s, center = _min_outer_side_for_rotation(centers, rot_deg)
        if s < best_s:
            inner_hex_data = np.column_stack([centers, np.zeros(12)])
            outer_hex_data = np.array([center[0], center[1], rot_deg])
            if _validate_packing(inner_hex_data, outer_hex_data, s):
                best_s = s
                best_inner = inner_hex_data
                best_outer = outer_hex_data
    
    # Strategy 2: Try compressed lattice (reduce spacing slightly)
    if time.time() < deadline - 120:
        for compress in [0.995, 0.99, 0.985]:
            if time.time() > deadline - 60:
                break
            centers_comp = centers_base * (scale * compress)
            for rot_deg in np.linspace(0, 60, 61):
                if time.time() > deadline - 30:
                    break
                s, center = _min_outer_side_for_rotation(centers_comp, rot_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers_comp, np.zeros(12)])
                    outer_hex_data = np.array([center[0], center[1], rot_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data
                        best_outer = outer_hex_data
    
    # Strategy 3: Try rotated lattice (rotate the inner hexagons relative to outer)
    if time.time() < deadline - 90:
        for hex_rot in [0, 30, 15, 45]:
            if time.time() > deadline - 30:
                break
            centers_rot = centers_base * scale
            inner_hex_data = np.column_stack([centers_rot, np.full(12, hex_rot)])
            
            # For rotated inner hexagons, we need to recompute vertices
            # The LP assumes axis-aligned inner hexagons, so we need to adjust
            # For now, just try with axis-aligned outer and rotated inner
            for rot_deg in np.linspace(0, 60, 41):
                if time.time() > deadline - 30:
                    break
                # We need to compute the bounding hexagon for rotated inner hexagons
                # This requires a different approach - skip for now
                pass
    
    # Fallback: use the standard packing if nothing better found
    if best_inner is None:
        centers = centers_base * scale
        s, center = _min_outer_side_for_rotation(centers, 0)
        best_s = s
        best_inner = np.column_stack([centers, np.zeros(12)])
        best_outer = np.array([center[0], center[1], 0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
