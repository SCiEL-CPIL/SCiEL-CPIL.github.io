# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, differential_evolution


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side_for_rotation(inner_centers, inner_angles, rot_deg):
    """
    Given inner hex centers and angles, find minimum side length s of a regular hexagon
    (rotated by rot_deg) that contains all 12 inner hexagons.
    Returns (s, best_center_offset) or (inf, None) if infeasible.
    """
    rot = np.radians(rot_deg)
    
    # All inner vertices
    all_verts = []
    for i in range(len(inner_centers)):
        cx, cy = inner_centers[i]
        aang = inner_angles[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        all_verts.extend(verts.tolist())
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


def _lattice_centers(e1, e2, lattice_points):
    """Compute centers from lattice basis vectors and integer lattice points."""
    centers = np.zeros((len(lattice_points), 2))
    for i, (a, b) in enumerate(lattice_points):
        centers[i] = a * e1 + b * e2
    return centers


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses crystallographic lattice representation with optimized lattice basis vectors.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Lattice points for 12 hexagons in a compact arrangement
    # Center + 6 first ring + 5 closest from second ring
    lattice_points = [
        (0, 0),      # center
        (1, 0),      # 6 neighbors
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (-1, 2),     # 5 from second ring (closest to center)
        (-2, 1),
        (1, -2),
        (2, -1),
        (0, 2),
    ]
    lattice_points = np.array(lattice_points)
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Use a fixed, slightly expanded lattice to ensure strict separation (gap > 1e-6)
    # Standard spacing is 2. We use 2.002 to create a gap of 0.002 between adjacent hexes.
    scale = 1.002
    e1_std = np.array([2.0 * scale, 0.0])
    e2_std = np.array([1.0 * scale, np.sqrt(3) * scale])
    centers_std = _lattice_centers(e1_std, e2_std, lattice_points)
    inner_angles_std = np.zeros(12)
    
    # Strategy: Fine rotation search
    # The shape is fixed, so we only need to find the best rotation of the outer hexagon.
    # We scan rotations from 0 to 60 degrees.
    for rot_deg in np.linspace(0, 60, 361):
        if time.time() > deadline - 30:
            break
        s, center = _min_outer_side_for_rotation(centers_std, inner_angles_std, rot_deg)
        if s < best_s:
            inner_hex_data = np.column_stack([centers_std, inner_angles_std])
            outer_hex_data = np.array([center[0], center[1], rot_deg])
            if _validate_packing(inner_hex_data, outer_hex_data, s):
                best_s = s
                best_inner = inner_hex_data
                best_outer = outer_hex_data
    
    # Strategy 2: Try standard hexagonal lattice with fine rotation search
    if time.time() < deadline - 60:
        # Standard hexagonal lattice with spacing 2
        e1_std = np.array([2.0, 0.0])
        e2_std = np.array([1.0, sqrt3])
        centers_std = _lattice_centers(e1_std, e2_std, lattice_points)
        inner_angles_std = np.zeros(12)
        
        # Fine rotation search
        for rot_deg in np.linspace(0, 60, 121):
            if time.time() > deadline - 30:
                break
            s, center = _min_outer_side_for_rotation(centers_std, inner_angles_std, rot_deg)
            if s < best_s:
                inner_hex_data = np.column_stack([centers_std, inner_angles_std])
                outer_hex_data = np.array([center[0], center[1], rot_deg])
                if _validate_packing(inner_hex_data, outer_hex_data, s):
                    best_s = s
                    best_inner = inner_hex_data
                    best_outer = outer_hex_data
    
    # Strategy 3: Try slightly compressed lattice
    if time.time() < deadline - 30:
        for scale in [0.998, 0.996, 0.994, 0.992]:
            if time.time() > deadline - 20:
                break
            e1_comp = np.array([2.0 * scale, 0.0])
            e2_comp = np.array([1.0 * scale, sqrt3 * scale])
            centers_comp = _lattice_centers(e1_comp, e2_comp, lattice_points)
            inner_angles_comp = np.zeros(12)
            
            for rot_deg in np.linspace(0, 60, 61):
                if time.time() > deadline - 15:
                    break
                s, center = _min_outer_side_for_rotation(centers_comp, inner_angles_comp, rot_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers_comp, inner_angles_comp])
                    outer_hex_data = np.array([center[0], center[1], rot_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data
                        best_outer = outer_hex_data
    
    # Fallback: use standard packing
    if best_inner is None:
        e1_std = np.array([2.0, 0.0])
        e2_std = np.array([1.0, sqrt3])
        centers_std = _lattice_centers(e1_std, e2_std, lattice_points)
        inner_angles_std = np.zeros(12)
        s, center = _min_outer_side_for_rotation(centers_std, inner_angles_std, 0)
        best_s = s
        best_inner = np.column_stack([centers_std, inner_angles_std])
        best_outer = np.array([center[0], center[1], 0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
