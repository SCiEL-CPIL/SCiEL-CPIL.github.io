# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))

def _hex_edge_normals(vertices):
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals[i] = normal / norm
    return normals

def _separation_check(v1, v2):
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _is_inside_outer(inner_vertices, outer_vertices):
    n_out = len(outer_vertices)
    for v in inner_vertices:
        inside = True
        for i in range(n_out):
            p1 = outer_vertices[i]
            p2 = outer_vertices[(i + 1) % n_out]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _compute_s(inner_positions, outer_center, outer_angle_deg, inner_angles):
    outer_center = np.array(outer_center)
    inner_centers = np.array(inner_positions)
    shifted = inner_centers - outer_center
    
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    all_vertices = []
    for i in range(len(inner_centers)):
        cx, cy = shifted[i]
        verts = _hex_vertices(cx, cy, 1, inner_angles[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    
    s = 2 * max_proj / np.sqrt(3)
    return s

def _compute_s_fast(verts_relative, outer_angle_deg):
    """Compute minimal enclosing hexagon side length given vertices relative to center."""
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    projections = verts_relative @ normals.T
    max_proj = projections.max()
    
    s = 2 * max_proj / np.sqrt(3)
    return s

def _generate_honeycomb_clusters():
    """Generate several honeycomb cluster configurations with 12 hexagons.
    Returns list of (positions, angles) where positions is (12,2) and angles is (12,)."""
    clusters = []
    
    # Standard honeycomb: center + ring of 6 + extended ring
    # In a honeycomb, adjacent hexagons touch. For unit hexagons (side 1),
    # the distance between centers of touching hexagons is 2*sqrt(3)/2 = sqrt(3)
    # Wait, for flat-top hexagons with side 1, the distance between centers of adjacent hexagons is sqrt(3).
    # Actually, for a regular hexagon with side length s, the distance from center to vertex is s.
    # The distance from center to edge midpoint is s*sqrt(3)/2.
    # Two touching hexagons: center-to-center distance = 2 * s*sqrt(3)/2 = s*sqrt(3).
    # For s=1: distance = sqrt(3).
    
    d = np.sqrt(3)  # distance between centers of touching hexagons
    
    # Cluster 1: 3-4-3-2 arrangement (or similar compact arrangement)
    # Let's use a hexagonal grid coordinate system.
    # Axial coordinates (q, r) -> Cartesian: x = d*(q + r/2), y = d*(3*r/2) for pointy-top
    # Or for flat-top: x = d*(3*q/2), y = d*(r + q/2)
    
    # Let's use flat-top orientation. Angle 0 means flat top.
    # For flat-top hexagons, the 6 neighbors are at angles 0, 60, 120, 180, 240, 300 degrees
    # at distance d = sqrt(3).
    
    # Actually, let me think more carefully.
    # A regular hexagon with side 1, flat-top (angle=0):
    # Vertices at angles 0, 60, 120, 180, 240, 300 from center, at distance 1.
    # The "pointy" directions are at 30, 90, 150, 210, 270, 330.
    # Adjacent hexagons share an edge. The edge midpoint is at distance sqrt(3)/2 from center.
    # So center-to-center distance for touching hexagons = 2 * sqrt(3)/2 = sqrt(3).
    
    # Neighbor directions for flat-top hexagons:
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330.
    # So neighbors are in directions 30, 90, 150, 210, 270, 330 at distance sqrt(3).
    
    neighbor_dirs = np.radians([30, 90, 150, 210, 270, 330])
    neighbor_offsets = d * np.column_stack((np.cos(neighbor_dirs), np.sin(neighbor_dirs)))
    
    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for flat-top hexagons."""
        # In a hex grid, moving in direction i changes axial coords.
        # Let's just use a lookup approach: build the cluster by BFS.
        x = d * (q + 0.5 * r)
        y = d * (np.sqrt(3) / 2 * r)
        # Hmm, this isn't quite right. Let me use a simpler approach.
        return x, y
    
    # Simpler: define clusters explicitly using neighbor steps.
    # Start with center at (0,0). Add neighbors.
    
    # Cluster A: center + 6 neighbors + 5 more (12 total)
    # Positions in a hex grid pattern
    pos_A = []
    pos_A.append((0, 0))  # center
    # 6 neighbors
    for i in range(6):
        pos_A.append(tuple(neighbor_offsets[i]))
    # Next ring: pick 5 more to make 12
    # The next ring has 12 positions. We pick 5.
    # Positions in next ring are at distance 2*d in various directions, or d*sqrt(3) in diagonal directions.
    # Let's add: (2*d*cos(30), 2*d*sin(30)), etc.
    # Actually, the second ring positions are at:
    # For each neighbor direction, go two steps: 2*neighbor_offsets[i]
    # And the "diagonal" positions: neighbor_offsets[i] + neighbor_offsets[(i+1)%6]
    
    # Let's add 5 from the second ring to make a compact shape.
    # Add: 2*offset[0], 2*offset[1], 2*offset[2], 2*offset[3], 2*offset[4]
    for i in range(5):
        pos_A.append(tuple(2 * neighbor_offsets[i]))
    
    pos_A = np.array(pos_A)
    angles_A = np.zeros(12)
    clusters.append((pos_A, angles_A))
    
    # Cluster B: different arrangement
    pos_B = []
    pos_B.append((0, 0))
    for i in range(6):
        pos_B.append(tuple(neighbor_offsets[i]))
    # Add different 5 from second ring
    for i in [0, 1, 2, 3, 4]:
        pos_B.append(tuple(neighbor_offsets[i] + neighbor_offsets[(i+1)%6]))
    
    pos_B = np.array(pos_B)
    angles_B = np.zeros(12)
    clusters.append((pos_B, angles_B))
    
    # Cluster C: more elongated
    pos_C = []
    pos_C.append((0, 0))
    for i in range(6):
        pos_C.append(tuple(neighbor_offsets[i]))
    # Add in a line
    for i in range(5):
        pos_C.append(tuple(neighbor_offsets[0] * (i+1) + neighbor_offsets[5] * 0))
    # That's not right. Let me just offset along one direction.
    pos_C = []
    pos_C.append((0, 0))
    for i in range(6):
        pos_C.append(tuple(neighbor_offsets[i]))
    # Add 5 more in a row extending from one neighbor
    base = neighbor_offsets[0]
    for i in range(1, 6):
        pos_C.append(tuple(base * i))
    
    pos_C = np.array(pos_C)
    angles_C = np.zeros(12)
    clusters.append((pos_C, angles_C))
    
    return clusters

def _validate_and_compute_s(pos, angles):
    """Validate the configuration and compute s. Returns (s, oc, oa, valid)."""
    # Check separation between all pairs
    valid = True
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(pos[i,0], pos[i,1], 1, angles[i])
            v2 = _hex_vertices(pos[j,0], pos[j,1], 1, angles[j])
            if not _separation_check(v1, v2):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None, None, None, False
    
    # Compute all vertices
    all_vertices = []
    for i in range(12):
        verts = _hex_vertices(pos[i,0], pos[i,1], 1, angles[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Find centroid
    centroid = all_vertices.mean(axis=0)
    verts_relative = all_vertices - centroid
    
    # Optimize outer hexagon angle
    def obj_outer(oa):
        return _compute_s_fast(verts_relative, oa)
    
    # Grid search first
    oa_grid = np.linspace(0, 60, 121)
    s_grid = np.array([_compute_s_fast(verts_relative, oa) for oa in oa_grid])
    best_idx = s_grid.argmin()
    oa_init = oa_grid[best_idx]
    
    # Refine with minimization
    try:
        res = minimize(obj_outer, oa_init, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
        oa = res.x[0]
        s = res.fun
    except:
        oa = oa_init
        s = _compute_s_fast(verts_relative, oa)
    
    # Check that all inner vertices are inside the outer hexagon
    oc = centroid
    outer_verts = _hex_vertices(oc[0], oc[1], s, oa)
    for i in range(12):
        inner_verts = _hex_vertices(pos[i,0], pos[i,1], 1, angles[i])
        if not _is_inside_outer(inner_verts, outer_verts):
            return s, oc, oa, False
    
    return s, oc, oa, True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_s = float('inf')
    best_pos = None
    best_angles = None
    best_oc = None
    best_oa = None
    
    # Generate initial clusters
    clusters = _generate_honeycomb_clusters()
    
    # Try each cluster with various global rotations
    for cluster_idx, (pos, angles) in enumerate(clusters):
        if time.time() > deadline - 30:
            break
        
        # Try global rotation angles
        for rot_deg in range(0, 30, 1):
            if time.time() > deadline - 30:
                break
            
            # Rotate all positions by rot_deg around origin
            theta = np.radians(rot_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            
            pos_rot = pos @ rot_matrix.T
            angles_rot = (angles + rot_deg) % 60
            
            s, oc, oa, valid = _validate_and_compute_s(pos_rot, angles_rot)
            if valid and s < best_s:
                best_s = s
                best_pos = pos_rot.copy()
                best_angles = angles_rot.copy()
                best_oc = oc.copy()
                best_oa = oa
        
        # Also try with some angle variations
        for angle_var in [0, 15, 30, 45]:
            if time.time() > deadline - 30:
                break
            
            angles_var = np.full(12, angle_var)
            s, oc, oa, valid = _validate_and_compute_s(pos, angles_var)
            if valid and s < best_s:
                best_s = s
                best_pos = pos.copy()
                best_angles = angles_var.copy()
                best_oc = oc.copy()
                best_oa = oa
    
    # Local optimization: perturb positions slightly
    if best_pos is not None and time.time() < deadline - 60:
        # Try small random perturbations
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            
            pos_pert = best_pos + np.random.normal(0, 0.1, best_pos.shape)
            s, oc, oa, valid = _validate_and_compute_s(pos_pert, best_angles)
            if valid and s < best_s:
                best_s = s
                best_pos = pos_pert.copy()
                best_oc = oc.copy()
                best_oa = oa
        
        # Try angle perturbations
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            
            angles_pert = (best_angles + np.random.uniform(-5, 5, 12)) % 60
            s, oc, oa, valid = _validate_and_compute_s(best_pos, angles_pert)
            if valid and s < best_s:
                best_s = s
                best_angles = angles_pert.copy()
                best_oc = oc.copy()
                best_oa = oa
    
    # Fallback: use the original seed if nothing worked
    if best_pos is None:
        init_pos = np.array([
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17],
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
            [0, -4]
        ])
        init_angles = np.zeros(12)
        s, oc, oa, valid = _validate_and_compute_s(init_pos, init_angles)
        best_pos = init_pos
        best_angles = init_angles
        best_oc = oc if oc is not None else np.array([0, 0])
        best_oa = oa if oa is not None else 0
        best_s = s if s is not None else 8.0
    
    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_pos[i, 0]
        inner_hex_data[i, 1] = best_pos[i, 1]
        inner_hex_data[i, 2] = best_angles[i]
    
    outer_hex_data = np.array([best_oc[0], best_oc[1], best_oa])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
