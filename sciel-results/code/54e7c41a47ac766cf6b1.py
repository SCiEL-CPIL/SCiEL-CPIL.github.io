# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side_from_positions(positions, angles_deg):
    """
    Given centers (N,2) and angles (N,), compute the minimum side length s of a
    regular hexagon centered at origin with angle 0 that contains all N unit hexagons.
    s_min = (2/sqrt(3)) * max over all vertices v of max over 6 inward normals n_k of (n_k . v)
    """
    # Inward normals for outer hex with vertices at 0, 60, ..., 300 degrees
    # Edge between vertex at angle theta_i and theta_{i+1}: inward normal is at angle (theta_i + theta_{i+1})/2 + 180? 
    # Actually for a regular hexagon centered at origin with vertices at 0,60,...,300:
    # The edge from vertex 0 to vertex 60 has midpoint direction 30 degrees.
    # The inward normal points toward origin, so inward normal direction is 30 + 180 = 210 degrees? No.
    # Let's think: the hexagon is the set of points where n_k . v <= R for all 6 inward normals.
    # For a regular hexagon with "radius" (center to vertex) = s, the apothem (center to edge midpoint) = s * cos(30) = s * sqrt(3)/2.
    # The inward normal for the edge between vertices at 0 and 60 degrees points in direction 30 degrees (outward) so inward is -30 or 150? 
    # The edge midpoint is at angle 30, distance s*cos(30) from center. The outward normal is in direction 30 degrees.
    # So inward normal is in direction 30 + 180 = 210 degrees? That doesn't seem right either.
    # Let me just compute: for the hexagon defined by vertices at angles 0,60,...,300 with side s,
    # the constraint is that for each edge, the signed distance from center to edge is s*sqrt(3)/2.
    # The outward normal for edge between vertex i and i+1 is in the direction of the angle bisector.
    # For vertices at 0 and 60 degrees, the bisector is at 30 degrees. Outward normal = (cos30, sin30).
    # Inward normal = -(cos30, sin30).
    # A point v is inside if inward_normal . v >= -s*sqrt(3)/2, i.e., outward_normal . v <= s*sqrt(3)/2.
    # So s_min = (2/sqrt(3)) * max_v max_k (outward_normal_k . v)
    
    outward_normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))],
    ])
    
    # Compute all vertices
    all_verts = []
    for i in range(len(positions)):
        v = _hex_vertices(positions[i,0], positions[i,1], 1.0, angles_deg[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # For each vertex, compute max over 6 outward normals of (n . v)
    # (72, 6) matrix
    dot_products = all_verts @ outward_normals.T
    max_dot = np.max(dot_products)
    
    s_min = (2.0 / np.sqrt(3.0)) * max_dot
    return s_min

def _check_nonoverlap(positions, angles_deg, tol=1e-6):
    """
    Check that no two inner hexagons overlap using separating axis test.
    Returns True if all pairs are non-overlapping.
    """
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles_deg[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = verts_list[i]
            v2 = verts_list[j]
            # Get edge normals from both hexagons
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            # Normals (perpendicular to edges)
            n1 = np.column_stack([-edges1[:,1], edges1[:,0]])
            n2 = np.column_stack([-edges2[:,1], edges2[:,0]])
            all_normals = np.vstack([n1, n2])
            
            overlap = True
            for normal in all_normals:
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - tol or max2 < min1 - tol:
                    overlap = False
                    break
            if overlap:
                return False
    return True

def _initial_positions():
    """
    Create a tight initial arrangement of 12 unit hexagons.
    Uses 1 center + 6 ring + 5 outer ring to make exactly 12.
    """
    sqrt3 = np.sqrt(3)
    positions = []
    # Center
    positions.append([0.0, 0.0])
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        positions.append([sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)])
    # Ring 2: 5 more in the gaps (at angles 30, 90, 150, 210, 270, distance 2*sqrt(3))
    # We skip one to ensure total count is 12.
    for k in range(5):
        angle = np.radians(60 * k + 30)
        positions.append([2*sqrt3 * np.cos(angle), 2*sqrt3 * np.sin(angle)])
    
    positions = np.array(positions)
    angles_deg = np.zeros(12)
    return positions, angles_deg

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    # Initial positions
    positions, angles_deg = _initial_positions()
    
    # Compute initial s
    s = _min_outer_side_from_positions(positions, angles_deg)
    best_positions = positions.copy()
    best_angles = angles_deg.copy()
    best_s = s
    
    # Verify non-overlap
    if not _check_nonoverlap(positions, angles_deg):
        # Fallback: use a looser arrangement
        positions = np.array([
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
            [0, -4.0],
        ], dtype=float)
        angles_deg = np.zeros(12)
        s = _min_outer_side_from_positions(positions, angles_deg)
        best_positions = positions.copy()
        best_angles = angles_deg.copy()
        best_s = s
    
    # Simulated annealing
    if time.time() < deadline:
        T = 0.5
        T_min = 0.001
        cooling = 0.995
        max_steps = 50000
        rng = np.random.default_rng(42)
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random hexagon to move
            idx = rng.integers(0, 12)
            orig_pos = positions[idx].copy()
            orig_angle = angles_deg[idx]
            
            # Random perturbation
            dx = rng.normal(0, T * 0.5)
            dy = rng.normal(0, T * 0.5)
            dangle = rng.normal(0, T * 5.0)
            
            new_pos = positions.copy()
            new_pos[idx] = [orig_pos[0] + dx, orig_pos[1] + dy]
            new_angles = angles_deg.copy()
            new_angles[idx] = orig_angle + dangle
            
            # Check non-overlap
            if not _check_nonoverlap(new_pos, new_angles):
                # Reject
                continue
            
            # Compute new s
            new_s = _min_outer_side_from_positions(new_pos, new_angles)
            
            # Accept if better or with probability exp(-(new_s - s)/T)
            if new_s < s:
                positions = new_pos
                angles_deg = new_angles
                s = new_s
                if s < best_s:
                    best_s = s
                    best_positions = positions.copy()
                    best_angles = angles_deg.copy()
            else:
                delta = new_s - s
                if delta < 0:
                    positions = new_pos
                    angles_deg = new_angles
                    s = new_s
                    if s < best_s:
                        best_s = s
                        best_positions = positions.copy()
                        best_angles = angles_deg.copy()
                else:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        positions = new_pos
                        angles_deg = new_angles
                        s = new_s
            
            T = max(T * cooling, T_min)
        
        # Use best found
        positions = best_positions
        angles_deg = best_angles
        s = best_s
    
    # Final: ensure validity
    if not _check_nonoverlap(positions, angles_deg):
        # Fallback to seed
        positions = np.array([
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
            [0, -4.0],
        ], dtype=float)
        angles_deg = np.zeros(12)
        s = _min_outer_side_from_positions(positions, angles_deg)
    
    # Recompute s precisely
    s = _min_outer_side_from_positions(positions, angles_deg)
    
    inner_hex_data = np.column_stack([positions, angles_deg])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
