import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    sqrt3 = np.sqrt(3.0)

    def hex_vertices(cx, cy, angle_deg, s=1.0):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def lattice_to_cartesian(q, r):
        """Axial hex coordinates to Cartesian (pointy-top, unit side)."""
        x = sqrt3 * (q + 0.5 * r)
        y = 1.5 * r
        return x, y

    def compute_min_side(vertices_flat, outer_angle_deg):
        """
        Given flat vertices (N, 2) and outer hexagon angle,
        compute minimum side length S such that outer hex (centered at origin)
        contains all vertices.
        """
        a = np.deg2rad(outer_angle_deg)
        normal_angles = a + np.deg2rad(np.arange(30, 390, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        # (N,2) @ (2,6) -> (N,6)
        projections = vertices_flat @ normals.T
        max_proj = np.max(projections, axis=0)
        # For a regular hexagon with side S and circumradius S,
        # the apothem (distance from center to edge) = S * sqrt(3)/2
        # So we need S * sqrt(3)/2 >= max_proj for all 6 normals
        # But also need S * sqrt(3)/2 >= -min_proj (the other side)
        min_proj = np.min(projections, axis=0)
        needed = np.maximum(max_proj, -min_proj)
        return 2.0 * np.max(needed) / sqrt3

    def check_containment(inner_verts, outer_verts, tol=1e-6):
        """Check all inner vertices are inside or on outer hexagon."""
        n_outer = len(outer_verts)
        for i in range(n_outer):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % n_outer]
            edge = p2 - p1
            # Cross product: should be >= -tol for inside
            for v in inner_verts:
                cp = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cp < -tol:
                    return False
        return True

    def check_separation(verts_list, tol=1e-6):
        """SAT check: all pairs of hexagons must be separated."""
        n = len(verts_list)
        for i in range(n):
            v1 = verts_list[i]
            for j in range(i + 1, n):
                v2 = verts_list[j]
                # 12 edge normals
                all_ok = False
                for k in range(6):
                    e1 = np.roll(v1, -1, axis=0) - v1
                    n1 = np.array([-e1[k, 1], e1[k, 0]])
                    n1 /= np.linalg.norm(n1)
                    proj1 = v1 @ n1
                    proj2 = v2 @ n1
                    if np.min(proj2) - np.max(proj1) > tol or np.min(proj1) - np.max(proj2) > tol:
                        all_ok = True
                        break
                if not all_ok:
                    for k in range(6):
                        e2 = np.roll(v2, -1, axis=0) - v2
                        n2 = np.array([-e2[k, 1], e2[k, 0]])
                        n2 /= np.linalg.norm(n2)
                        proj1 = v1 @ n2
                        proj2 = v2 @ n2
                        if np.min(proj2) - np.max(proj1) > tol or np.min(proj1) - np.max(proj2) > tol:
                            all_ok = True
                            break
                if not all_ok:
                    return False
        return True

    def check_edge_lengths(verts_list, tol=1e-8):
        for v in verts_list:
            for k in range(6):
                p1 = v[k]
                p2 = v[(k + 1) % 6]
                if abs(np.linalg.norm(p2 - p1) - 1.0) > tol:
                    return False
        return True

    def evaluate_packing(positions, angles_deg, outer_angle_deg):
        """
        positions: (11, 2) Cartesian centers
        angles_deg: (11,) rotation angles for each inner hex
        outer_angle_deg: rotation of outer hexagon
        Returns (is_valid, side_length)
        """
        verts_list = []
        all_verts = []
        for i in range(11):
            v = hex_vertices(positions[i, 0], positions[i, 1], angles_deg[i], 1.0)
            verts_list.append(v)
            all_verts.append(v)
        all_verts = np.concatenate(all_verts, axis=0)  # (66, 2)

        S = compute_min_side(all_verts, outer_angle_deg)
        if S < np.sqrt(11.0):
            return False, S

        # Check containment with the computed S
        outer_verts = hex_vertices(0, 0, outer_angle_deg, S)
        if not check_containment(all_verts, outer_verts):
            return False, S

        # Check separation
        if not check_separation(verts_list):
            return False, S

        # Check edge lengths
        if not check_edge_lengths(verts_list):
            return False, S

        return True, S

    # Generate candidate lattice position sets
    # Axial coordinates (q, r) for 11 hexagons
    # Try various compact arrangements
    candidate_sets = []

    # Set 1: 1 + 6 + 4 (center, ring of 6, 4 more)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (0,2), (-2,2), (2,-2)
    set1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,2), (2,-2)]
    candidate_sets.append(set1)

    # Set 2: 1 + 6 + 4 different
    set2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (-2,1), (-1,2), (1,-2)]
    candidate_sets.append(set2)

    # Set 3: More spread
    set3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-2,1), (-1,-1)]
    candidate_sets.append(set3)

    # Set 4: Row-based
    set4 = [(-2,-1), (-1,-1), (0,-1), (1,-1), (2,-1), (-1,0), (0,0), (1,0), (-2,1), (0,1), (2,1)]
    candidate_sets.append(set4)

    # Set 5: Diamond
    set5 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1), (2,0), (0,2)]
    candidate_sets.append(set5)

    # Set 6: Another compact
    set6 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (1,1), (-2,0), (-1,-2)]
    candidate_sets.append(set6)

    # Set 7: 3-4-4
    set7 = [(-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (-1,-1), (0,-1), (1,-1), (2,0), (0,2)]
    candidate_sets.append(set7)

    # Set 8: 2-3-3-3
    set8 = [(-1,-1), (0,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (-2,0), (2,1), (0,-2)]
    candidate_sets.append(set8)

    # Set 9: Compact star
    set9 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (-2,1), (1,-2), (-1,-2)]
    candidate_sets.append(set9)

    # Set 10: Line + offsets
    set10 = [(-2,0), (-1,0), (0,0), (1,0), (2,0), (-1,1), (0,1), (1,1), (-1,-1), (0,-1), (1,-1)]
    candidate_sets.append(set10)

    # Set 11: 1+5+5
    set11 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (2,0), (1,1), (-2,1), (1,-1), (-2,-1)]
    candidate_sets.append(set11)

    # Set 12: 1+6+4 yet another
    set12 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (-2,1), (0,2), (2,1)]
    candidate_sets.append(set12)

    # For each candidate set, try different inner hex rotations and outer hex orientations
    best_S = np.inf
    best_result = None

    # Inner hex rotation options (degrees)
    inner_rot_options = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # Outer hex rotation options
    outer_rot_options = list(range(0, 180, 15))

    for set_idx, positions_axial in enumerate(candidate_sets):
        if time.time() > deadline - 30:
            break

        # Convert to Cartesian
        positions_cart = np.array([lattice_to_cartesian(q, r) for q, r in positions_axial])

        # Try combinations of inner rotations (use same rotation for all to keep it manageable)
        for inner_rot in inner_rot_options:
            angles_deg = np.full(11, inner_rot)

            for outer_rot in outer_rot_options:
                if time.time() > deadline - 15:
                    break

                valid, S = evaluate_packing(positions_cart, angles_deg, outer_rot)
                if valid and S < best_S:
                    best_S = S
                    best_result = (positions_cart.copy(), angles_deg.copy(), outer_rot)

        # Also try mixed rotations: alternate 0 and 30 degrees
        if time.time() > deadline - 15:
            break
        for outer_rot in outer_rot_options:
            if time.time() > deadline - 15:
                break
            # Alternate rotations
            for alt_pattern in [0, 1]:
                angles_deg = np.array([0 if (i % 2 == alt_pattern) else 30 for i in range(11)])
                valid, S = evaluate_packing(positions_cart, angles_deg, outer_rot)
                if valid and S < best_S:
                    best_S = S
                    best_result = (positions_cart.copy(), angles_deg.copy(), outer_rot)

    # If we found a valid result, format and return
    if best_result is not None:
        positions_cart, angles_deg, outer_rot = best_result
        inner_hex_data = np.column_stack([positions_cart[:, 0], positions_cart[:, 1], angles_deg])
        outer_hex_data = np.array([0.0, 0.0, float(outer_rot)])
        return inner_hex_data, outer_hex_data, float(best_S)

    # Fallback: construct a safe large packing
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length
