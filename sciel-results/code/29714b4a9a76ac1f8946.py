"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)

        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True

    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of (x, y) for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)

        if not all_verts:
            return 0.0

        all_verts = np.vstack(all_verts)

        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))

        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())

        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    # === LP-based packing optimization ===
    # The key insight: for a fixed set of inner hexagon angles and a fixed outer hexagon angle,
    # the problem of minimizing the outer side length subject to non-overlap can be formulated
    # using the support function. However, non-overlap is non-convex. 
    # 
    # Instead, we use a different approach:
    # 1. For a fixed configuration of inner hexagon positions and angles, compute the required
    #    outer side length.
    # 2. Use a mixed approach: fix the relative positions to be on a hexagonal lattice (which 
    #    guarantees non-overlap for the standard orientation), then optimize the overall 
    #    transformation (rotation, scaling of the lattice, and outer hexagon angle) to minimize
    #    the bounding hexagon side length.
    #
    # Actually, a better structural change: 
    # We know that for hexagons of side 1 on a hexagonal lattice, the minimum distance between
    # centers is 2 (for touching). The honeycomb arrangement is optimal for large n.
    # For n=12, we can try to find the optimal subset of the hexagonal lattice that minimizes
    # the bounding hexagon.
    #
    # Let's use a different representation: instead of optimizing positions directly, we
    # parameterize the packing by:
    # - The lattice basis vectors (which determine the relative positions)
    # - The set of 12 lattice points used
    # - The rotation of the entire lattice
    # - The rotation of the outer hexagon
    #
    # But this is complex. Let's try a simpler structural change:
    # Use a force-directed / physics-based approach with a better potential function,
    # combined with simulated annealing for escape from local minima.

    def physics_optimize(positions, angles, outer_angle, max_iters=20000, dt=0.01, cooling_rate=0.9995):
        """
        Physics-based optimization with simulated annealing.
        Forces: 
        - Repulsion between overlapping hexagons
        - Attraction toward center (to minimize bounding box)
        """
        pos = np.array(positions, dtype=float)
        n = len(pos)
        current_score = get_config_score(list(map(tuple, pos)), angles, outer_angle)
        best_pos = pos.copy()
        best_score = current_score

        temperature = 1.0

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Compute forces
            forces = np.zeros_like(pos)
            verts = [hex_vertices(pos[i][0], pos[i][1], angles[i]) for i in range(n)]

            # Repulsion forces for overlapping pairs
            for i in range(n):
                for j in range(i+1, n):
                    if hexagons_overlap(verts[i], verts[j]):
                        # Push apart
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        if dist > 1e-10:
                            force_mag = 10.0 / (dist + 0.1)
                            forces[i] -= force_mag * d / dist
                            forces[j] += force_mag * d / dist

            # Attraction toward center (weak)
            center = pos.mean(axis=0)
            for i in range(n):
                forces[i] -= 0.01 * (pos[i] - center)

            # Update positions
            new_pos = pos + forces * dt

            # Compute new score
            new_score = get_config_score(list(map(tuple, new_pos)), angles, outer_angle)

            # Simulated annealing acceptance
            if new_score < current_score:
                pos = new_pos
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_pos = pos.copy()
            else:
                # Accept with probability exp(-(new_score - current_score) / temperature)
                delta = new_score - current_score
                if delta < 100:  # Don't accept huge jumps
                    prob = math.exp(-delta / max(temperature, 1e-10))
                    if np.random.random() < prob:
                        pos = new_pos
                        current_score = new_score

            temperature *= cooling_rate

        return list(map(tuple, best_pos)), angles, outer_angle, best_score

    # Generate initial honeycomb layouts
    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for a hex grid with side length 1."""
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def generate_honeycomb_layouts():
        """Generate multiple honeycomb-like layouts for 12 hexagons."""
        layouts = []
        ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

        # Layout A: Center + Ring 1 + 5 from Ring 2
        layout_a = [(0, 0)] + list(ring1_dirs)
        ring2_candidates = []
        for (q, r) in ring1_dirs:
            for dq, dr in ring1_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in layout_a and (nq, nr) not in ring2_candidates:
                    ring2_candidates.append((nq, nr))
        layout_a.extend(ring2_candidates[:5])
        layouts.append(layout_a)

        # Layout B: 3 rows of 4
        layout_b = []
        for row in range(3):
            for col in range(4):
                q = col + (row % 2) * 0.5
                r = row
                layout_b.append((q, r))
        layouts.append(layout_b)

        # Layout C: 4 rows of 3
        layout_c = []
        for row in range(4):
            for col in range(3):
                q = col + (row % 2) * 0.5
                r = row
                layout_c.append((q, r))
        layouts.append(layout_c)

        # Layout D: 2 rows of 6
        layout_d = []
        for row in range(2):
            for col in range(6):
                q = col + (row % 2) * 0.5
                r = row
                layout_d.append((q, r))
        layouts.append(layout_d)

        # Layout E: 6 rows of 2
        layout_e = []
        for row in range(6):
            for col in range(2):
                q = col + (row % 2) * 0.5
                r = row
                layout_e.append((q, r))
        layouts.append(layout_e)

        return layouts

    def layout_to_positions(layout, rotation_deg=0.0):
        """Convert a honeycomb layout (axial coords) to Cartesian positions, centered."""
        positions = []
        for q, r in layout:
            x, y = axial_to_cartesian(q, r)
            positions.append((x, y))

        # Center the layout
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]

        # Apply rotation if needed
        if rotation_deg != 0:
            theta = math.radians(rotation_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rotated = []
            for x, y in positions:
                nx = x * cos_t - y * sin_t
                ny = x * sin_t + y * cos_t
                rotated.append((nx, ny))
            positions = rotated

        return positions

    # Main search
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    layouts = generate_honeycomb_layouts()

    for layout in layouts:
        if time.time() > deadline - 30:
            break

        # Try multiple rotations and outer angles
        for rot_deg in [0, 15, 30, 45, 60]:
            if time.time() > deadline - 15:
                break

            for outer_angle in [0, 30, 60]:
                if time.time() > deadline - 10:
                    break

                positions = layout_to_positions(layout, rotation_deg=rot_deg)
                angles = [0.0] * N

                score = get_config_score(positions, angles, outer_angle)

                if score < best_score:
                    best_score = score
                    best_positions = positions
                    best_angles = angles
                    best_outer_angle = outer_angle

                # If valid, try physics optimization
                if score < float('inf') and time.time() < deadline - 20:
                    phys_result = physics_optimize(positions, angles, outer_angle, max_iters=5000)
                    phys_pos, phys_ang, phys_oa, phys_score = phys_result
                    if phys_score < best_score:
                        best_score = phys_score
                        best_positions = phys_pos
                        best_angles = phys_ang
                        best_outer_angle = phys_oa

    # Fallback: If no valid configuration was found
    if best_positions is None or best_score == float('inf'):
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            x = c * 2.2 + (r % 2) * 1.1
            y = r * 1.9
            positions.append((x, y))
            angles.append(0.0)

        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(x - cx, y - cy) for x, y in positions]

        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, angles, 0.0)

    # Final cleanup: center the configuration
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0, 0)] * N
        best_angles = [0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]

    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
