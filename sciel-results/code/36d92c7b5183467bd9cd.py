"""Rewritten: sequential QP placement of hexagons to minimize container size."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = np.array([[math.cos(angle_rad + i * math.pi / 3),
                           math.sin(angle_rad + i * math.pi / 3)] for i in range(6)])
        return verts + np.array([cx, cy])

    def sat_overlap(v1, v2, tol=1e-9):
        """SAT overlap test for two convex hexagons (6x2 arrays). Returns True if overlapping."""
        for i in range(6):
            for pts in (v1, v2):
                edge = pts[(i + 1) % 6] - pts[i]
                normal = np.array([-edge[1], edge[0]])
                proj_a = np.dot(v1, normal)
                proj_b = np.dot(v2, normal)
                if proj_a.max() + tol < proj_b.min() or proj_b.max() + tol < proj_a.min():
                    return False
        return True

    def compute_outer_side(positions, angles, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        all_verts = np.vstack([hex_vertices(positions[i][0], positions[i][1], angles[i])
                               for i in range(N)])
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        return 2.0 * max_proj / math.sqrt(3)

    def score_config(positions, angles, outer_angle_deg):
        """Return outer side length if no overlaps, else inf."""
        verts = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap(verts[i], verts[j]):
                    return float('inf')
        return compute_outer_side(positions, angles, outer_angle_deg)

    def place_hexagon_sequentially(placed_positions, placed_angles, outer_angle_deg, 
                                    initial_pos=(0, 0), angle=0.0):
        """
        Place a new hexagon to maximize minimum clearance from already-placed hexagons
        and container boundary. Uses SLSQP to solve a small QP-like problem.
        """
        n_placed = len(placed_positions)
        
        def objective(x):
            pos = (x[0], x[1])
            ang = x[2]
            new_vert = hex_vertices(pos[0], pos[1], ang)
            
            # Overlap penalty
            overlap_pen = 0.0
            for i in range(n_placed):
                if sat_overlap(new_vert, hex_vertices(placed_positions[i][0], 
                                                     placed_positions[i][1], 
                                                     placed_angles[i])):
                    # Estimate penetration
                    d = np.array(pos) - np.array(placed_positions[i])
                    dist = np.linalg.norm(d)
                    overlap_pen += max(0.0, 1.7 - dist) ** 2
            
            # Container containment penalty
            theta = math.radians(outer_angle_deg)
            max_proj = 0.0
            for k in range(6):
                a = theta + k * math.pi / 3
                normal = np.array([math.cos(a), math.sin(a)])
                proj = np.dot(new_vert, normal)
                max_proj = max(max_proj, proj.max())
            container_side = 2.0 * max_proj / math.sqrt(3)
            
            # We want to minimize the required container size
            return container_side + 1000.0 * overlap_pen

        # Initial guess
        x0 = np.array([initial_pos[0], initial_pos[1], angle])
        
        try:
            result = minimize(objective, x0, method='SLSQP', 
                             options={'maxiter': 50, 'ftol': 1e-6})
            x_final = result.x
            return (x_final[0], x_final[1]), x_final[2]
        except Exception:
            return initial_pos, angle

    def generate_initial_layouts():
        """Generate diverse initial layouts."""
        layouts = []

        # Layout 1: Honeycomb ring (center + 6 + 5 from ring 2)
        ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        layout = [(0, 0)]
        layout.extend(ring1_dirs)
        # Ring 2 candidates
        ring2 = []
        for (q, r) in ring1_dirs:
            for dq, dr in ring1_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in layout and (nq, nr) not in ring2:
                    ring2.append((nq, nr))
        layout.extend(ring2[:5])
        layouts.append(layout)

        # Layout 2: 3x4 staggered grid
        layout2 = []
        for row in range(3):
            for col in range(4):
                q = col + (0.5 if row % 2 else 0)
                r = row
                layout2.append((q, r))
        layouts.append(layout2)

        # Layout 3: 4x3 staggered
        layout3 = []
        for row in range(4):
            for col in range(3):
                q = col + (0.5 if row % 2 else 0)
                r = row
                layout3.append((q, r))
        layouts.append(layout3)

        # Layout 4: 2x6
        layout4 = []
        for row in range(2):
            for col in range(6):
                q = col + (0.5 if row % 2 else 0)
                r = row
                layout4.append((q, r))
        layouts.append(layout4)

        # Layout 5: Circular arrangement
        layout5 = []
        for i in range(N):
            a = 2 * math.pi * i / N
            layout5.append((math.cos(a) * 2, math.sin(a) * 2))
        layouts.append(layout5)

        # Layout 6: Two concentric rings (6+6, no center)
        layout6 = []
        for i in range(6):
            a = 2 * math.pi * i / 6
            layout6.append((math.cos(a) * 1.5, math.sin(a) * 1.5))
        for i in range(6):
            a = 2 * math.pi * i / 6 + math.pi / 6
            layout6.append((math.cos(a) * 3, math.sin(a) * 3))
        layouts.append(layout6)

        return layouts

    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def layout_to_positions(layout, rotation_deg=0.0, scale=1.0):
        """Convert layout to Cartesian positions, centered and optionally rotated/scaled."""
        positions = []
        for item in layout:
            if len(item) == 2 and all(isinstance(v, (int, float)) for v in item):
                # Treat as axial coordinates
                x, y = axial_to_cartesian(item[0], item[1])
                positions.append((x, y))
            else:
                positions.append(item)

        # Center
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]

        # Scale
        if scale != 1.0:
            positions = [(x * scale, y * scale) for x, y in positions]

        # Rotate
        if rotation_deg != 0:
            theta = math.radians(rotation_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            positions = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
                         for x, y in positions]

        return positions

    # Main search
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    layouts = generate_initial_layouts()
    rotations = [0, 15, 30, 45, 60, 75, 90, 120, 135, 150]
    outer_angles = [0, 30, 60, 90, 120, 150]

    for layout in layouts:
        if time.time() > deadline - 20:
            break

        for rot in rotations:
            if time.time() > deadline - 10:
                break

            for oa in outer_angles:
                if time.time() > deadline - 5:
                    break

                positions = layout_to_positions(layout, rotation_deg=rot)
                angles = [0.0] * N

                score = score_config(positions, angles, oa)
                if score < best_score:
                    best_score = score
                    best_positions = positions[:]
                    best_angles = angles[:]
                    best_outer_angle = oa

                # If valid, try sequential refinement
                if score < float('inf') and time.time() < deadline - 15:
                    refined_positions = positions[:]
                    refined_angles = angles[:]
                    
                    # Try to refine by re-placing each hexagon sequentially
                    for i in range(N):
                        if time.time() > deadline - 5:
                            break
                        
                        # Place hexagon i considering all others
                        placed_pos = [refined_positions[j] for j in range(N) if j != i]
                        placed_ang = [refined_angles[j] for j in range(N) if j != i]
                        
                        # Initial guess is current position
                        new_pos, new_ang = place_hexagon_sequentially(
                            placed_pos, placed_ang, oa,
                            initial_pos=refined_positions[i],
                            angle=refined_angles[i]
                        )
                        
                        # Check if new placement is valid
                        test_positions = refined_positions[:]
                        test_angles = refined_angles[:]
                        test_positions[i] = new_pos
                        test_angles[i] = new_ang
                        
                        test_score = score_config(test_positions, test_angles, oa)
                        if test_score < float('inf'):
                            refined_positions[i] = new_pos
                            refined_angles[i] = new_ang
                    
                    refined_score = score_config(refined_positions, refined_angles, oa)
                    if refined_score < best_score:
                        best_score = refined_score
                        best_positions = refined_positions
                        best_angles = refined_angles
                        best_outer_angle = oa

    # Fallback
    if best_positions is None or best_score == float('inf'):
        positions = []
        for i in range(N):
            r = i // 4
            c = i % 4
            x = c * 2.2 + (r % 2) * 1.1
            y = r * 1.9
            positions.append((x, y))
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(x - cx, y - cy) for x, y in positions]
        best_positions = positions
        best_angles = [0.0] * N
        best_outer_angle = 0.0
        best_score = compute_outer_side(positions, best_angles, 0.0)

    # Center final configuration
    cx = sum(p[0] for p in best_positions) / N
    cy = sum(p[1] for p in best_positions) / N
    final_positions = [(x - cx, y - cy) for x, y in best_positions]

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]

    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
