"""Combined program: discrete lattice topology search + continuous refinement for hexagon packing."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)

        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i

        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        """
        min_gap = float('inf')
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: DISCRETE LATTICE SEARCH ---
    # Generate candidate configurations on hexagonal lattices

    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])

    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # Try different lattice scales and rotations
    for scale_x in [0.92, 0.95, 0.98, 1.0, 1.02, 1.05]:
        for scale_y in [0.92, 0.95, 0.98, 1.0, 1.02, 1.05]:
            if time.time() > deadline - 60:
                break

            v1 = base_v1 * np.array([scale_x, 1.0])
            v2 = base_v2 * np.array([1.0, scale_y])

            # Generate lattice points
            points = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    p = i * v1 + j * v2
                    points.append(p)

            # Sort by distance from origin and take closest 12
            points.sort(key=lambda p: np.dot(p, p))
            selected = np.array(points[:12])

            # Center the cluster
            centroid = np.mean(selected, axis=0)
            selected -= centroid

            # Try different global rotations
            for rot_deg in [0, 15, 30, 45, 60, 75, 90]:
                if time.time() > deadline - 30:
                    break

                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated_selected = selected @ rot_mat.T

                positions = list(map(tuple, rotated_selected))
                angles = [0.0] * N

                # Check validity
                if is_valid(positions, angles):
                    # Try different outer hexagon angles
                    for outer_angle in [0, 15, 30, 45, 60, 75, 90]:
                        score = calc_min_outer_side(positions, angles, outer_angle)
                        if score < best_score:
                            best_score = score
                            best_positions = positions[:]
                            best_angles = angles[:]
                            best_outer_angle = outer_angle

    # --- PHASE 2: CONTINUOUS REFINEMENT ---
    # Use scipy.optimize.minimize to refine the best discrete configuration

    if best_positions is not None:
        def objective(params):
            if time.time() > deadline - 5:
                return 1e10
            pos = np.array(params[:2*N]).reshape(N, 2)
            ang_params = params[2*N:2*N+N]
            angles = list(ang_params * 30)  # Scale to degrees

            positions = list(map(tuple, pos))

            # Check validity with tolerance
            if not is_valid(positions, angles, tol=1e-3):
                return 1e6

            return calc_min_outer_side(positions, angles, best_outer_angle)

        # Initial parameters
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[2*i] = best_positions[i][0]
            init_params[2*i+1] = best_positions[i][1]
            init_params[2*N+i] = 0.0  # angles in units of 30 degrees

        # Run optimization
        try:
            result = minimize(objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_score and result.fun < 1e5:
                # Verify the result is valid
                pos = np.array(result.x[:2*N]).reshape(N, 2)
                ang_params = result.x[2*N:2*N+N]
                angles = list(ang_params * 30)
                positions = list(map(tuple, pos))

                if is_valid(positions, angles, tol=1e-4):
                    best_score = result.fun
                    best_positions = positions
                    best_angles = angles
        except Exception:
            pass

    # --- PHASE 3: LOCAL SEARCH WITH ANGLE VARIATION ---
    # Try varying individual hexagon angles for further improvement

    if best_positions is not None and time.time() < deadline - 10:
        for trial in range(20):
            if time.time() > deadline - 5:
                break

            # Start from current best
            positions = best_positions[:]
            angles = best_angles[:]

            # Randomly perturb a few angles
            for _ in range(3):
                idx = np.random.randint(0, N)
                angles[idx] = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])

            if is_valid(positions, angles, tol=1e-4):
                for outer_angle in [0, 15, 30, 45, 60, 75, 90]:
                    score = calc_min_outer_side(positions, angles, outer_angle)
                    if score < best_score:
                        best_score = score
                        best_angles = angles[:]
                        best_outer_angle = outer_angle

    # Fallback if no valid configuration found
    if best_positions is None:
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
        best_score = calc_min_outer_side(positions, angles, 0.0)

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
