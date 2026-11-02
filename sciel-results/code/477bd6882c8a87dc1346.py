"""
Strategy: Symmetry-constrained structural search.
Instead of 36 free DOF, we fix the 12 inner unit hexagons into a canonical
hexagonal cluster: 1 at the center, 6 in a first ring at radius r1, and 5 in
a second ring at radius r2 (one slot of the 6-fold second ring is removed to
make 12 total). This preserves the natural hexagonal packing geometry while
reducing the problem to 6 continuous parameters: (r1, r2, ring1_rot, ring2_rot,
cluster_rot, outer_rot). We then optimize these 6 parameters with a fast,
deterministic multi-start Nelder-Mead optimizer followed by a short SA polish.
This structural prior lets the optimizer explore the 6D landscape exhaustively
within the time budget, whereas the 36D SA in the parent program wanders.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0  # apothem of unit hexagon

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (kill at 150s)

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        # Normals to the sides of a regular hexagon centered at origin
        # Sides are at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if overlapping.
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def inner_positions_from_params(params):
        """
        params: [r1, r2, ring1_rot_deg, ring2_rot_deg, cluster_rot_deg, outer_rot_deg]
        Returns list of (x, y, angle_deg) for 12 inner hexagons (all at 0 angle relative to world,
        but we can also rotate each inner hexagon with the cluster).
        """
        r1 = params[0]
        r2 = params[1]
        ring1_rot = params[2]
        ring2_rot = params[3]
        cluster_rot = params[4]
        # outer_rot = params[5]  (used separately)

        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = math.radians(60 * k + ring1_rot + cluster_rot)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            # Rotate each inner hexagon to align with the cluster
            positions.append((x, y, cluster_rot))
        # Ring 2: 5 hexagons (skip one slot for 12 total)
        for k in range(5):
            angle = math.radians(60 * k + 30 + ring2_rot + cluster_rot)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            positions.append((x, y, cluster_rot))
        return positions

    def min_outer_side(inner_positions, outer_angle_deg, outer_center=(0.0, 0.0)):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        if not all_verts:
            return 0.0
        all_verts = np.vstack(all_verts)

        # Normals for the outer hexagon at angle outer_angle_deg
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad

        # Calculate max projection relative to outer center
        max_proj = 0.0
        for na in normals:
            nx = math.cos(na)
            ny = math.sin(na)
            projs = (all_verts[:, 0] - outer_center[0]) * nx + (all_verts[:, 1] - outer_center[1]) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p

        # S * sqrt(3)/2 >= max_proj  => S >= max_proj / (sqrt(3)/2)
        return max_proj / (SQRT3 / 2)

    def objective(params):
        """
        params: [r1, r2, ring1_rot_deg, ring2_rot_deg, cluster_rot_deg, outer_rot_deg]
        Returns the side length of the minimal enclosing outer hexagon, plus penalties.
        """
        r1 = params[0]
        r2 = params[1]
        ring1_rot = params[2]
        ring2_rot = params[3]
        cluster_rot = params[4]
        outer_rot = params[5]

        # Ensure radii are positive and reasonable
        if r1 < 1.5 or r2 < 2.5:
            return 100.0 + (1.5 - r1) * 10 + (2.5 - r2) * 10

        inner_positions = inner_positions_from_params(params)

        # Check disjointness
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    penalty += 10.0

        s = min_outer_side(inner_positions, outer_rot, (0.0, 0.0))

        return s + penalty

    # Initial guesses for multi-start
    np.random.seed(42)

    best_params = None
    best_score = float('inf')

    # Multi-start Nelder-Mead
    initial_guesses = [
        [2.0, 3.5, 0.0, 0.0, 0.0, 0.0],
        [2.0, 3.5, 30.0, 0.0, 0.0, 0.0],
        [2.0, 3.5, 0.0, 30.0, 0.0, 0.0],
        [2.0, 3.5, 0.0, 0.0, 30.0, 0.0],
        [2.0, 3.5, 0.0, 0.0, 0.0, 30.0],
        [2.1, 3.6, 15.0, 15.0, 15.0, 15.0],
        [1.9, 3.4, -15.0, -15.0, -15.0, -15.0],
        [2.0, 3.5, 0.0, 0.0, 0.0, 0.0],  # baseline
    ]

    for guess in initial_guesses:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If no good result yet, use the baseline
    if best_params is None:
        best_params = np.array([2.0, 3.5, 0.0, 0.0, 0.0, 0.0])
        best_score = objective(best_params)

    # Short SA polish around the best
    if time.time() < deadline - 10:
        current_params = best_params.copy()
        current_score = best_score
        T = 0.1
        cooling = 0.95
        max_iters = 500
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            new_params = current_params.copy()
            new_params[0] += np.random.uniform(-0.1, 0.1)  # r1
            new_params[1] += np.random.uniform(-0.1, 0.1)  # r2
            new_params[2] += np.random.uniform(-10, 10)     # ring1_rot
            new_params[3] += np.random.uniform(-10, 10)     # ring2_rot
            new_params[4] += np.random.uniform(-10, 10)     # cluster_rot
            new_params[5] += np.random.uniform(-10, 10)     # outer_rot

            new_score = objective(new_params)
            if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            T *= cooling
            if T < 0.001:
                T = 0.001

    # Extract best inner positions
    inner = inner_positions_from_params(best_params)

    # Compute final outer side and angle
    outer_rot = best_params[5]
    s = min_outer_side(inner, outer_rot, (0.0, 0.0))

    center = (0.0, 0.0)
    side = s
    angle = outer_rot

    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break

    if not valid:
        side *= 1.01

    return inner, center, side, angle
