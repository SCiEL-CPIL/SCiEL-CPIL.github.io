import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices (flat-top orientation)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
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
        normals = np.array(normals)  # (6, 2)

        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i

        # s = inradius / (sqrt(3)/2) = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2, tol=1e-9):
        """
        Fast SAT check for disjointness.
        Returns True if disjoint (or touching within tol), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)

        axes = []
        for i in range(6):
            edge1 = poly1[(i + 1) % 6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i + 1) % 6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)

        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 - tol or max2 < min1 - tol:
                return True

        return False

    def check_all_disjoint(polys, tol=1e-9):
        for i in range(N):
            for j in range(i + 1, N):
                if not check_disjoint(polys[i], polys[j], tol):
                    return False
        return True

    # Generate honeycomb lattice points
    sqrt3 = math.sqrt(3)
    lattice_points = []
    max_coord = 6
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            lattice_points.append((x, y, q, r))

    # Sort by distance from origin to prioritize central clusters
    lattice_points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)

    best_side = float('inf')
    best_positions = None
    best_angles = None

    # Strategy:
    # 1. Generate random valid configurations on the lattice.
    # 2. For each, compute the bounding box side.
    # 3. Keep the best few.
    # 4. Run Nelder-Mead on the positions of the best configuration to squeeze.

    candidates = []

    # Sample 500 random subsets of 12 from the first 40 lattice points
    # This is fast enough.
    for _ in range(500):
        if time.time() > deadline - 30:
            break

        # Randomly pick 12 indices from 0 to 39
        indices = np.random.choice(40, size=12, replace=False)
        indices = sorted(indices)

        points = [lattice_points[i] for i in indices]
        positions = np.array([[p[0], p[1]] for p in points])
        angles = np.zeros(12)

        side = calc_min_outer_side(positions, angles, 0.0)
        candidates.append((side, positions, angles))

    # Sort by side length
    candidates.sort(key=lambda x: x[0])

    # Take the top 5 candidates and refine them
    for side, positions, angles in candidates[:5]:
        if time.time() > deadline - 10:
            break

        # Refinement: Optimize positions only.
        # We keep angles fixed at 0 for now to maintain lattice alignment benefits.
        # If we allow angle changes, we need to re-check disjointness, which is expensive.
        # Let's try optimizing positions with a constraint that they stay close to the lattice points?
        # Or just free optimize positions, but if they overlap, penalize.

        def objective(params):
            pos = params.reshape(12, 2)
            # Simple penalty for overlap
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, 0.0) for i in range(12)]
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    if not check_disjoint(polys[i], polys[j]):
                        penalty += 10.0
            s = calc_min_outer_side(pos, angles, 0.0)
            return s + penalty

        res = minimize(objective, positions.flatten(), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})

        if res.fun < side + 1e-6:  # Improved or same
            new_pos = res.x.reshape(12, 2)
            # Verify validity
            polys = [get_hex_verts(new_pos[i][0], new_pos[i][1], 1.0, 0.0) for i in range(12)]
            if check_all_disjoint(polys):
                new_side = calc_min_outer_side(new_pos, angles, 0.0)
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = angles

    # If we didn't find anything (shouldn't happen), fallback to a simple grid
    if best_positions is None:
        # Use the first candidate
        side, positions, angles = candidates[0]
        best_side = side
        best_positions = positions
        best_angles = angles

    # Final check and return
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])

    # The outer hexagon is centered at (0,0) and aligned with 0 degrees
    return inner, (0.0, 0.0), best_side, 0.0
