import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        # verts: 6x2
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        # v1, v2: 6x2
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return max(0.0, min_overlap)

    def inside_violation(inner_verts, outer_verts):
        # inner_verts: 6x2, outer_verts: 6x2
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def compute_minimal_enclosing_hexagon(centers, angles, side=1.0):
        """
        Compute the side length and angle of the minimal regular hexagon enclosing the set of hexagons.
        centers: Nx2
        angles: N
        side: side length of inner hexagons
        Returns: (side_length, angle_rad, center_x, center_y)
        """
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i in range(len(centers)):
            verts = hex_vertices(centers[i, 0], centers[i, 1], side, angles[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # The minimal enclosing regular hexagon is determined by its orientation.
        # For a given orientation theta, the side length S is determined by the max width in the 6 normal directions.
        # Width w_k = max(v dot n_k) - min(v dot n_k)
        # S = max_k(w_k) / sqrt(3)  (since width of regular hexagon side S is S*sqrt(3))
        # Wait, the width of a regular hexagon with side S is S * sqrt(3) if measured between opposite sides.
        # The distance from center to vertex is S.
        # The distance from center to side (apothem) is S * sqrt(3) / 2.
        # So width = 2 * apothem = S * sqrt(3).
        # So S = width / sqrt(3).
        
        def calc_side(theta):
            angles_h = theta + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles_h), np.sin(angles_h)])
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            return max_width / np.sqrt(3)
        
        # Coarse search for theta
        thetas_coarse = np.linspace(0, np.pi / 3, 180)
        sides = np.array([calc_side(th) for th in thetas_coarse])
        best_idx = np.argmin(sides)
        
        # Fine search
        lo = thetas_coarse[max(0, best_idx - 2)]
        hi = thetas_coarse[min(len(thetas_coarse) - 1, best_idx + 2)]
        
        res = minimize_scalar(calc_side, bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
        best_theta = res.x
        best_side = res.fun
        
        # Center is the midpoint of the bounding box in the optimal orientation
        angles_h = best_theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles_h), np.sin(angles_h)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        
        # Reconstruct center from projections?
        # Actually, the center of the enclosing hexagon is the average of the center of the bounding box in each normal direction?
        # No, for a regular hexagon, the center is unique.
        # The center (cx, cy) must satisfy:
        # cx * n_kx + cy * n_ky + apothem = max(v dot n_k)
        # cx * n_kx + cy * n_ky - apothem = min(v dot n_k)
        # So cx * n_kx + cy * n_ky = (max + min) / 2
        # We have 6 equations, 2 unknowns. Least squares.
        
        rhs = (maxs + mins) / 2
        # normals is 6x2
        # Solve normals @ [cx, cy]^T = rhs
        # Using least squares
        center, _, _, _ = np.linalg.lstsq(normals, rhs, rcond=None)
        
        return best_side, best_theta, center[0], center[1]

    def objective(params):
        if time.time() > deadline:
            return 1e10

        # params: [x1, y1, a1, ..., x12, y12, a12]
        # 3 params per hexagon
        positions = params[:2 * N].reshape(N, 2)
        angles = params[2 * N:3 * N]
        
        # Compute minimal enclosing hexagon
        s, theta, cx, cy = compute_minimal_enclosing_hexagon(positions, angles, 1.0)
        
        # Penalty for overlaps
        total_penalty = 0.0
        inner_verts_list = []
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            inner_verts_list.append(verts)
            
        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd
        
        return s + 10.0 * total_penalty

    def make_hex_config(d, rot=0.0):
        pos = []
        ang = []
        pos.append([0.0, 0.0])
        ang.append(rot)
        for k in range(6):
            a = np.radians(60 * k) + rot
            pos.append([d * np.cos(a), d * np.sin(a)])
            ang.append(rot)
        for k in range(5):
            a = np.radians(60 * k) + rot
            pos.append([2 * d * np.cos(a), 2 * d * np.sin(a)])
            ang.append(rot)
        return np.array(pos), np.array(ang)

    def make_staggered_config(d, rot=0.0):
        pos = []
        ang = []
        for row in range(3):
            for col in range(4):
                x = col * d
                y = row * d * np.sqrt(3)
                if row % 2 == 1:
                    x += d / 2
                pos.append([x, y])
                ang.append(rot)
        return np.array(pos), np.array(ang)

    # Generate diverse initial configurations
    configs = []
    configs.append(make_hex_config(2.1))
    configs.append(make_hex_config(2.1, rot=np.pi / 6))
    configs.append(make_hex_config(2.2))
    configs.append(make_hex_config(2.3))
    configs.append(make_staggered_config(2.1))
    configs.append(make_staggered_config(2.2))
    # Add a random config
    pos = np.random.randn(N, 2) * 2.0
    ang = np.random.rand(N) * np.pi / 3
    configs.append((pos, ang))

    best_obj = 1e10
    best_params = None

    for pos, ang in configs:
        if time.time() > deadline:
            break

        params = np.concatenate([pos.flatten(), ang])

        try:
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Refine best
    if best_params is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Extract solution
    if best_params is not None:
        final_pos = best_params[:2 * N].reshape(N, 2)
        final_angles = best_params[2 * N:3 * N]
        
        # Compute final enclosing hexagon
        s, theta, cx, cy = compute_minimal_enclosing_hexagon(final_pos, final_angles, 1.0)
        
        inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_angles[i])] for i in range(N)]
        outer_center = (cx, cy)
        outer_side = s
        outer_angle_deg = np.degrees(theta)
        
        return inner, outer_center, outer_side, outer_angle_deg
    else:
        # Fallback
        pos, ang = make_hex_config(2.5)
        s, theta, cx, cy = compute_minimal_enclosing_hexagon(pos, ang, 1.0)
        inner = [[pos[i, 0], pos[i, 1], np.degrees(ang[i])] for i in range(N)]
        outer_center = (cx, cy)
        outer_side = s
        outer_angle_deg = np.degrees(theta)
        return inner, outer_center, outer_side, outer_angle_deg
