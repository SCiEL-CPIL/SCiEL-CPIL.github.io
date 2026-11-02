from scipy.optimize import minimize, minimize_scalar
import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_all_vertices(positions, alpha_rad):
        """Returns (N*6, 2) array of all vertices."""
        verts_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            verts_list.append(verts)
        return np.vstack(verts_list)

    def min_enclosing_hex_side_and_params(all_verts, alpha_rad):
        """
        Computes the minimum side length of a regular hexagon enclosing all_verts,
        and returns the side length, optimal outer angle, and center.
        """
        def width_max(phi):
            # Normals for outer hexagon at angle phi
            angles = phi + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            # Project all points
            dots = all_verts @ normals.T
            # Width in each direction
            widths = dots.max(axis=0) - dots.min(axis=0)
            # The side length s is proportional to max width
            # s * sqrt(3) = max_width => s = max_width / sqrt(3)
            return np.max(widths)

        # Minimize the max width to find optimal phi
        res_phi = minimize_scalar(width_max, bounds=(0, np.pi/3), method='bounded',
                                  options={'xatol': 1e-8})
        best_phi = res_phi.x
        max_width = res_phi.fun
        side = max_width / np.sqrt(3)
        
        # Compute center
        angles = best_phi + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        
        # Solve normals.T @ center = mids
        # normals is 6x2, normals.T is 2x6
        # We want 2x1 center
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        center = tuple(c)
        
        return side, best_phi, center

    def make_objective(alpha_rad):
        def objective(params):
            if time.time() > deadline:
                return 1e10
            positions = params.reshape(N, 2)
            all_verts = get_all_vertices(positions, alpha_rad)
            side, _, _ = min_enclosing_hex_side_and_params(all_verts, alpha_rad)
            return side
        return objective

    def make_flower_init():
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    inits = [make_flower_init()]
    
    # Add a few perturbed inits
    inits.append(inits[0] + 0.1 * np.random.randn(*inits[0].shape))
    inits.append(inits[0] + 0.2 * np.random.randn(*inits[0].shape))

    best_side = np.inf
    best_positions = None
    best_alpha = 0.0
    best_theta = 0.0
    best_center = (0.0, 0.0)

    alphas = np.radians([0, 15, 30, 45]) # 0-60 is enough due to symmetry, sample 4 points

    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        for init_pos in inits:
            if time.time() > deadline - 5:
                break
            obj_func = make_objective(alpha)
            x0 = init_pos.flatten()
            
            try:
                res = minimize(
                    obj_func,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                pos = res.x.reshape(N, 2)
                # Recompute to get exact side and params
                all_verts = get_all_vertices(pos, alpha)
                side, theta, center = min_enclosing_hex_side_and_params(all_verts, alpha)
                
                if side < best_side:
                    best_side = side
                    best_positions = pos.copy()
                    best_alpha = alpha
                    best_theta = theta
                    best_center = center

            except Exception:
                pass

    if best_positions is None:
        # Fallback
        best_positions = inits[0]
        best_alpha = 0.0
        all_verts = get_all_vertices(best_positions, best_alpha)
        best_side, best_theta, best_center = min_enclosing_hex_side_and_params(all_verts, best_alpha)

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = best_center
    outer_side = best_side
    outer_angle_deg = np.degrees(best_theta)

    return inner, outer_center, outer_side, outer_angle_deg
