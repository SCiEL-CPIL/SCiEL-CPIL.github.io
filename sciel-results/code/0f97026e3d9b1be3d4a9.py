import numpy as np
from scipy.optimize import minimize_scalar, minimize
import time


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 600s limit

    sqrt3 = np.sqrt(3)

    # Axial coordinates for 12 hexes in a compact hexagonal arrangement
    axial_coords = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1],
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1], [-2, 0]
    ])

    def get_centers(d):
        q = axial_coords[:, 0]
        r = axial_coords[:, 1]
        x = d * (q + r / 2)
        y = d * r * sqrt3 / 2
        return np.column_stack([x, y])

    def compute_min_s(d, centers=None):
        """Compute minimum outer hex side s for given centers."""
        if centers is None:
            centers = get_centers(d)
        
        thetas = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        max_val = 0.0
        for cx, cy in centers:
            for i in range(6):
                angle_i = i * np.pi / 3
                vx = cx + np.cos(angle_i)
                vy = cy + np.sin(angle_i)
                for j in range(6):
                    val = vx * cos_t[j] + vy * sin_t[j]
                    if val > max_val:
                        max_val = val
        
        s = max_val / (sqrt3 / 2)
        return s

    def check_no_overlap(centers, tol=1e-6):
        """Check if any two inner hexes overlap using SAT."""
        for i in range(12):
            for j in range(i + 1, 12):
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                
                v1 = np.array([[cx1 + np.cos(k * np.pi / 3), cy1 + np.sin(k * np.pi / 3)] for k in range(6)])
                v2 = np.array([[cx2 + np.cos(k * np.pi / 3), cy2 + np.sin(k * np.pi / 3)] for k in range(6)])
                
                all_normals = []
                for v in [v1, v2]:
                    for k in range(6):
                        p1 = v[k]
                        p2 = v[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-10:
                            normal = normal / norm
                            all_normals.append(normal)
                
                for normal in all_normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                        break
                else:
                    return False
        return True

    def full_objective(params):
        """Objective: minimize s with constraints."""
        if len(params) != 25:
            return 1e6
        centers = params[:24].reshape(12, 2)
        s = params[24]
        
        if s < sqrt3:
            return 1e6
        
        # Check all inner hexes fit in outer hex
        thetas = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        inradius = s * sqrt3 / 2
        
        for i in range(12):
            cx, cy = centers[i]
            for vi in range(6):
                angle_vi = vi * np.pi / 3
                vx = cx + np.cos(angle_vi)
                vy = cy + np.sin(angle_vi)
                for j in range(6):
                    val = vx * cos_t[j] + vy * sin_t[j]
                    if val > inradius + 1e-6:
                        return 1e6 + (val - inradius)
        
        # Check no overlaps
        if not check_no_overlap(centers):
            return 1e6 + 1
        
        return s

    # Step 1: Optimize lattice spacing d
    d_min = sqrt3  # minimum for no overlap
    d_max = sqrt3 * 3  # upper bound

    def lattice_objective(d):
        if d < d_min * 0.99:
            return 1e6
        return compute_min_s(d)

    result_lattice = minimize_scalar(lattice_objective, bounds=(d_min, d_max), method='bounded',
                                     options={'xatol': 1e-6})
    d_opt = result_lattice.x
    s_lattice = result_lattice.fun
    centers_lattice = get_centers(d_opt)

    # Time check
    if time.time() - start_time > deadline - 60:
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i] = [centers_lattice[i, 0], centers_lattice[i, 1], 0]
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, s_lattice

    # Step 2: Free-position refinement
    params_init = np.concatenate([centers_lattice.flatten(), [s_lattice]])

    try:
        result_free = minimize(full_objective, params_init, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-5,
                                        'adaptive': True})
    except Exception:
        result_free = None

    if time.time() - start_time > deadline - 15:
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i] = [centers_lattice[i, 0], centers_lattice[i, 1], 0]
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, s_lattice

    if result_free is not None and result_free.fun < s_lattice and result_free.fun < 1e5:
        params_best = result_free.x
        centers_best = params_best[:24].reshape(12, 2)
        s_best = params_best[24]
    else:
        centers_best = centers_lattice
        s_best = s_lattice

    # Final validation
    if not check_no_overlap(centers_best):
        centers_best = centers_lattice
        s_best = s_lattice

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [centers_best[i, 0], centers_best[i, 1], 0]
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, s_best
