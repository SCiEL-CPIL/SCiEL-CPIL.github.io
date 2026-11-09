import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill

    sqrt3 = np.sqrt(3)

    # 12 unit hexagon centers in a compact hexagonal cluster
    # Axial coordinates (q, r) -> Cartesian for flat-top hexagons, side=1
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, -1), (2, -2), (1, -2), (-1, -2), (-2, -1)
    ]
    centers = np.array([
        [1.5 * q, sqrt3 * (r + q / 2.0)]
        for q, r in axial
    ])  # shape (12, 2)

    # Base vertices of a unit regular hexagon at angle 0
    base_angles = np.radians(np.arange(6) * 60)
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])  # (6, 2)

    def compute_s(theta, phi):
        """
        Compute the minimum side length s of a regular hexagon (centered at origin,
        orientation phi) that contains all 12 inner unit hexagons (rotated by theta).
        theta, phi in radians.
        """
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])

        # Rotated centers of inner hexagons
        rc = centers @ R.T  # (12, 2)

        # All 72 vertices: each inner hexagon contributes 6 vertices
        # verts[i, j] = rc[i] + R @ base_vertices[j]
        rotated_base = base_vertices @ R.T  # (6, 2)
        verts = rc[:, np.newaxis, :] + rotated_base[np.newaxis, :, :]  # (12, 6, 2)
        verts_flat = verts.reshape(-1, 2)  # (72, 2)

        # Outer hexagon outward edge normals
        outer_angles = phi + np.radians(np.arange(6) * 60)
        outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])  # (6, 2)

        # Max projection of any vertex onto any outer normal
        projs = verts_flat @ outer_normals.T  # (72, 6)
        max_proj = np.max(projs)

        # Apothem = s * sqrt(3)/2, so s = 2 * max_proj / sqrt(3)
        return (2.0 / sqrt3) * max_proj

    # --- Coarse vectorized grid search over (theta, phi) in [0, 60 deg) ---
    n_grid = 360
    theta_grid = np.linspace(0, np.radians(60), n_grid, endpoint=False)
    phi_grid = np.linspace(0, np.radians(60), n_grid, endpoint=False)

    best_s = float('inf')
    best_theta = 0.0
    best_phi = 0.0

    for theta in theta_grid:
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        rc = centers @ R.T  # (12, 2)
        rotated_base = base_vertices @ R.T  # (6, 2)
        verts = rc[:, np.newaxis, :] + rotated_base[np.newaxis, :, :]  # (12, 6, 2)
        verts_flat = verts.reshape(-1, 2)  # (72, 2)

        # For all phi in the grid simultaneously
        # outer_angles[k, m] = phi_grid[k] + m * 60 deg, for k in range(n_grid), m in 0..5
        outer_angles = phi_grid[:, np.newaxis] + np.radians(np.arange(6) * 60)[np.newaxis, :]  # (n_grid, 6)
        outer_normals = np.column_stack([np.cos(outer_angles).ravel(), np.sin(outer_angles).ravel()]).reshape(n_grid, 6, 2)

        # projs[k, i, m] = verts_flat[i] . outer_normals[k, m]
        # Use einsum: 'ij,kjm->kmi' -> then max over i,m
        projs = np.einsum('ij,kjm->kmi', verts_flat, outer_normals)  # (n_grid, 6, 72)
        max_projs = np.max(projs, axis=(1, 2))  # (n_grid,)
        s_vals = (2.0 / sqrt3) * max_projs

        idx = np.argmin(s_vals)
        if s_vals[idx] < best_s:
            best_s = s_vals[idx]
            best_theta = theta
            best_phi = phi_grid[idx]

        if time.time() > deadline - 120:
            break

    # --- Local refinement with Nelder-Mead ---
    def objective(x):
        return compute_s(x[0], x[1])

    x0 = [best_theta, best_phi]
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 100000,
                                   'adaptive': True})
        if result.fun < best_s and np.isfinite(result.fun):
            best_s = result.fun
            best_theta, best_phi = result.x
    except Exception:
        pass

    # --- Construct final packing ---
    theta, phi = best_theta, best_phi
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    rotated_centers = centers @ R.T

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = rotated_centers[i, 0]
        inner_hex_data[i, 1] = rotated_centers[i, 1]
        inner_hex_data[i, 2] = np.degrees(theta)

    outer_hex_data = np.array([0.0, 0.0, np.degrees(phi)])
    outer_hex_side_length = float(best_s)

    return inner_hex_data, outer_hex_data, outer_hex_side_length
