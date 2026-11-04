import numpy as np
import time
import scipy.optimize

N = 11
S3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = 145.0

    # 11-hexagonal rigid cluster based on triangular lattice
    # Lattice vectors for unit hexagons (side 1, vertex at 0 deg):
    # u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    u = np.array([1.5, S3/2])
    v = np.array([0.0, S3])

    # Lattice coordinates (i, j) for the 11 closest points to origin
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]

    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    # Scale up to ensure strict disjointness (touching is not disjoint)
    cluster_centers *= 1.001

    # Pre-compute all 66 vertices (11 hexagons × 6 vertices) at angle=0
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, S3/2],
        [-0.5, S3/2],
        [-1.0, 0.0],
        [-0.5, -S3/2],
        [0.5, -S3/2]
    ])

    # All vertices for each hexagon (relative to its center)
    # Shape: (N, 6, 2)
    all_base_verts = np.tile(base_verts, (N, 1, 1))

    # For each hexagon i, its vertices in cluster frame (before global rotation)
    # verts_i = cluster_centers[i] + all_base_verts[i]
    # Shape: (N, 6, 2)
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]

    # Flatten to (66, 2)
    flat_verts = cluster_verts.reshape(-1, 2)

    # Pre-compute normals for outer hexagon at various angles
    # We'll optimize over both cluster rotation theta and outer hex rotation phi

    def compute_side_from_angles(theta_deg, phi_deg):
        """Compute the minimum outer hexagon side length for given cluster rotation and outer hex rotation."""
        theta_rad = np.deg2rad(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])

        # Rotate all vertices
        rotated = flat_verts @ R.T

        # For the outer hexagon with rotation angle phi
        phi_rad = np.deg2rad(phi_deg)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        projs = rotated @ normals.T  # (66, 6)
        max_proj = np.max(projs)
        L = 2.0 * max_proj / S3

        return L

    # Phase 1: Coarse grid search over both angles
    # The cluster has 6-fold symmetry, so theta period is 60 degrees
    # The outer hexagon also has 6-fold symmetry, so phi period is 60 degrees
    
    best_theta = 0.0
    best_phi = 0.0
    best_side = float('inf')

    # Coarse grid: 1 degree resolution
    theta_coarse = np.linspace(0, 60, 61, endpoint=False)
    phi_coarse = np.linspace(0, 60, 61, endpoint=False)

    for theta in theta_coarse:
        if time.time() > deadline - 30:
            break
        for phi in phi_coarse:
            side = compute_side_from_angles(theta, phi)
            if side < best_side:
                best_side = side
                best_theta = theta
                best_phi = phi

    if time.time() > deadline - 20:
        pass  # Use best found so far

    # Phase 2: Fine grid around best
    fine_range = 1.0  # degrees
    theta_fine = np.linspace(best_theta - fine_range, best_theta + fine_range, 201)
    phi_fine = np.linspace(best_phi - fine_range, best_phi + fine_range, 201)

    for theta in theta_fine:
        if time.time() > deadline - 15:
            break
        for phi in phi_fine:
            side = compute_side_from_angles(theta, phi)
            if side < best_side:
                best_side = side
                best_theta = theta
                best_phi = phi

    if time.time() > deadline - 10:
        pass

    # Phase 3: Local optimization with scipy Nelder-Mead
    if time.time() < deadline - 5:
        def obj(params):
            return compute_side_from_angles(params[0], params[1])

        try:
            res = scipy.optimize.minimize(
                obj,
                np.array([best_theta, best_phi]),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if 0 <= res.x[0] < 60 and 0 <= res.x[1] < 60:
                side_res = compute_side_from_angles(res.x[0], res.x[1])
                if side_res < best_side:
                    best_theta = float(res.x[0])
                    best_phi = float(res.x[1])
                    best_side = side_res
        except Exception:
            pass

    # Phase 4: High-precision polish pass
    if time.time() < deadline - 2:
        def obj_high_prec(params):
            return compute_side_from_angles(params[0], params[1])

        try:
            res = scipy.optimize.minimize(
                obj_high_prec,
                np.array([best_theta, best_phi]),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if 0 <= res.x[0] < 60 and 0 <= res.x[1] < 60:
                side_res = compute_side_from_angles(res.x[0], res.x[1])
                if side_res < best_side:
                    best_theta = float(res.x[0])
                    best_phi = float(res.x[1])
                    best_side = side_res
        except Exception:
            pass

    # Build the inner hexagon positions
    # Each inner hexagon is at cluster_centers[i] rotated by best_theta, with its own angle = best_theta
    angle_rad = np.deg2rad(best_theta)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])

    rotated_centers = cluster_centers @ R.T

    inner = []
    for i in range(N):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        ang = float(best_theta)
        inner.append([x, y, ang])

    return inner, (0.0, 0.0), float(best_side), float(best_phi)
