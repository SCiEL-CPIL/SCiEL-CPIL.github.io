import numpy as np
import time

N = 11
S3 = np.sqrt(3.0)

NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C, A):
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def exact_nonoverlap(C, A, eps=1e-9):
    for i in range(N):
        for j in range(i + 1, N):
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, A, S, eps=1e-9):
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S)


def build_rigid_cluster(scale_factor=1.002):
    d = S3
    u = np.array([1.5, d / 2])
    v = np.array([0.0, d])

    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]

    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    cluster_centers *= scale_factor

    all_base_verts = np.tile(UNIT, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)

    return cluster_centers, flat_verts


def compute_side(flat_verts, cluster_angle_deg, phi_deg):
    """
    Compute min outer hexagon side for given cluster rotation and outer hexagon angle.
    Both angles in degrees.
    """
    angle_rad = np.deg2rad(cluster_angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    rotated = flat_verts @ R.T

    phi_rad = np.deg2rad(phi_deg)
    normals = np.array([
        [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
        for k in range(6)
    ])
    projs = rotated @ normals.T
    max_proj = np.max(projs)
    return 2.0 * max_proj / S3


def optimize_rigid_cluster(flat_verts, deadline):
    """
    Exhaustively optimize the rigid cluster by searching over global rotation angle
    and outer hexagon angle. Uses a combination of coarse grid search, fine grid search,
    golden section search, and Nelder-Mead for high-precision optimization.
    Returns (best_cluster_angle, best_phi, best_side).
    """
    start = time.time()

    # Phase 1: Coarse grid search over cluster angle (0 to 60 degrees)
    # and outer hexagon angle (0 to 60 degrees)
    # Since the cluster is asymmetric, we search the full 60-degree range
    angles_coarse = np.linspace(0, 60, 361, endpoint=False)
    phis_coarse = np.linspace(0, 60, 361, endpoint=False)

    best_side = float('inf')
    best_angle = 0.0
    best_phi = 0.0

    for a in angles_coarse:
        for p in phis_coarse:
            s = compute_side(flat_verts, a, p)
            if s < best_side:
                best_side = s
                best_angle = a
                best_phi = p

    if time.time() > deadline - 20:
        return best_angle, best_phi, best_side

    # Phase 2: Fine grid search around the best coarse result
    fine_range = 3.0  # degrees
    angles_fine = np.linspace(best_angle - fine_range, best_angle + fine_range, 121)
    phis_fine = np.linspace(best_phi - fine_range, best_phi + fine_range, 121)

    for a in angles_fine:
        for p in phis_fine:
            s = compute_side(flat_verts, a, p)
            if s < best_side:
                best_side = s
                best_angle = a
                best_phi = p

    if time.time() > deadline - 10:
        return best_angle, best_phi, best_side

    # Phase 3: Golden section search on cluster angle (fixing phi)
    def obj_angle(a_val):
        return compute_side(flat_verts, a_val, best_phi)

    a_center = best_angle
    a_range = 1.5
    a_lower = max(0, a_center - a_range)
    a_upper = min(60, a_center + a_range)

    gr = (np.sqrt(5) - 1) / 2
    c_val = a_upper - gr * (a_upper - a_lower)
    d_val = a_lower + gr * (a_upper - a_lower)
    fc = obj_angle(c_val)
    fd = obj_angle(d_val)

    for _ in range(100):
        if a_upper - a_lower < 1e-8:
            break
        if fc < fd:
            a_upper = d_val
            d_val = c_val
            fd = fc
            c_val = a_upper - gr * (a_upper - a_lower)
            fc = obj_angle(c_val)
        else:
            a_lower = c_val
            c_val = d_val
            fc = fd
            d_val = a_lower + gr * (a_upper - a_lower)
            fd = obj_angle(d_val)

    best_a = (a_lower + a_upper) / 2
    best_s_a = obj_angle(best_a)
    if best_s_a < best_side:
        best_angle = best_a
        best_side = best_s_a

    if time.time() > deadline - 5:
        return best_angle, best_phi, best_side

    # Phase 4: Golden section search on phi (fixing cluster angle)
    def obj_phi(p_val):
        return compute_side(flat_verts, best_angle, p_val)

    p_center = best_phi
    p_range = 1.5
    p_lower = max(0, p_center - p_range)
    p_upper = min(60, p_center + p_range)

    c_val = p_upper - gr * (p_upper - p_lower)
    d_val = p_lower + gr * (p_upper - p_lower)
    fc = obj_phi(c_val)
    fd = obj_phi(d_val)

    for _ in range(100):
        if p_upper - p_lower < 1e-8:
            break
        if fc < fd:
            p_upper = d_val
            d_val = c_val
            fd = fc
            c_val = p_upper - gr * (p_upper - p_lower)
            fc = obj_phi(c_val)
        else:
            p_lower = c_val
            c_val = d_val
            fc = fd
            d_val = p_lower + gr * (p_upper - p_lower)
            fd = obj_phi(d_val)

    best_p = (p_lower + p_upper) / 2
    best_s_p = obj_phi(best_p)
    if best_s_p < best_side:
        best_phi = best_p
        best_side = best_s_p

    if time.time() > deadline - 2:
        return best_angle, best_phi, best_side

    # Phase 5: Nelder-Mead polish on both angles simultaneously
    from scipy.optimize import minimize

    def polish_obj(theta):
        return compute_side(flat_verts, theta[0], theta[1])

    try:
        res = minimize(
            polish_obj,
            np.array([best_angle, best_phi]),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if 0 <= res.x[0] <= 60 and 0 <= res.x[1] <= 60:
            side_res = compute_side(flat_verts, float(res.x[0]), float(res.x[1]))
            if side_res < best_side:
                best_angle = float(res.x[0])
                best_phi = float(res.x[1])
                best_side = side_res
    except Exception:
        pass

    return best_angle, best_phi, best_side


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    # Build the rigid 11-hexagon cluster on the triangular lattice
    cluster_centers, flat_verts = build_rigid_cluster(scale_factor=1.002)

    # Optimize the rigid cluster by searching over global rotation and outer hexagon angle
    best_angle, best_phi, best_side = optimize_rigid_cluster(flat_verts, deadline - 5)

    # Apply the optimal rotation to the cluster centers
    angle_rad = np.deg2rad(best_angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    C_rotated = cluster_centers @ R.T

    # The inner hexagons all have the same rotation as the cluster
    A_rotated = np.full(N, best_angle)

    # Verify validity
    if not valid(C_rotated, A_rotated, best_side):
        # Fallback: use a simple valid configuration
        C_rotated = np.zeros((N, 2))
        for i in range(N):
            C_rotated[i] = [6.0 * i, 0.0]
        A_rotated = np.zeros(N)
        best_side = outer_side(C_rotated, A_rotated)

    Cn = C_rotated
    An = A_rotated
    Sn = best_side

    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
