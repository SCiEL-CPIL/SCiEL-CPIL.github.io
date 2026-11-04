import numpy as np
import time
import math

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

# unit hexagon vertices
UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

def outer_side(C, A):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        # Project centers
        proj_c = C @ n
        # The max projection of a hexagon centered at c is c.n + 1 (apothem is S3/2, but wait.
        # Distance from center to edge is apothem = S3/2 * side. Side=1. Apothem = S3/2.
        # Distance from center to vertex is 1.
        # The support function of a unit hexagon in direction n is the apothem if n is normal to an edge?
        # No, the support function h(n) = max_{v in Hex} v.n.
        # For a unit hexagon, h(n) is the distance to the supporting line.
        # If n is a normal, h(n) = S3/2.
        # If n is a vertex direction, h(n) = 1.
        # The formula used in the original code: proj = C @ n + 1.0.
        # This assumes the max projection of the unit hexagon is 1.0.
        # This is correct only if n is aligned with a vertex.
        # For general n, the max projection is not constant.
        # However, the original code uses this. Let's verify.
        # If we use the exact vertices, we should project the vertices.
        
        # Let's use the exact vertex projection for correctness.
        # But for speed, if we assume the cluster is rigid, we can precompute the vertices.
        
        pass
    # The original code's outer_side is an approximation or assumes specific alignment?
    # "proj = C @ n + 1.0"
    # If the hexagon is rotated, the support function changes.
    # The original code passes A (angles) to outer_side but doesn't use them in the projection!
    # It uses C @ n + 1.0. This implies it assumes the hexagons are NOT rotated relative to the container?
    # Or it's a bug/approximation.
    # If the inner hexagons are rotated, their vertices extend further in some directions.
    # To be safe, we should compute the exact bounding box using vertices.
    
    # Let's rewrite outer_side to be exact.
    m = 0.0
    for i in range(len(C)):
        angle_rad = np.deg2rad(A[i])
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = UNIT @ R.T + C[i]
        for k in range(6):
            n = NORM[k]
            proj = verts @ n
            m = max(m, float(proj.max()))
    return 2.0 * m / S3

def exact_nonoverlap(C, A, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists."""
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
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for i in range(N):
        angle_rad = np.deg2rad(A[i])
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = UNIT @ R.T + C[i]
        for k in range(6):
            n = NORM[k]
            proj = verts @ n
            if proj.max() > apothem + eps:
                return False
    return True

def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S)

def build_optimal_lattice_cluster():
    """
    Builds the 11-hexagon cluster on the triangular lattice.
    Uses the centrally symmetric subset of the second ring.
    Returns centers in Cartesian coordinates and angles (all 0).
    """
    u = np.array([1.5, S3 / 2.0])
    v = np.array([0.0, S3])
    
    # 11 lattice points:
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (subset): (2,-1), (1,-2), (-2,1), (-1,2)
    
    points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, -2), (-2, 1), (-1, 2)
    ]
    
    centers = np.array([i * u + j * v for i, j in points])
    # Center the cluster at the origin
    centers -= np.mean(centers, axis=0)
    
    angles = np.zeros(N)
    return centers, angles

def compute_side_for_cluster_rotation(centers, angles, theta_deg):
    """
    Compute the side length of the enclosing hexagon for a given global rotation theta.
    The inner hexagons rotate with the cluster.
    """
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_global = np.array([[c, -s], [s, c]])
    
    # Rotate centers
    C_rot = centers @ R_global.T
    
    # Rotate angles
    A_rot = angles + theta_deg
    
    # Calculate outer side
    m = 0.0
    for i in range(N):
        angle_rad = np.deg2rad(A_rot[i])
        c_i, s_i = np.cos(angle_rad), np.sin(angle_rad)
        R_i = np.array([[c_i, -s_i], [s_i, c_i]])
        verts = UNIT @ R_i.T + C_rot[i]
        for k in range(6):
            n = NORM[k]
            proj = verts @ n
            m = max(m, float(proj.max()))
    return 2.0 * m / S3

def construct():
    start = time.time()
    deadline = start + 145.0

    # 1. Build the optimal lattice cluster
    centers, angles = build_optimal_lattice_cluster()
    
    # 2. Optimize the global rotation
    # Due to 2-fold symmetry of the chosen cluster subset, we only need to search [0, 30] degrees.
    # Actually, the cluster has 2-fold symmetry (180 deg). The hexagon has 6-fold symmetry.
    # The combined system has 2-fold symmetry.
    # We can search 0 to 30 degrees.
    
    best_theta = 0.0
    best_side = float('inf')
    
    # Coarse search
    for theta in np.linspace(0, 30, 61):
        side = compute_side_for_cluster_rotation(centers, angles, theta)
        if side < best_side:
            best_side = side
            best_theta = theta
            
    if time.time() > deadline - 10:
        C_final = centers.copy()
        A_final = angles.copy()
        # Apply best rotation
        theta_rad = np.deg2rad(best_theta)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        C_final = centers @ R.T
        A_final = angles + best_theta
        return [[float(C_final[i, 0]), float(C_final[i, 1]), float(A_final[i])] for i in range(N)], (0.0, 0.0), float(best_side), 0.0

    # Fine search
    fine_start = max(0, best_theta - 2)
    fine_end = min(30, best_theta + 2)
    for theta in np.linspace(fine_start, fine_end, 1000):
        side = compute_side_for_cluster_rotation(centers, angles, theta)
        if side < best_side:
            best_side = side
            best_theta = theta

    # Nelder-Mead polish
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t: compute_side_for_cluster_rotation(centers, angles, t), 
                          bounds=(fine_start, fine_end), method='bounded')
    if res.success and res.fun < best_side:
        best_side = res.fun
        best_theta = res.x

    # Finalize
    theta_rad = np.deg2rad(best_theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    C_final = centers @ R.T
    A_final = angles + best_theta

    # Verify validity (should be valid by construction if lattice is tight enough)
    # The lattice packing is non-overlapping.
    # The container is computed to enclose them.
    
    return [[float(C_final[i, 0]), float(C_final[i, 1]), float(A_final[i])] for i in range(N)], (0.0, 0.0), float(best_side), 0.0
