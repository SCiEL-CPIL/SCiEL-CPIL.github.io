import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Pre-compute the rigid 3x4 honeycomb cluster of unit hexagons ---
    # Hex grid: 3 columns, 4 rows. Axial coords -> Cartesian.
    # For a regular hex grid with side length 1:
    #   horizontal spacing between columns: sqrt(3)
    #   vertical spacing between rows: 1.5
    #   odd columns shifted up by 0.75
    def build_cluster():
        """Build 3x4 honeycomb cluster, return (12,2) array of centers."""
        centers = []
        for row in range(4):
            for col in range(3):
                x = col * np.sqrt(3)
                y = row * 1.5
                if col % 2 == 1:
                    y += 0.75
                centers.append([x, y])
        # Center the cluster at origin
        arr = np.array(centers)
        centroid = arr.mean(axis=0)
        arr -= centroid
        return arr

    cluster_centers = build_cluster()  # shape (12, 2)

    # Vertices of a single unit hexagon (flat-top orientation, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    unit_hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
        for a in [0, 60, 120, 180, 240, 300]
    ])  # shape (6, 2)

    # All 12 hexagons' vertices in the cluster frame (before global transform)
    # Each hexagon i has vertices: cluster_centers[i] + unit_hex_verts
    # We'll compute these on the fly for speed

    # --- Analytic containment check ---
    # A regular hexagon of side length S, centered at (cx, cy), rotated by theta,
    # contains a point p if and only if for each of the 6 face normals:
    #   dot(p - center, normal_k) <= S * sqrt(3) / 2
    # The face normals of a hexagon with vertex angle offset alpha are at
    #   alpha + 30 + k*60 degrees for k=0..5

    def min_enclosing_hex_side(points, center, rot_deg):
        """
        Given a set of 2D points and a candidate outer hexagon (center, rotation),
        compute the minimum side length S such that all points are inside the hexagon.
        
        The hexagon has vertices at angles rot + k*60, and face normals at
        rot + 30 + k*60. For point p to be inside:
            dot(p - center, n_k) <= S * sqrt(3)/2  for all k
        So S >= max_k dot(p - center, n_k) / (sqrt(3)/2) for each point p.
        Overall S = max over all points and all normals.
        """
        rot_rad = np.deg2rad(rot_deg)
        # 6 face normal angles
        normal_angles = rot_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        # points: (M, 2), center: (2,)
        rel = points - center  # (M, 2)
        # dot products: (M, 6)
        dots = rel @ normals.T
        # max dot per normal, then overall max
        max_dot = np.max(dots)
        S = max_dot / (np.sqrt(3) / 2.0)
        return max(0.0, S)

    def get_cluster_points(x, y, rot_deg):
        """Apply global transform to cluster: translate + rotate."""
        rot_rad = np.deg2rad(rot_deg)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad), np.cos(rot_rad)]])
        # All vertices of all 12 hexagons in cluster frame
        # For each hex i: vertices = cluster_centers[i] + unit_hex_verts
        # Total: 12 * 6 = 72 vertices
        # But we can be smarter: the convex hull of the cluster is what matters.
        # Actually, for containment in a convex set, we only need the convex hull vertices.
        # The convex hull of the 3x4 honeycomb of unit hexagons:
        # Let's just compute all 72 vertices and take the convex hull.
        
        # All vertex positions in cluster frame
        all_verts = []
        for i in range(12):
            verts = cluster_centers[i] + unit_hex_verts
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)  # (72, 2)
        
        # Apply rotation
        rotated = all_verts @ R.T
        # Apply translation
        translated = rotated + np.array([x, y])
        return translated

    def objective(params):
        x, y, rot_deg, S = params
        points = get_cluster_points(x, y, rot_deg)
        required_S = min_enclosing_hex_side(points, np.array([0.0, 0.0]), 0.0)
        # We want S >= required_S. Penalty if S < required_S.
        # Minimize S subject to S >= required_S.
        # Use a penalty formulation: minimize S + penalty * max(0, required_S - S)^2
        violation = max(0.0, required_S - S)
        return S + 100.0 * violation * violation

    # --- Optimization ---
    best_S = float('inf')
    best_params = None

    # Initial guess: the cluster has a certain bounding size.
    # Estimate: the 3x4 hex cluster spans roughly:
    #   width ~ 3*sqrt(3) + 2 (for hex widths) ~ 7.2
    #   height ~ 4*1.5 + 1 ~ 7
    # A hexagon enclosing this needs side ~ 5-6.
    S0 = 5.5

    # Try several initial rotations and use Nelder-Mead
    initial_configs = [
        [0.0, 0.0, 0.0, S0],
        [0.0, 0.0, 15.0, S0],
        [0.0, 0.0, -15.0, S0],
        [0.0, 0.0, 30.0, S0],
        [0.0, 0.0, -30.0, S0],
        [0.0, 0.0, 45.0, S0],
        [0.0, 0.0, 60.0, S0],
        [0.5, 0.3, 10.0, S0],
        [-0.5, -0.3, -10.0, S0],
        [0.3, -0.5, 20.0, S0],
    ]

    for x0 in initial_configs:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5,
                                    'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Also try a finer grid search over rotation with analytic S computation
    # For each rotation, compute the required S analytically (no optimization needed
    # for x,y since we can center the cluster optimally)
    # Actually, for a given rotation, the optimal (x,y) centers the bounding hex.
    # Let's do: for each rotation, compute all cluster points, then find the
    # minimum S hexagon (centered at origin, rotated by 0) that contains them.
    # But we can also shift the cluster. The optimal shift for a given rotation
    # is to center the "bounding region" appropriately.
    
    # Simpler: for each rotation, compute required S with center at origin,
    # and also try small offsets.
    
    for rot_deg in np.arange(0, 60, 1.0):
        if time.time() > deadline:
            break
        for dx in [-0.5, 0.0, 0.5]:
            for dy in [-0.5, 0.0, 0.5]:
                points = get_cluster_points(dx, dy, rot_deg)
                S_req = min_enclosing_hex_side(points, np.array([0.0, 0.0]), 0.0)
                if S_req < best_S:
                    best_S = S_req
                    best_params = np.array([dx, dy, rot_deg, S_req])

    # Refine with Nelder-Mead from the best grid point
    if best_params is not None:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6,
                                    'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final refinement: try to reduce S slightly by adjusting position
    if best_params is not None:
        # The optimal center for a given rotation minimizes the max extent.
        # Let's do a local search on (x, y) for fixed rotation and S.
        x, y, rot_deg, S = best_params
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for dx in np.linspace(-0.2, 0.2, 5):
                for dy in np.linspace(-0.2, 0.2, 5):
                    points = get_cluster_points(x + dx, y + dy, rot_deg)
                    S_req = min_enclosing_hex_side(points, np.array([0.0, 0.0]), 0.0)
                    if S_req < best_S - 1e-6:
                        best_S = S_req
                        best_params = np.array([x + dx, y + dy, rot_deg, S_req])
                        x, y = x + dx, y + dy
                        improved = True
            if not improved:
                break

    # --- Construct the output ---
    if best_params is None:
        # Fallback: use the cluster as-is with a generous outer hex
        x, y, rot_deg, S = 0.0, 0.0, 0.0, 8.0
    else:
        x, y, rot_deg, S = best_params

    # Ensure S is at least what's needed (safety margin)
    points = get_cluster_points(x, y, rot_deg)
    S_check = min_enclosing_hex_side(points, np.array([0.0, 0.0]), 0.0)
    S = max(S, S_check + 1e-6)

    # Generate inner hexagon positions
    rot_rad = np.deg2rad(rot_deg)
    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                  [np.sin(rot_rad), np.cos(rot_rad)]])
    
    inner = []
    for i in range(12):
        cx, cy = cluster_centers[i]
        # Apply rotation and translation
        px, py = R @ np.array([cx, cy]) + np.array([x, y])
        inner.append([px, py, rot_deg])

    return inner, (0.0, 0.0), float(S), 0.0
