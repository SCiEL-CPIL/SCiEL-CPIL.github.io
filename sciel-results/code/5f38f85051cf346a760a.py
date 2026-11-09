# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    """Return 6 outward edge normals for a hexagon given its 6 vertices in order."""
    normals = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        # outward normal: perpendicular, pointing away from center
        n = np.array([-edge[1], edge[0]])
        # ensure it points outward (away from centroid)
        centroid = hex_verts.mean(axis=0)
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _separation(v1, n1, v2, n2):
    """
    Check if two hexagons with vertices v1 (6x2) and v2 (6x2) and normals n1, n2 (6x2)
    are separated. Returns True if separated by > 1e-6 on at least one of 12 normals.
    """
    # Project all vertices of both hexagons onto each of the 12 normals
    # For each normal, check if the projected intervals are apart by > 1e-6
    all_normals = np.vstack([n1, n2])  # 12 x 2
    for k in range(12):
        n = all_normals[k]
        proj1 = v1 @ n  # 6 values
        proj2 = v2 @ n  # 6 values
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Intervals are apart if max1 < min2 - gap or max2 < min1 - gap
        if (max1 < min2 - 1e-6) or (max2 < min1 - 1e-6):
            return True
    return False


def _all_separated(inner_hexes):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation(inner_hexes[i][0], inner_hexes[i][1],
                               inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _min_outer_side(inner_hexes):
    """
    Find the minimum side length S such that all inner hexagons fit inside
    a regular hexagon centered at origin with side S and angle 0.
    Returns (S, outer_angle) or (inf, 0) if infeasible.
    """
    # For a given outer hexagon (center=0, side=S, angle=0), check if all inner
    # vertices are inside.
    # Outer hexagon vertices for angle 0, side S:
    # The constraint is: for each outer edge from p1 to p2, for each inner vertex p,
    # cross(p2-p1, p-p1) >= -1e-6

    def _fits(S, angle_deg):
        outer_verts = _hex_vertices(0, 0, S, angle_deg)
        for hex_verts, normals in inner_hexes:
            for p in hex_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    # Try to find minimum S by binary search over S and angle
    # First, get a rough bound
    # The inner hexagons have some extent; compute bounding radius
    max_r = 0
    for hex_verts, normals in inner_hexes:
        r = np.max(np.linalg.norm(hex_verts, axis=1))
        max_r = max(max_r, r)
    
    # A hexagon with side S has inradius = S*sqrt(3)/2, circumradius = S
    # So we need S >= max_r (circumradius) as a lower bound
    S_low = max_r
    S_high = S_low * 2.0
    
    # Binary search for minimum S with angle=0 first
    for _ in range(50):
        S_mid = (S_low + S_high) / 2.0
        if _fits(S_mid, 0.0):
            S_high = S_mid
        else:
            S_low = S_mid
    
    best_S = S_high
    best_angle = 0.0
    
    # Try different angles to see if we can do better
    for angle_deg in [0, 10, 20, 30, 40, 50, 60]:
        # Binary search for this angle
        sl = max_r * 0.95
        sh = max_r * 2.0
        for _ in range(50):
            sm = (sl + sh) / 2.0
            if _fits(sm, angle_deg):
                sh = sm
            else:
                sl = sm
        if sh < best_S:
            best_S = sh
            best_angle = angle_deg
    
    return best_S, best_angle


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    t_start = time.time()
    deadline = 600.0
    
    # Seed arrangement
    seed_positions = np.array(
        [
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17],
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
        ]
    )
    
    # Compute vertices and normals for seed
    def _build_inner_hexes(positions, angles):
        """Build list of (vertices, normals) for each inner hexagon."""
        hexes = []
        for i in range(11):
            verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            norms = _edge_normals(verts)
            hexes.append((verts, norms))
        return hexes
    
    # Objective: minimize outer side length
    def _objective(params):
        # params: 22 values = 11 positions (x,y) + 11 angles
        positions = params[:22].reshape(11, 2)
        angles = params[22:]
        
        # Check time
        if time.time() > deadline - 30:
            return 1e10
        
        # Build hexes
        inner_hexes = _build_inner_hexes(positions, angles)
        
        # Check separation
        if not _all_separated(inner_hexes):
            # Penalty for violations
            penalty = 0.0
            for i in range(11):
                for j in range(i + 1, 11):
                    v1, n1 = inner_hexes[i]
                    v2, n2 = inner_hexes[j]
                    all_normals = np.vstack([n1, n2])
                    min_gap = np.inf
                    for k in range(12):
                        n = all_normals[k]
                        proj1 = v1 @ n
                        proj2 = v2 @ n
                        gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                        # We want gap > 1e-6, so if gap <= 1e-6, it's a violation
                        # Actually, separation means at least one normal has gap > 1e-6
                        # If no normal has gap > 1e-6, it's a violation
                        if gap > 1e-6:
                            min_gap = 0
                            break
                    else:
                        # No normal separates them
                        # Compute minimum violation
                        min_gap = min(
                            proj2.min() - proj1.max(),
                            proj1.min() - proj2.max()
                        )
                        penalty += max(0, 1e-6 - min_gap)
            if penalty > 0:
                return 1e10 + penalty
        
        # Find min outer side
        S, angle = _min_outer_side(inner_hexes)
        if not np.isfinite(S):
            return 1e10
        return S
    
    # Initial guess
    x0 = np.zeros(33)
    x0[:22] = seed_positions.flatten()
    x0[22:] = 0.0  # angles
    
    # Use Nelder-Mead
    best_params = x0.copy()
    best_S, best_angle = _min_outer_side(_build_inner_hexes(seed_positions, np.zeros(11)))
    
    n_iter = 0
    while time.time() < deadline - 60:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            n_iter += 1
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
                # Recompute best_angle
                positions = best_params[:22].reshape(11, 2)
                angles = best_params[22:]
                inner_hexes = _build_inner_hexes(positions, angles)
                best_S, best_angle = _min_outer_side(inner_hexes)
        except Exception as e:
            break
        
        if time.time() > deadline - 60:
            break
    
    # Final result
    positions = best_params[:22].reshape(11, 2)
    angles = best_params[22:]
    
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
