"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 disjoint unit regular hexagons inside a regular hexagon.
    Minimize the outer side length.
    Strategy: 2D grid search over global rotation, then local optimization
    of positions (translation + rotation per hexagon) with a hard constraint
    that hexagons are non-overlapping and inside the outer hexagon.
    We use a smooth surrogate for the non-overlap constraint.
    """
    t_start = time.time()
    deadline = 150.0 - 10.0  # 140s wall-clock limit, keep 10s safety margin
    
    def hex_vertices(center, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        cx, cy = center
        theta = np.deg2rad(angle_deg)
        angles = theta + np.pi / 3 * np.arange(6)
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack((x, y))
    
    def hex_in_hexagon_distance(inner_verts, outer_center, outer_side, outer_angle_deg):
        """
        Return minimum distance from inner hexagon to outside of outer hexagon.
        Positive means fully inside, negative means overlapping boundary.
        We compute the max over vertices of (distance to outer hexagon boundary inward).
        Actually, we want the minimum over vertices of the signed distance to outer hex.
        If all vertices are inside, the hexagon is inside (since hexagons are convex).
        """
        oc = np.array(outer_center)
        outer_verts = hex_vertices(oc, outer_side, outer_angle_deg)
        # Signed distance from a point to a convex polygon (positive if inside)
        # We use the support function approach: for a convex polygon, point is inside
        # if it's on the correct side of all edges.
        min_dist = np.inf
        for v in inner_verts:
            dist = np.inf
            for i in range(6):
                e0 = outer_verts[i]
                e1 = outer_verts[(i + 1) % 6]
                # Edge vector
                e = e1 - e0
                # Inward normal
                e_perp = np.array([-e[1], e[0]])
                # Normalize
                norm = np.linalg.norm(e_perp)
                if norm < 1e-12:
                    continue
                e_perp = e_perp / norm
                # Check if it's inward: the center of outer hex should be on the positive side
                center_test = oc - e0
                if np.dot(center_test, e_perp) < 0:
                    e_perp = -e_perp
                # Signed distance from v to this edge (positive if inside)
                d = np.dot(v - e0, e_perp)
                dist = min(dist, d)
            min_dist = min(min_dist, dist)
        return min_dist
    
    def pair_overlap_distance(h1, h2):
        """
        Estimate the overlap distance between two hexagons.
        Returns 0 if they are disjoint (with some margin), negative if overlapping.
        We use the minimum over vertices of h1 of distance to h2 boundary, and vice versa.
        """
        v1 = h1
        v2 = h2
        min_d = np.inf
        # For each vertex of v1, find min distance to edges of v2
        for p in v1:
            d_to_v2 = np.inf
            for i in range(6):
                a = v2[i]
                b = v2[(i + 1) % 6]
                ab = b - a
                t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                d = np.linalg.norm(p - closest)
                d_to_v2 = min(d_to_v2, d)
            min_d = min(min_d, d_to_v2)
        # For each vertex of v2, find min distance to edges of v1
        for p in v2:
            d_to_v1 = np.inf
            for i in range(6):
                a = v1[i]
                b = v1[(i + 1) % 6]
                ab = b - a
                t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                d = np.linalg.norm(p - closest)
                d_to_v1 = min(d_to_v1, d)
            min_d = min(min_d, d_to_v1)
        # If min_d > 0, they are disjoint with gap min_d
        # We want a penalty that is 0 when min_d >= 0, and grows when min_d < 0
        # Use softplus-like: penalty = -min(min_d, 0) if we want to push them apart
        return min_d
    
    def total_overlap_penalty(positions, angles, outer_side, outer_center, outer_angle):
        """
        Compute total overlap penalty.
        positions: (12, 2) array of centers
        angles: (12,) array of angles in degrees
        Returns (inside_penalty, pair_penalty)
        """
        inside_pen = 0.0
        pair_pen = 0.0
        verts_list = []
        for i in range(N):
            c = positions[i]
            a = angles[i]
            v = hex_vertices(c, 1.0, a)
            verts_list.append(v)
            # Inside outer hexagon
            d = hex_in_hexagon_distance(v, outer_center, outer_side, outer_angle)
            if d < 0:
                inside_pen += -d ** 2
        for i in range(N):
            for j in range(i + 1, N):
                d = pair_overlap_distance(verts_list[i], verts_list[j])
                if d < 0:
                    pair_pen += (-d) ** 2
        return inside_pen, pair_pen
    
    def objective(params, outer_side, outer_center, outer_angle):
        positions = params[:2 * N].reshape(N, 2)
        angles = params[2 * N:]
        inside_pen, pair_pen = total_overlap_penalty(positions, angles, outer_side, outer_center, outer_angle)
        # We want to minimize outer_side, but here we fix outer_side and check feasibility
        # The penalty is the sum of violations
        return inside_pen + pair_pen
    
    def is_valid(positions, angles, outer_side, outer_center, outer_angle, tol=1e-6):
        inside_pen, pair_pen = total_overlap_penalty(positions, angles, outer_side, outer_center, outer_angle)
        return inside_pen < tol and pair_pen < tol
    
    # Initial guess: arrange 12 hexagons in a hexagonal pattern
    # 1 center, 6 around, 5 more... let's use a standard hexagonal packing
    # For 12 hexagons, a good layout is a hexagon of radius 2 (1+6+... no, that's 7)
    # 12 = 1 + 6 + 5? No. Let's use a 3x4 rectangular grid or a honeycomb pattern.
    # Actually, a hexagonal number: 1, 7, 19. 12 is not a hex number.
    # Let's place them in a 2x6 grid or 3x4 grid, or a honeycomb of 4 rows.
    
    # Generate initial positions in a honeycomb pattern
    def initial_honeycomb(n):
        # Place n hexagons in a honeycomb pattern centered at origin
        # Row 0: 1 hex at (0,0)
        # Row 1: 3 hex at (1, 0.866), (0, 0.866), (-1, 0.866) ... wait, hex packing
        # Side length 1, distance between centers in honeycomb: sqrt(3)
        dx = np.sqrt(3)
        dy = 1.5
        positions = []
        angles = np.zeros(n)
        # Let's just use a simple grid for initialization
        # 3 rows of 4
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * dx
                y = (r - 1) * dy
                positions.append([x, y])
        # That's 12
        return np.array(positions), angles
    
    init_pos, init_ang = initial_honeycomb(N)
    init_params = np.concatenate([init_pos.flatten(), init_ang])
    
    best_score = np.inf
    best_result = (init_pos, init_ang, 20.0, (0.0, 0.0), 0.0)
    
    # Grid search over outer rotation and side length
    # Outer hexagon side length: try from small to large
    # Outer rotation: 0 to 60 degrees
    outer_rotations = np.linspace(0, 60, 13)  # 13 values
    outer_sides = np.linspace(6.0, 12.0, 25)  # 25 values
    
    # For each outer configuration, try to fit the inner hexagons
    # We'll do a coarse search first, then refine
    
    for outer_rot in outer_rotations:
        for outer_side in outer_sides:
            if time.time() > deadline:
                break
            # Optimize inner positions and angles
            res = minimize(objective, init_params, args=(outer_side, (0.0, 0.0), outer_rot),
                           method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-4})
            if res.success:
                pos = res.x[:2*N].reshape(N,2)
                ang = res.x[2*N:]
                if is_valid(pos, ang, outer_side, (0.0, 0.0), outer_rot):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (pos, ang, outer_side, (0.0, 0.0), outer_rot)
    
    if time.time() > deadline - 20:
        # Return best found so far
        pos, ang, side, oc, oang = best_result
        inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
        return inner, oc, side, oang
    
    # Refine: for the best outer configuration, do a finer optimization
    if best_score < np.inf:
        pos, ang, side, oc, oang = best_result
        # Try to reduce side length further
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            new_side = side * 0.98
            res = minimize(objective, np.concatenate([pos.flatten(), ang]),
                           args=(new_side, oc, oang),
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5})
            if res.success:
                new_pos = res.x[:2*N].reshape(N,2)
                new_ang = res.x[2*N:]
                if is_valid(new_pos, new_ang, new_side, oc, oang):
                    side = new_side
                    pos = new_pos
                    ang = new_ang
                    best_score = side
                    best_result = (pos, ang, side, oc, oang)
                else:
                    break
    
    pos, ang, side, oc, oang = best_result
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    return inner, oc, side, oang
# EVOLVE-BLOCK-END
