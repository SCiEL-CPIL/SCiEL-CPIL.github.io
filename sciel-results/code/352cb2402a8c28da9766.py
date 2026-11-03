import math
import time
import numpy as np

N = 12
SQRT3 = math.sqrt(3)
DEADLINE_FRACTION = 0.72


def construct():
    start = time.time()
    deadline = start + 150.0

    # Precompute unit hexagon vertices (pointy-top, side length 1)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    def hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
        if angle_deg != 0.0:
            theta = math.radians(angle_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            verts = (unit_verts @ R.T) * side
        else:
            verts = unit_verts * side
        return verts + np.array([cx, cy])

    def min_outer_side_for_container(centers, angles, outer_angle_deg):
        """
        Given inner hexagon centers and angles, compute the minimum side length
        of a regular hexagon (with given rotation) that contains all of them,
        centered at the optimal location.
        
        The optimal center is found by solving a linear program or by
        iterative projection. For a regular hexagon, the constraint is:
        For each of 6 directions (normals at outer_angle + 30 + k*60):
            -A <= dot(center, n_k) - max_proj_k <= A
            -A <= min_proj_k - dot(center, n_k) <= A
        where A = side * sqrt(3) / 2.
        
        So we need:
            max_proj_k - A <= dot(center, n_k) <= A - min_proj_k
        for all k.
        
        This is feasible if for all pairs (i, j):
            max_proj_i - A + dot(c, n_i) - (dot(c, n_j) + A - min_proj_j) <= ...
        
        Actually, simpler: the minimum A is half the maximum over all directions
        of (max_proj - min_proj) projected along the hexagon normals.
        
        For a container hexagon with side S and rotation theta:
        The apothem is S * sqrt(3) / 2.
        In each of the 6 normal directions n_k, the constraint is:
            |dot(c, n_k) - proj_k| <= S * sqrt(3) / 2
        where proj_k is the range of projections of all inner vertices onto n_k.
        
        We need: for each k, (max_proj_k - min_proj_k) <= S * sqrt(3)
        (since the width in direction n_k must fit within the hexagon's width)
        
        Wait, that's not quite right. The hexagon's "width" in direction n_k 
        (where n_k is a normal) is 2 * apothem = S * sqrt(3).
        
        So we need: for each k, (max_proj_k - min_proj_k) <= S * sqrt(3).
        
        But we also need to find a center c such that:
            c * n_k in [max_proj_k - S*sqrt(3)/2, min_proj_k + S*sqrt(3)/2]
        
        This is feasible iff for all k, l:
            (max_proj_k - S*sqrt(3)/2) + (c * (n_k - n_l)) <= (min_proj_l + S*sqrt(3)/2)
        
        Actually, let's just use a simpler approach: 
        For a given outer angle, the minimum S is determined by the maximum
        "extent" of all inner vertices when projected onto the 6 normal directions.
        
        The minimum S such that a hexagon of rotation theta contains all points P:
        S = max_k (max_proj_k - min_proj_k) / sqrt(3)
        
        And the center can be chosen as c where c * n_k = (max_proj_k + min_proj_k) / 2
        for each k. This system may not have an exact solution, but we can find
        a center that works via linear programming or just use the mean.
        
        For the score, we just need S. The center returned is used to verify
        containment, so we need a valid center.
        """
        theta = math.radians(outer_angle_deg)
        normals = np.array([
            [math.cos(theta + math.pi/6 + k * math.pi/3),
             math.sin(theta + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])
        
        # Collect all vertices
        all_verts = []
        for i in range(N):
            cx, cy = centers[i]
            ang = angles[i]
            v = hex_vertices(cx, cy, 1.0, ang)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)  # (72, 2)
        
        projs = all_verts @ normals.T  # (72, 6)
        min_projs = np.min(projs, axis=0)  # (6,)
        max_projs = np.max(projs, axis=0)  # (6,)
        widths = max_projs - min_projs  # (6,)
        
        max_width = np.max(widths)
        side = max_width / SQRT3
        
        # Find a valid center: solve for c such that
        # dot(c, n_k) = (max_proj_k + min_proj_k) / 2 for all k
        # This is an overdetermined system; use least squares
        targets = (max_projs + min_projs) / 2.0
        # c @ normals.T = targets  =>  normals @ c = targets
        c, _, _, _ = np.linalg.lstsq(normals, targets, rcond=None)
        
        # Verify containment with this center
        # Shift all vertices by -c, then check
        shifted = all_verts - c
        shifted_projs = shifted @ normals.T
        A = side * SQRT3 / 2
        # Check: for each k, -A <= shifted_proj_k <= A
        # This should hold by construction, but verify
        if np.max(np.abs(shifted_projs)) > A + 1e-9:
            # Fallback: use centroid
            c = np.mean(all_verts, axis=0)
            shifted = all_verts - c
            shifted_projs = shifted @ normals.T
            if np.max(np.abs(shifted_projs)) > A + 1e-9:
                # Increase side slightly
                side = np.max(np.abs(shifted_projs)) * 2 / SQRT3 + 1e-9
        
        return side, c

    def check_no_overlap(centers, angles, eps=1e-10):
        """Check that no two inner hexagons overlap. Returns True if valid."""
        polys = [hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = polys[i], polys[j]
                
                # Bounding box quick check
                if (np.max(v1[:, 0]) < np.min(v2[:, 0]) - eps or
                    np.max(v2[:, 0]) < np.min(v1[:, 0]) - eps or
                    np.max(v1[:, 1]) < np.min(v2[:, 1]) - eps or
                    np.max(v2[:, 1]) < np.min(v1[:, 1]) - eps):
                    continue
                
                # Check if any vertex of v1 is inside v2 or vice versa
                if _any_point_in_polygon(v1, v2, eps) or _any_point_in_polygon(v2, v1, eps):
                    return False
                
                # Check edge intersections
                if _edges_intersect(v1, v2, eps):
                    return False
        return True

    def _point_in_convex_poly(pt, poly, eps=1e-12):
        """Check if point is inside a convex polygon (using cross products)."""
        n = len(poly)
        sign = None
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
            if abs(cross) < eps:
                continue
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    def _any_point_in_polygon(points, poly, eps):
        for pt in points:
            if _point_in_convex_poly(pt, poly, eps):
                return True
        return False

    def _segments_intersect(p1, p2, p3, p4, eps=1e-12):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
           ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
            return True
        return False

    def _edges_intersect(v1, v2, eps):
        for i in range(6):
            p1, p2 = v1[i], v1[(i + 1) % 6]
            for j in range(6):
                p3, p4 = v2[j], v2[(j + 1) % 6]
                if _segments_intersect(p1, p2, p3, p4, eps):
                    return True
        return False

    def objective(params):
        """
        Objective: minimize outer side length with overlap penalty.
        params: [x0, y0, x1, y1, ..., x11, y11] (angles fixed at 0)
        """
        if time.time() > deadline * DEADLINE_FRACTION:
            return 1e10
        
        centers = np.array(params).reshape(N, 2)
        angles = np.zeros(N)
        
        # Overlap penalty
        overlap = 0.0
        polys = [hex_vertices(centers[i][0], centers[i][1], 1.0, 0.0) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = polys[i], polys[j]
                # Bounding box
                if (np.max(v1[:, 0]) < np.min(v2[:, 0]) - 1e-9 or
                    np.max(v2[:, 0]) < np.min(v1[:, 0]) - 1e-9 or
                    np.max(v1[:, 1]) < np.min(v2[:, 1]) - 1e-9 or
                    np.max(v2[:, 1]) < np.min(v1[:, 1]) - 1e-9):
                    continue
                if _any_point_in_polygon(v1, v2, 1e-9) or _any_point_in_polygon(v2, v1, 1e-9):
                    overlap += 10.0
                elif _edges_intersect(v1, v2, 1e-9):
                    overlap += 10.0
            if overlap > 50.0:
                break
        
        if overlap > 0:
            return overlap + 1.0
        
        # Compute minimum outer side (try a few outer angles)
        best_side = float('inf')
        for oa in [0.0, 30.0, 15.0, 45.0]:
            s, _ = min_outer_side_for_container(centers, angles, oa)
            if s < best_side:
                best_side = s
        
        return best_side

    # --- Generate initial configurations ---
    
    configs = []
    
    # Config 1: Hexagonal lattice, 12 closest points to origin
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = SQRT3 * (i + j / 2.0)
            y = 3.0 * j
            lattice.append((x, y))
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    init1 = np.array(lattice[:12])
    init1 -= np.mean(init1, axis=0)
    configs.append(init1.flatten().tolist())
    
    # Config 2: 3-4-3-2 arrangement (rows)
    # Row 0: 3 hexes at y=0
    # Row 1: 4 hexes at y=SQRT3
    # Row 2: 3 hexes at y=2*SQRT3
    # Row 3: 2 hexes at y=3*SQRT3
    row_widths = [3, 4, 3, 2]
    y_spacing = SQRT3
    x_spacing = 2.0
    pts = []
    y_offset = 0
    for r, w in enumerate(row_widths):
        y = y_offset + r * y_spacing
        x_start = -(w - 1) * x_spacing / 2.0
        for k in range(w):
            x = x_start + k * x_spacing
            # Offset alternating rows
            if r % 2 == 1:
                x += x_spacing / 2.0
            pts.append((x, y))
    init2 = np.array(pts)
    init2 -= np.mean(init2, axis=0)
    configs.append(init2.flatten().tolist())
    
    # Config 3: 2-5-5-0 (two rows of 5, one row of 2)
    pts3 = []
    for k in range(5):
        pts3.append((k * 2.0 - 4.0, 0.0))
        pts3.append((k * 2.0 - 4.0, SQRT3))
    # Two more
    pts3.append((-1.0, 2 * SQRT3))
    pts3.append((1.0, 2 * SQRT3))
    init3 = np.array(pts3)
    init3 -= np.mean(init3, axis=0)
    configs.append(init3.flatten().tolist())
    
    # Config 4: Ring + center
    center = [(0, 0)]
    for k in range(6):
        a = math.radians(60 * k)
        center.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    # 5 more around the ring
    for k in range(5):
        a = math.radians(60 * k + 30)
        center.append((3.5 * math.cos(a), 3.5 * math.sin(a)))
    init4 = np.array(center)
    init4 -= np.mean(init4, axis=0)
    configs.append(init4.flatten().tolist())
    
    # Config 5: Compact 4x3 grid-like
    pts5 = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 - 3.0
            y = r * 2.0 * SQRT3 / 2.0
            if r % 2 == 1:
                x += 1.0
            pts5.append((x, y))
    # That's 12
    init5 = np.array(pts5)
    init5 -= np.mean(init5, axis=0)
    configs.append(init5.flatten().tolist())
    
    # Evaluate all configs
    best_params = None
    best_obj = float('inf')
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg
    
    # --- Local optimization using scipy ---
    from scipy.optimize import minimize
    
    # Try multiple random perturbations of the best config
    rng = np.random.default_rng(42)
    
    for restart in range(8):
        if time.time() > deadline * DEADLINE_FRACTION:
            break
        
        if restart == 0:
            x0 = np.array(best_params)
        else:
            x0 = np.array(best_params) + rng.normal(0, 0.5, size=2*N)
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Also try Powell for smoother convergence
        if time.time() < deadline * DEADLINE_FRACTION:
            try:
                res2 = minimize(
                    objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'xtol': 1e-5,
                        'ftol': 1e-7
                    }
                )
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    best_params = res2.x
            except Exception:
                pass
    
    # --- Final result ---
    final_centers = np.array(best_params).reshape(N, 2)
    final_angles = np.zeros(N)
    
    # Validate
    if not check_no_overlap(final_centers, final_angles):
        # Fallback to config 1 (lattice)
        final_centers = init1
        final_angles = np.zeros(N)
        if not check_no_overlap(final_centers, final_angles):
            # Last resort: simple grid
            pts_fb = []
            for r in range(3):
                for c in range(4):
                    pts_fb.append((c * 2.5 - 3.75, r * 3.0))
            final_centers = np.array(pts_fb)
            final_angles = np.zeros(N)
    
    # Compute final side with best outer angle
    best_side = float('inf')
    best_outer_angle = 0.0
    for oa in np.linspace(0, 60, 61):
        s, c = min_outer_side_for_container(final_centers, final_angles, oa)
        if s < best_side:
            best_side = s
            best_outer_angle = oa
            best_center = c
    
    inner = []
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], 0.0])
    
    return inner, [best_center[0], best_center[1]], best_side, best_outer_angle
