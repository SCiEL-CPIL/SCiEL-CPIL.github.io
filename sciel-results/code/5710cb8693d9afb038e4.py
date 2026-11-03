import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Geometry helpers ---
    # Unit hexagon with flat-top orientation (vertices at angles 0,60,...)
    # The 6 outward face normals are at 30, 90, 150, 210, 270, 330 degrees
    FACE_NORMALS = np.array([
        [math.cos(math.pi / 6 + k * math.pi / 3),
         math.sin(math.pi / 6 + k * math.pi / 3)]
        for k in range(6)
    ])  # (6, 2)

    # For non-overlap of two unit hexagons with same orientation:
    # Minkowski sum is a hexagon of side 2, apothem = sqrt(3)
    # Non-overlap condition: for all 6 face normals n, |d . n| >= sqrt(3)
    MIN_SEP = math.sqrt(3)

    def hex_vertices(cx, cy, s, angle_deg):
        """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3
            verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
        return verts

    def point_in_hexagon(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside the hexagon using half-plane tests."""
        a = math.radians(angle_deg)
        for k in range(6):
            theta = a + math.pi / 6 + k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            dist = (px - cx) * nx + (py - cy) * ny
            if dist > s * APOTHEM + 1e-9:
                return False
        return True

    def check_validity(inner_pts, inner_angles, outer_center, outer_side, outer_angle):
        """Full validity check: no overlaps, all inside outer hexagon."""
        ox, oy = outer_center
        os_ = outer_side
        oa = outer_angle

        # Check all inner hexagons are inside outer
        for i in range(N):
            ix, iy, ia = inner_pts[i][0], inner_pts[i][1], inner_angles[i]
            # Check center is inside
            if not point_in_hexagon(ix, iy, ox, oy, os_, oa):
                return False
            # Check all 6 vertices are inside
            for vx, vy in hex_vertices(ix, iy, 1.0, ia):
                if not point_in_hexagon(vx, vy, ox, oy, os_, oa):
                    return False

        # Check no overlaps between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                if inner_angles[i] != inner_angles[j]:
                    # Different rotations: check vertex-vertex and vertex-edge
                    vi = hex_vertices(inner_pts[i][0], inner_pts[i][1], 1.0, inner_angles[i])
                    vj = hex_vertices(inner_pts[j][0], inner_pts[j][1], 1.0, inner_angles[j])
                    # Simple check: no vertex of one inside the other
                    for vx, vy in vi:
                        if point_in_hexagon(vx, vy, inner_pts[j][0], inner_pts[j][1], 1.0, inner_angles[j]):
                            return False
                    for vx, vy in vj:
                        if point_in_hexagon(vx, vy, inner_pts[i][0], inner_pts[i][1], 1.0, inner_angles[i]):
                            return False
                else:
                    # Same rotation: use Minkowski sum condition
                    dx = inner_pts[i][0] - inner_pts[j][0]
                    dy = inner_pts[i][1] - inner_pts[j][1]
                    projs = np.abs(dx * FACE_NORMALS[:, 0] + dy * FACE_NORMALS[:, 1])
                    if np.min(projs) < MIN_SEP - 1e-9:
                        return False
        return True

    def compute_exact_S(inner_pts, inner_angles, outer_center, outer_angle):
        """Compute the minimal outer side length S for given inner positions and outer angle."""
        ox, oy = outer_center
        oa = math.radians(outer_angle)
        # Outer hexagon face normals
        outer_normals = np.array([
            [math.cos(oa + math.pi / 6 + k * math.pi / 3),
             math.sin(oa + math.pi / 6 + k * math.pi / 3)]
            for k in range(6)
        ])

        # For each inner hexagon, compute max projection of its vertices onto each outer normal
        max_proj = 0.0
        for i in range(N):
            ix, iy = inner_pts[i][0], inner_pts[i][1]
            ia = math.radians(inner_angles[i])
            # Vertices of inner hexagon
            for k in range(6):
                theta = ia + k * math.pi / 3
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                # Project onto each outer normal
                for m in range(6):
                    proj = (vx - ox) * outer_normals[m, 0] + (vy - oy) * outer_normals[m, 1]
                    abs_proj = abs(proj)
                    if abs_proj > max_proj:
                        max_proj = abs_proj
        S = (max_proj + APOTHEM) / APOTHEM
        return S, max_proj

    # --- Initial candidate configurations (axial lattice, all same angle) ---
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate several diverse starting configurations
    def make_lattice_config(rows, cols, offset_type='simple'):
        """Create a hexagonal lattice configuration."""
        pts = []
        for row in range(rows):
            for col in range(cols):
                if offset_type == 'simple':
                    q = col
                    r = row
                elif offset_type == 'stagger':
                    q = col + (row % 2) * 0.5
                    r = row
                pts.append((q, r))
        return pts

    # Diverse initial configs
    init_configs = []

    # Config 1: 4-3-4 hexagonal arrangement
    c1 = [(0,0),(1,0),(2,0),(3,0),(0.5,1),(1.5,1),(2.5,1),(0,2),(1,2),(2,2),(3,2)]
    init_configs.append(c1)

    # Config 2: 3-4-4
    c2 = [(0,0),(1,0),(2,0),(0.5,1),(1.5,1),(2.5,1),(3.5,1),(0,2),(1,2),(2,2),(3,2)]
    init_configs.append(c2)

    # Config 3: Central + ring + extras
    c3 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    init_configs.append(c3)

    # Config 4: 5-3-3
    c4 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0.5,1),(1.5,1),(2.5,1),(0,2),(1,2),(2,2)]
    init_configs.append(c4)

    # Config 5: 4-4-3
    c5 = [(0,0),(1,0),(2,0),(3,0),(0.5,1),(1.5,1),(2.5,1),(3.5,1),(0,2),(1,2),(2,2)]
    init_configs.append(c5)

    # Config 6: 3-3-3-2
    c6 = [(0,0),(1,0),(2,0),(0.5,1),(1.5,1),(2.5,1),(0,2),(1,2),(2,2),(0.5,3),(1.5,3)]
    init_configs.append(c6)

    # Config 7: 2-4-2-3
    c7 = [(0,0),(1,0),(0.5,1),(1.5,1),(2.5,1),(3.5,1),(0,2),(1,2),(0.5,3),(1.5,3),(2.5,3)]
    init_configs.append(c7)

    # Config 8: Compact cluster
    c8 = [(0,0),(1,0),(2,0),(0.5,1),(1.5,1),(2.5,1),(0,2),(1,2),(2,2),(0.5,3),(1.5,3)]
    init_configs.append(c8)

    # Config 9: Elongated
    c9 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0.5,1),(1.5,1),(2.5,1),(3.5,1),(0,2),(1,2)]
    init_configs.append(c9)

    # Config 10: 3-2-3-3
    c10 = [(0,0),(1,0),(2,0),(0.5,1),(1.5,1),(0,2),(1,2),(2,2),(0.5,3),(1.5,3),(2.5,3)]
    init_configs.append(c10)

    # --- Smooth objective function ---
    def smooth_objective(params):
        """
        params layout: [x1,y1, ..., xN,yN, cx, cy, beta, a1, a2, ..., aN]
        Total: 2*N + 2 + 1 + N = 3*N + 3 = 36 for N=11
        """
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        n = N
        pts = params[:2*n].reshape(n, 2)
        cx = params[2*n]
        cy = params[2*n + 1]
        beta = params[2*n + 2]
        angles = params[2*n + 3: 2*n + 3 + n]  # N angles in radians

        # Normalize beta to [0, pi/6)
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        # Outer hexagon face normals
        outer_normals = np.array([
            [math.cos(beta + math.pi / 6 + k * math.pi / 3),
             math.sin(beta + math.pi / 6 + k * math.pi / 3)]
            for k in range(6)
        ])

        # Compute max projection of all inner hexagon vertices onto outer normals
        all_projs = []
        for i in range(n):
            ix, iy = pts[i]
            ia = angles[i]
            for k in range(6):
                theta = ia + k * math.pi / 3
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                for m in range(6):
                    proj = (vx - cx) * outer_normals[m, 0] + (vy - cy) * outer_normals[m, 1]
                    all_projs.append(proj)
                    all_projs.append(-proj)

        all_projs = np.array(all_projs)
        max_val = np.max(all_projs)
        if max_val > 200:
            return 1e10

        # Log-sum-exp for smooth max
        temperature = 0.05
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM

        # Non-overlap penalty (soft)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(angles[i] - angles[j]) > 1e-6:
                    # Different rotations: approximate with center distance
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    # Minimum distance for non-overlap with different rotations ~ 2*APOTHEM + 2*APOTHEM = 4*APOTHEM
                    min_dist = 4 * APOTHEM
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2 * 50.0
                else:
                    # Same rotation: exact Minkowski condition
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    projs = np.abs(dx * FACE_NORMALS[:, 0] + dy * FACE_NORMALS[:, 1])
                    min_proj = np.min(projs)
                    if min_proj < MIN_SEP:
                        violation = MIN_SEP - min_proj
                        penalty += violation ** 2 * 200.0

        return S + penalty

    # --- Nelder-Mead implementation (pure Python, no scipy dependency for robustness) ---
    def nelder_mead(objective, x0, max_iter=5000, xatol=1e-10, fatol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.05 * (abs(xi[i]) + 1.0)
            simplex.append(xi)

        fvals = [objective(s) for s in simplex]

        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[0] - fvals[-1]) < fatol:
                break
            spread = max(abs(simplex[0][i] - simplex[-1][i]) for i in range(n))
            if spread < xatol:
                break

            # Centroid of all but worst
            centroid = np.mean([simplex[i] for i in range(n)], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = objective(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])

        # Return best
        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]]

    # --- Multi-start Nelder-Mead optimization ---
    best_S = float('inf')
    best_params = None

    for cfg_idx, axial_cfg in enumerate(init_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        pts = axial_to_cartesian(axial_cfg)

        # Compute initial center as centroid
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        beta0 = 0.0
        angles0 = np.zeros(N)  # All same angle initially

        # Initial params
        init_params = np.concatenate([pts.ravel(), [cx0, cy0, beta0], angles0])

        # Run Nelder-Mead
        try:
            result, fval = nelder_mead(smooth_objective, init_params, max_iter=3000, xatol=1e-8, fatol=1e-8)
            if fval < best_S:
                best_S = fval
                best_params = result.copy()
        except Exception:
            pass

        # Also try with small random perturbations
        for trial in range(3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            perturbed = init_params.copy()
            noise = np.random.randn(len(perturbed)) * 0.02
            perturbed += noise
            # Keep angles small
            angle_start = 2 * N + 3
            perturbed[angle_start:] = np.clip(perturbed[angle_start:], -0.1, 0.1)
            try:
                result, fval = nelder_mead(smooth_objective, perturbed, max_iter=2000, xatol=1e-8, fatol=1e-8)
                if fval < best_S:
                    best_S = fval
                    best_params = result.copy()
            except Exception:
                pass

    # --- Final polish with tighter parameters ---
    if best_params is not None:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                result, fval = nelder_mead(smooth_objective, best_params, max_iter=5000, xatol=1e-12, fatol=1e-12)
                if fval < best_S:
                    best_S = fval
                    best_params = result.copy()
            except Exception:
                pass

    # --- Extract solution and compute exact S ---
    if best_params is None:
        # Fallback: use first config
        pts = axial_to_cartesian(init_configs[0])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        beta = 0.0
        angles = np.zeros(N)
        inner_pts = pts
    else:
        n = N
        inner_pts = best_params[:2 * n].reshape(n, 2)
        cx = best_params[2 * n]
        cy = best_params[2 * n + 1]
        beta = ((best_params[2 * n + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        angles = best_params[2 * n + 3: 2 * n + 3 + n]

    # Compute exact S
    exact_S, max_proj = compute_exact_S(inner_pts, angles, (cx, cy), math.degrees(beta))

    # Verify validity
    inner_list = [(float(inner_pts[i][0]), float(inner_pts[i][1]), float(angles[i])) for i in range(N)]
    angle_degs = [float(math.degrees(a)) for a in angles]

    if not check_validity(inner_list, angle_degs, (float(cx), float(cy)), float(exact_S), float(math.degrees(beta))):
        # Fallback to a simple valid configuration
        pts = axial_to_cartesian(init_configs[0])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        beta = 0.0
        angles = np.zeros(N)
        inner_pts = pts
        exact_S, _ = compute_exact_S(inner_pts, angles, (cx, cy), 0.0)
        inner_list = [(float(inner_pts[i][0]), float(inner_pts[i][1]), 0.0) for i in range(N)]
        angle_degs = [0.0] * N

    # Build output
    inner = []
    for i in range(N):
        inner.append([float(inner_pts[i][0]), float(inner_pts[i][1]), float(angle_degs[i])])

    center = [float(cx), float(cy)]
    side = float(exact_S)
    rotation = float(math.degrees(beta))

    return inner, center, side, rotation
