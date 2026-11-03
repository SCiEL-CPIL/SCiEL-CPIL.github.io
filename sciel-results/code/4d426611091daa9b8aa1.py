import math
import time
import numpy as np

N = 12
DEADLINE = 145.0
SQRT3 = math.sqrt(3)

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

# Discrete orientations (degrees)
ORIENT_DEGS = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
ORIENT_RAD = [math.radians(a) for a in ORIENT_DEGS]


def _hex_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1, a2 = h1[2], h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    return False


def _is_valid(positions, angles_rad):
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_rad):
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    phi_vals = np.radians(np.arange(0, 60, 0.5))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    best_side = float(np.max(side_vals).min())
    best_idx = int(np.argmin(np.max(side_vals, axis=1)))
    best_phi = float(best_idx * 0.5)

    lo = max(0.0, best_phi - 0.5)
    hi = min(60.0, best_phi + 0.5)
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(50):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = eval_phi(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = eval_phi(x2)
    best_phi = (lo + hi) / 2.0
    best_side = min(best_side, eval_phi(best_phi))

    return best_side, best_phi, float(center[0]), float(center[1])


def _build_lattice():
    """Build a finite lattice of candidate sites for inner hexagon centers.
    Uses a hexagonal lattice with spacing sqrt(3) (the natural hex packing spacing)
    over a grid, plus a denser offset grid for flexibility."""
    sites = []
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    # Primary hex lattice
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            sites.append(p)
    # Add offset sites (half-step) for more flexibility
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = (i + 0.5) * a1 + (j + 0.5) * a2
            sites.append(p)
    # Deduplicate
    sites_arr = np.array(sites)
    unique_sites = []
    for p in sites_arr:
        is_dup = False
        for q in unique_sites:
            if np.linalg.norm(p - q) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_sites.append(p)
    return np.array(unique_sites)


def _compute_soft_side_lower_bound(positions, angles_rad, phi_rad):
    """Compute a soft (continuous) estimate of the outer side length.
    This is used as a bound in the branch-and-bound search."""
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center
    normal_angles = phi_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    return float(np.max(side_vals))


def _nelder_mead_polish(positions, angles_rad, phi_deg, max_iter=200, t_start=0, time_limit=20.0):
    """Local continuous polish using Nelder-Mead on positions + angles + phi.
    Objective: outer side length + collision penalty."""
    n_hex = N
    # Parameter vector: [x0, y0, a0, x1, y1, a1, ..., phi]
    params = np.zeros(3 * n_hex + 1)
    for i in range(n_hex):
        params[3 * i] = positions[i, 0]
        params[3 * i + 1] = positions[i, 1]
        params[3 * i + 2] = angles_rad[i]
    params[3 * n_hex] = math.radians(phi_deg)

    def objective(p):
        pos = np.zeros((n_hex, 2))
        ang = np.zeros(n_hex)
        for i in range(n_hex):
            pos[i, 0] = p[3 * i]
            pos[i, 1] = p[3 * i + 1]
            ang[i] = p[3 * i + 2]
        phi = p[3 * n_hex]
        # Collision penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                h1 = (pos[i, 0], pos[i, 1], ang[i])
                h2 = (pos[j, 0], pos[j, 1], ang[j])
                # Quick distance check
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                if dx * dx + dy * dy > 4.5:
                    continue
                # Full check
                if not _hex_disjoint_fast(h1, h2):
                    # Estimate overlap depth
                    c1, s1 = math.cos(ang[i]), math.sin(ang[i])
                    c2, s2 = math.cos(ang[j]), math.sin(ang[j])
                    v1x = pos[i, 0] + _UX * c1 - _UY * s1
                    v1y = pos[i, 1] + _UX * s1 + _UY * c1
                    v2x = pos[j, 0] + _UX * c2 - _UY * s2
                    v2y = pos[j, 1] + _UX * s2 + _UY * c2
                    en1x = _ENX * c1 - _ENY * s1
                    en1y = _ENX * s1 + _ENY * c1
                    en2x = _ENX * c2 - _ENY * s2
                    en2y = _ENX * s2 + _ENY * c2
                    min_gap = float('inf')
                    for k in range(6):
                        nx, ny = en1x[k], en1y[k]
                        p1 = v1x * nx + v1y * ny
                        p2 = v2x * nx + v2y * ny
                        gap = min(p1.max() - p2.min(), p2.max() - p1.min())
                        min_gap = min(min_gap, gap)
                        nx, ny = en2x[k], en2y[k]
                        p1 = v1x * nx + v1y * ny
                        p2 = v2x * nx + v2y * ny
                        gap = min(p1.max() - p2.min(), p2.max() - p1.min())
                        min_gap = min(min_gap, gap)
                    if min_gap < 0:
                        penalty += min_gap * min_gap
        # Outer side
        c = np.cos(ang)
        s = np.sin(ang)
        rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
        ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
        verts_x = rx + pos[:, 0:1]
        verts_y = ry + pos[:, 1:2]
        verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        side = float(np.max(side_vals))
        return side + 50.0 * penalty

    # Nelder-Mead
    dim = len(params)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    # Initial simplex
    simplex = [params.copy()]
    f_vals = [objective(params)]
    step = 0.1
    for d in range(dim):
        p = params.copy()
        p[d] += step
        simplex.append(p)
        f_vals.append(objective(p))

    best_idx = int(np.argmin(f_vals))
    best_f = f_vals[best_idx]
    best_params = simplex[best_idx].copy()

    it = 0
    while it < max_iter:
        if time.time() - t_start > time_limit:
            break
        it += 1
        # Sort
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        centroid = np.mean(simplex[:-1], axis=0)
        # Reflect
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        if fr < f_vals[0]:
            # Expand
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
        elif fr < f_vals[-2]:
            simplex[-1] = xr
            f_vals[-1] = fr
        else:
            # Contract
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < f_vals[-1]:
                simplex[-1] = xc
                f_vals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = objective(simplex[i])

        new_best = int(np.argmin(f_vals))
        if f_vals[new_best] < best_f:
            best_f = f_vals[new_best]
            best_params = simplex[new_best].copy()

    # Extract
    pos = np.zeros((n_hex, 2))
    ang = np.zeros(n_hex)
    for i in range(n_hex):
        pos[i, 0] = best_params[3 * i]
        pos[i, 1] = best_params[3 * i + 1]
        ang[i] = best_params[3 * i + 2]
    phi_deg = math.degrees(best_params[3 * n_hex])
    return pos, ang, phi_deg


def _discrete_bnb_search(lattice, t_start, time_limit):
    """Branch-and-bound over discrete site-orientation assignments.
    Uses symmetry breaking: hexagon 0 is fixed at the origin with orientation 0."""
    n_sites = len(lattice)
    n_orients = len(ORIENT_RAD)
    # Symmetry: fix hex 0 at site closest to origin, orientation 0
    origin_idx = int(np.argmin(np.sum(lattice ** 2, axis=1)))
    fixed_site = origin_idx
    fixed_orient = 0

    best_side = float('inf')
    best_assignment = None
    best_positions = None
    best_angles = None

    # Precompute site positions
    site_pos = lattice

    # Branch-and-bound with best-first search
    # State: (assigned_count, positions_so_far, angles_so_far, current_bound)
    # Use a priority queue ordered by lower bound
    import heapq

    # Initial state
    init_pos = np.zeros((N, 2))
    init_ang = np.zeros(N)
    init_pos[0] = site_pos[fixed_site]
    init_ang[0] = ORIENT_RAD[fixed_orient]
    init_bound = _compute_soft_side_lower_bound(init_pos, init_ang, 0.0)

    # Priority queue: (bound, assigned_count, positions_tuple, angles_tuple)
    # For efficiency, store positions/angles as lists
    pq = []
    heapq.heappush(pq, (init_bound, 1, list(init_pos), list(init_ang)))

    nodes_expanded = 0
    while pq:
        if time.time() - t_start > time_limit:
            break
        bound, count, pos_list, ang_list = heapq.heappop(pq)
        if bound >= best_side:
            continue
        nodes_expanded += 1
        if count == N:
            # Complete assignment
            pos = np.array(pos_list)
            ang = np.array(ang_list)
            if _is_valid(pos, ang):
                side, phi, cx, cy = _compute_min_outer_side(pos, ang)
                if side < best_side:
                    best_side = side
                    best_assignment = (pos.copy(), ang.copy(), phi)
                    best_positions = pos.copy()
                    best_angles = ang.copy()
            continue

        # Try all sites and orientations for hexagon `count`
        for si in range(n_sites):
            for oi in range(n_orients):
                new_pos_list = pos_list.copy()
                new_ang_list = ang_list.copy()
                new_pos_list[count] = site_pos[si].tolist()
                new_ang_list[count] = ORIENT_RAD[oi]

                # Check collision with already placed
                valid = True
                for j in range(count):
                    if not _hex_disjoint_fast(
                        (new_pos_list[count][0], new_pos_list[count][1], new_ang_list[count]),
                        (new_pos_list[j][0], new_pos_list[j][1], new_ang_list[j])
                    ):
                        valid = False
                        break
                if not valid:
                    continue

                # Compute soft bound
                tmp_pos = np.array(new_pos_list[:count+1])
                tmp_ang = np.array(new_ang_list[:count+1])
                soft_bound = _compute_soft_side_lower_bound(tmp_pos, tmp_ang, 0.0)
                if soft_bound >= best_side:
                    continue

                heapq.heappush(pq, (soft_bound, count + 1, new_pos_list, new_ang_list))

    return best_side, best_positions, best_angles, nodes_expanded


def _make_honeycomb():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang


def construct():
    t0 = time.time()

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Phase 1: Discrete branch-and-bound search
    lattice = _build_lattice()
    t_bnb_start = time.time()
    bnb_time = min(40.0, DEADLINE - 60.0 - (t0 - t0))
    bnb_side, bnb_pos, bnb_ang, nodes = _discrete_bnb_search(lattice, t_bnb_start, bnb_time)

    if bnb_pos is not None:
        if _is_valid(bnb_pos, bnb_ang):
            s, phi, cx, cy = _compute_min_outer_side(bnb_pos, bnb_ang)
            if s < best_side:
                best_side = s
                best_positions = bnb_pos.copy()
                best_angles = bnb_ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    # Phase 2: Try multiple initial configurations and polish
    inits = [
        _make_honeycomb(),
    ]
    # Add the BNB result if valid
    if bnb_pos is not None and _is_valid(bnb_pos, bnb_ang):
        inits.append((bnb_pos.copy(), bnb_ang.copy()))

    # Add some random lattice-based inits
    rng = np.random.RandomState(42)
    for _ in range(3):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Pick random distinct sites
        sites_idx = rng.choice(len(lattice), size=N, replace=False)
        for i in range(N):
            pos[i] = lattice[sites_idx[i]]
            ang[i] = ORIENT_RAD[rng.randint(len(ORIENT_RAD))]
        if _is_valid(pos, ang):
            inits.append((pos, ang))

    for init_idx, (pos0, ang0_rad) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 30.0:
            break
        ang0_deg = np.degrees(ang0_rad)
        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0_rad)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0_rad.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        # Polish
        remaining = DEADLINE - 35.0 - (time.time() - t0)
        if remaining < 5.0:
            break
        polish_time = min(remaining * 0.7, 25.0)
        pos_pol, ang_pol, phi_pol = _nelder_mead_polish(
            pos0, ang0_rad, phi0,
            max_iter=300, t_start=time.time(), time_limit=polish_time
        )
        if _is_valid(pos_pol, ang_pol):
            s_pol, phi_pol_opt, cx_pol, cy_pol = _compute_min_outer_side(pos_pol, ang_pol)
            if s_pol < best_side:
                best_side = s_pol
                best_positions = pos_pol.copy()
                best_angles = ang_pol.copy()
                best_phi = phi_pol_opt
                best_cx = cx_pol
                best_cy = cy_pol

    # Final fallback
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = _make_honeycomb()
        best_positions = pos0.copy()
        best_angles = ang0.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(math.degrees(best_angles[i]))])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
