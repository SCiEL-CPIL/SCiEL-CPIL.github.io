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

def _make_ring_init(rng, radius_scale=1.0, angle_noise=0.0):
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    # 6 outer, 6 inner
    for i in range(6):
        theta = i * np.pi / 3
        r = 2.5 * SQRT3 * radius_scale
        pos[i, 0] = r * np.cos(theta) + rng.uniform(-0.2, 0.2)
        pos[i, 1] = r * np.sin(theta) + rng.uniform(-0.2, 0.2)
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        r = 1.2 * SQRT3 * radius_scale
        pos[6 + i, 0] = r * np.cos(theta) + rng.uniform(-0.2, 0.2)
        pos[6 + i, 1] = r * np.sin(theta) + rng.uniform(-0.2, 0.2)
    if angle_noise > 0:
        ang = rng.uniform(-angle_noise, angle_noise, N)
    return pos, ang

def _make_honeycomb_init(rng):
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
    # add small noise
    pos += rng.uniform(-0.1, 0.1, (N, 2))
    return pos, ang

def _push_apart(pos, ang_rad, max_iters=50):
    """Iteratively push overlapping hexagons apart."""
    for _ in range(max_iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                h1 = (pos[i, 0], pos[i, 1], ang_rad[i])
                h2 = (pos[j, 0], pos[j, 1], ang_rad[j])
                if _hex_disjoint_fast(h1, h2):
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-6:
                    dx = rng.uniform(-0.1, 0.1)
                    dy = rng.uniform(-0.1, 0.1)
                    dist = math.sqrt(dx * dx + dy * dy)
                # desired separation: approx 2 * apothem = SQRT3 for aligned, use 1.8 as safe
                target = 1.8
                if dist < target:
                    push = (target - dist) / (2.0 * dist)
                    pos[i, 0] += dx * push
                    pos[i, 1] += dy * push
                    pos[j, 0] -= dx * push
                    pos[j, 1] -= dy * push
                    moved = True
        if not moved:
            break
    return pos

def _cross(parent_a, parent_b, rng, F=0.8):
    """Differential evolution cross: child = a + F*(b - a) + noise"""
    pos_a, ang_a = parent_a
    pos_b, ang_b = parent_b
    pos_c = pos_a + F * (pos_b - pos_a)
    ang_c = ang_a + F * (ang_b - ang_a)
    # add small noise
    pos_c += rng.uniform(-0.1, 0.1, (N, 2))
    ang_c += rng.uniform(-5, 5, N)
    # repair
    pos_c = _push_apart(pos_c, ang_c, max_iters=30)
    return pos_c, ang_c

def _mutate(pos, ang_rad, rng, step=0.3):
    """Random mutation: move a few hexagons."""
    pos_new = pos.copy()
    ang_new = ang_rad.copy()
    n_mut = rng.randint(1, 4)
    indices = rng.choice(N, size=n_mut, replace=False)
    for idx in indices:
        pos_new[idx, 0] += rng.uniform(-step, step)
        pos_new[idx, 1] += rng.uniform(-step, step)
        ang_new[idx] += rng.uniform(-10, 10)
    pos_new = _push_apart(pos_new, ang_new, max_iters=30)
    return pos_new, ang_new

def _local_polish(pos, ang_rad, time_limit, rng):
    """Coordinate descent polish."""
    current_side, _, _, _ = _compute_min_outer_side(pos, ang_rad)
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang_rad.copy()
    t_start = time.time()
    step = 0.05
    min_step = 0.0005
    no_improve = 0
    max_no_improve = 10

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        order = rng.permutation(N)
        for idx in order:
            if time.time() - t_start > time_limit:
                break
            best_move = None
            best_move_side = best_side
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = best_pos.copy()
                    new_ang = best_ang.copy()
                    if dim == 0:
                        new_pos[idx, 0] += direction * step
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step
                    else:
                        new_ang[idx] += direction * step
                    if not _is_valid(new_pos, new_ang):
                        continue
                    new_side, _, _, _ = _compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_move_side - 1e-12:
                        best_move_side = new_side
                        best_move = (new_pos.copy(), new_ang.copy())
            if best_move is not None:
                best_pos, best_ang = best_move
                best_side = best_move_side
                improved = True
                no_improve = 0
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                step *= 0.5
                no_improve = 0
            else:
                step *= 0.8

    return best_side, best_pos, best_ang

def construct():
    t0 = time.time()
    rng = np.random.RandomState(12345)

    # Initialize population
    pop_size = 20
    population = []
    # Seed with diverse inits
    population.append(_make_ring_init(rng, radius_scale=1.0))
    population.append(_make_ring_init(rng, radius_scale=0.9))
    population.append(_make_ring_init(rng, radius_scale=1.1))
    population.append(_make_honeycomb_init(rng))
    population.append(_make_ring_init(rng, radius_scale=1.0, angle_noise=10))
    while len(population) < pop_size:
        parent = population[rng.randint(len(population))]
        new_ind = _cross(parent, population[rng.randint(len(population))], rng)
        population.append(new_ind)

    # Evaluate initial population
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos, ang in population:
        if not _is_valid(pos, ang):
            continue
        s, phi, cx, cy = _compute_min_outer_side(pos, ang)
        if s < best_side:
            best_side = s
            best_pos = pos.copy()
            best_ang = ang.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # Main loop: DE-style evolution
    gen = 0
    while time.time() - t0 < DEADLINE - 20.0:
        gen += 1
        # Decay step size
        step = max(0.1, 0.5 * (1.0 - gen / 100.0))

        # Generate offspring via cross and mutation
        new_population = []
        n_offspring = pop_size // 2
        for _ in range(n_offspring):
            # Pick two parents
            i = rng.randint(pop_size)
            j = rng.randint(pop_size)
            if i == j:
                j = (j + 1) % pop_size
            if rng.random() < 0.7:
                # Cross
                new_ind = _cross(population[i], population[j], rng, F=0.8)
            else:
                # Mutate
                new_ind = _mutate(population[i][0], population[i][1], rng, step=step)
            new_population.append(new_ind)

        # Evaluate new offspring
        for pos, ang in new_population:
            if not _is_valid(pos, ang):
                continue
            s, phi, cx, cy = _compute_min_outer_side(pos, ang)
            if s < best_side:
                best_side = s
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

        # Selection: keep best pop_size individuals from combined population
        combined = population + new_population
        # Sort by side length (valid only)
        scored = []
        for pos, ang in combined:
            if _is_valid(pos, ang):
                s, _, _, _ = _compute_min_outer_side(pos, ang)
                scored.append((s, pos, ang))
            else:
                scored.append((float('inf'), pos, ang))
        scored.sort(key=lambda x: x[0])
        population = [(pos, ang) for _, pos, ang in scored[:pop_size]]

    # Final polish on best
    remaining = DEADLINE - 15.0 - (time.time() - t0)
    if remaining > 5.0 and best_pos is not None:
        polish_side, polish_pos, polish_ang = _local_polish(best_pos, best_ang, remaining, rng)
        if polish_side < best_side:
            best_side = polish_side
            best_pos = polish_pos
            best_ang = polish_ang
            best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, best_ang)

    # Fallback
    if best_pos is None or not _is_valid(best_pos, best_ang):
        pos0, ang0 = _make_ring_init(rng)
        pos0 = _push_apart(pos0, ang0, max_iters=100)
        best_pos = pos0
        best_ang = ang0
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, best_ang)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(math.degrees(best_ang[i]))])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
