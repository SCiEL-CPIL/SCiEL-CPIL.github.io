"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _eval_packing(xy, r):
    """Evaluate packing: return (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
    """Grid of N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.003
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random init with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _force_relax(xy, r, deadline, max_iter=20000, dt_init=0.001, cooling=0.9997):
    """
    Force-directed relaxation:
    - Circles repel each other when overlapping (strong)
    - Circles are attracted to grow (pressure)
    - Wall constraints push circles inward
    - Radius grows proportionally to available space
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    dt = dt_init
    # Adaptive dt: start small, cool down
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Pressure: want to grow radius
        pressure = 0.02  # force pushing radius up

        for i in range(n):
            # Wall forces
            if xy[i, 0] - r[i] < 0.01:
                fx[i] += 50.0 * (0.01 - (xy[i, 0] - r[i]))
            if 1.0 - xy[i, 0] - r[i] < 0.01:
                fx[i] -= 50.0 * (0.01 - (1.0 - xy[i, 0] - r[i]))
            if xy[i, 1] - r[i] < 0.01:
                fy[i] += 50.0 * (0.01 - (xy[i, 1] - r[i]))
            if 1.0 - xy[i, 1] - r[i] < 0.01:
                fy[i] -= 50.0 * (0.01 - (1.0 - xy[i, 1] - r[i]))

            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    # Repulsion force (spring-like)
                    overlap = min_dist - dist
                    force = 100.0 * overlap
                    nx = dx / dist
                    ny = dy / dist
                    fx[i] += force * nx
                    fy[i] += force * ny
                    fx[j] -= force * nx
                    fy[j] -= force * ny
                elif dist < 1e-12:
                    # Random nudge
                    fx[i] += (np.random.rand() - 0.5) * 0.1
                    fy[i] += (np.random.rand() - 0.5) * 0.1

        # Update positions
        for i in range(n):
            xy[i, 0] += fx[i] * dt
            xy[i, 1] += fy[i] * dt
            # Clamp to valid region
            xy[i, 0] = max(r[i], min(1.0 - r[i], xy[i, 0]))
            xy[i, 1] = max(r[i], min(1.0 - r[i], xy[i, 1]))

        # Grow radii: each circle grows toward its maximum possible radius
        # (limited by walls and nearest neighbor)
        for i in range(n):
            max_r_wall = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            max_r_circ = max_r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                available = (dist - r[j]) / 1.0  # fraction of dist available
                if available < max_r_circ:
                    max_r_circ = available
            target_r = max_r_circ * 0.999  # 99.9% of available
            # Move radius toward target with damping
            r[i] += (target_r - r[i]) * 0.05 * dt * 100
            r[i] = max(0.001, min(r[i], max_r_wall))

        # Cool down
        dt *= cooling

        # Periodic evaluation
        if it % 50 == 0:
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

    # Final eval
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()

    return best_xy, best_r, best_total, best_valid


def _largest_empty_circle(xy, r, n_circles):
    """
    Find the largest circle that fits in the empty space, considering:
    1. Circle centers (radius 0)
    2. Circle-circle intersection points
    3. Wall midpoints
    4. Random sample points
    Returns (x, y, radius) of the best empty circle.
    """
    n = n_circles
    candidates = []

    # 1. Existing circle centers (as potential new centers with r=0 clearance)
    for i in range(n):
        candidates.append((xy[i, 0], xy[i, 1], 0.0))

    # 2. Circle-circle intersection points
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            d = math.sqrt(dx * dx + dy * dy)
            r1, r2 = r[i], r[j]
            if d < 1e-12 or d > r1 + r2 or d < abs(r1 - r2):
                continue
            # Law of cosines for intersection
            a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
            h_sq = r1 * r1 - a * a
            if h_sq < 0:
                continue
            h = math.sqrt(h_sq)
            px = xy[i, 0] + a * dx / d
            py = xy[i, 1] + a * dy / d
            # Two intersection points
            candidates.append((px + h * dy / d, py - h * dx / d, 0.0))
            candidates.append((px - h * dy / d, py + h * dx / d, 0.0))

    # 3. Wall midpoints and corners
    candidates.extend([
        (0.5, 0.0, 0.0), (0.5, 1.0, 0.0), (0.0, 0.5, 0.0), (1.0, 0.5, 0.0),
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
        (0.5, 0.5, 0.0),
    ])

    # 4. Random samples
    rng = np.random.RandomState(42)
    for _ in range(2000):
        x = rng.uniform(0.0, 1.0)
        y = rng.uniform(0.0, 1.0)
        candidates.append((x, y, 0.0))

    # Evaluate each candidate
    best_r = 0.0
    best_x, best_y = 0.5, 0.5
    for cx, cy, _ in candidates:
        # Clearance to walls
        wall_clear = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if wall_clear < 0.001:
            continue
        # Clearance to circles
        circ_clear = wall_clear
        for i in range(n):
            dx = cx - xy[i, 0]
            dy = cy - xy[i, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            clear = dist - r[i]
            if clear < circ_clear:
                circ_clear = clear
        if circ_clear > best_r:
            best_r = circ_clear
            best_x, best_y = cx, cy

    return best_x, best_y, best_r


def _topology_expansion(xy, r, deadline, max_additions=10):
    """
    Iteratively find the largest empty circle and add it, then re-relax.
    This is a structural growth strategy different from coordinate optimization.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    for add in range(max_additions):
        if time.time() > deadline:
            break

        # Find largest empty circle
        ex, ey, er = _largest_empty_circle(xy, r, n)
        if er < 0.005:
            break

        # Add the new circle
        xy = np.vstack([xy, [[ex, ey]]])
        r = np.append(r, er)
        n += 1

        # Short force relaxation to settle
        xy, r, total, valid = _force_relax(xy, r, deadline, max_iter=3000, dt_init=0.0005, cooling=0.999)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        # Trim back to N circles if we exceeded
        if n > N:
            # Remove smallest circles
            order = np.argsort(r)
            keep = np.sort(np.setdiff1d(np.arange(n), order[:n - N]))
            xy = xy[keep]
            r = r[keep]
            n = len(xy)

    return best_xy, best_r, best_total, best_valid


def _simulated_annealing(xy, r, deadline, max_iter=50000, t_init=0.05, t_min=0.001, cooling=0.9999):
    """
    Simulated annealing: perturb positions and radii, accept based on energy.
    Energy = -sum(r) + penalty for violations.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    def _energy(xy_cur, r_cur):
        """Energy: -sum(r) + large penalty for violations."""
        pen = 0.0
        for i in range(len(xy_cur)):
            x, y, ri = xy_cur[i, 0], xy_cur[i, 1], r_cur[i]
            if ri < 0:
                pen += 10.0 * ri * ri
            if x - ri < 0:
                pen += 10.0 * (x - ri) ** 2
            if 1.0 - x - ri < 0:
                pen += 10.0 * (1.0 - x - ri) ** 2
            if y - ri < 0:
                pen += 10.0 * (y - ri) ** 2
            if 1.0 - y - ri < 0:
                pen += 10.0 * (1.0 - y - ri) ** 2
        for i in range(len(xy_cur)):
            for j in range(i + 1, len(xy_cur)):
                dx = xy_cur[i, 0] - xy_cur[j, 0]
                dy = xy_cur[i, 1] - xy_cur[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                overlap = r_cur[i] + r_cur[j] - dist
                if overlap > 0:
                    pen += 10.0 * overlap * overlap
        return -np.sum(r_cur) + pen

    t = t_init
    cur_energy = _energy(xy, r)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if t < t_min:
            break

        # Perturb
        xy_new = xy.copy()
        r_new = r.copy()
        # Random circle to perturb
        i = np.random.randint(n)
        dx = np.random.normal(0, 0.02)
        dy = np.random.normal(0, 0.02)
        dr = np.random.normal(0, 0.01)
        xy_new[i, 0] = max(0.01, min(0.99, xy_new[i, 0] + dx))
        xy_new[i, 1] = max(0.01, min(0.99, xy_new[i, 1] + dy))
        r_new[i] = max(0.005, min(0.45, r_new[i] + dr))

        new_energy = _energy(xy_new, r_new)
        delta_e = new_energy - cur_energy

        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / t):
            xy = xy_new
            r = r_new
            cur_energy = new_energy

            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

        t *= cooling

    return best_xy, best_r, best_total, best_valid


def _local_search_fine(xy, r, deadline, grid_size=120):
    """
    Fine grid-based local search: for each circle, find the best position
    on a fine grid that maximizes its radius given others fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
            dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
            dists = np.sqrt(dx * dx + dy * dy)
            circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
            r_candidates = np.minimum(wall_clear, circle_clear)

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    def _update_best(xy, r, total, valid):
        nonlocal best_xy, best_r, best_total, best_valid
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    # Phase 1: Multiple grid inits + force relaxation
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        out_xy, out_r, total, valid = _force_relax(xy, r, deadline, max_iter=15000)
        _update_best(out_xy, out_r, total, valid)

    # Phase 2: Random inits + force relaxation
    for seed in range(6):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _force_relax(xy, r, deadline, max_iter=10000)
        _update_best(out_xy, out_r, total, valid)

    # Phase 3: Topology expansion from best
    if best_valid and best_xy is not None:
        if time.time() < deadline - 30:
            tox, tor, tot, val = _topology_expansion(best_xy.copy(), best_r.copy(), deadline, max_additions=5)
            _update_best(tox, tor, tot, val)

    # Phase 4: Simulated annealing from best
    if best_valid and best_xy is not None:
        if time.time() < deadline - 40:
            sax, sar, sat, sav = _simulated_annealing(best_xy.copy(), best_r.copy(), deadline, max_iter=30000)
            _update_best(sax, sar, sat, sav)

    # Phase 5: Fine local search from best
    if best_valid and best_xy is not None:
        if time.time() < deadline - 20:
            lsx, lsr, lst, lsv = _local_search_fine(best_xy.copy(), best_r.copy(), deadline, grid_size=120)
            _update_best(lsx, lsr, lst, lsv)

    # Phase 6: Another force relaxation pass (cooler, longer)
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            frx, frr, frt, frv = _force_relax(best_xy.copy(), best_r.copy(), deadline, max_iter=20000, dt_init=0.0003, cooling=0.9999)
            _update_best(frx, frr, frt, frv)

    # Phase 7: Final fine local search
    if best_valid and best_xy is not None:
        if time.time() < deadline - 5:
            l2x, l2r, l2t, l2v = _local_search_fine(best_xy.copy(), best_r.copy(), deadline, grid_size=150)
            _update_best(l2x, l2r, l2t, l2v)

    # Fallback
    if best_xy is None or not best_valid:
        side = int(math.ceil(math.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = max(ri, min(1.0 - ri, x))
        result[i, 1] = max(ri, min(1.0 - ri, y))

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(math.ceil(math.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
