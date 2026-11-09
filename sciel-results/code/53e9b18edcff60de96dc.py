"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import time
import numpy as np

N = 26
DEADLINE = 140.0  # safety margin vs 150s wall-clock kill


def _clamp_inside(pos, rad):
    """Move center so the circle stays inside [0,1]^2; returns (pos, rad)."""
    x, y = pos
    r = rad
    if x < r:
        x = r
    if x > 1.0 - r:
        x = 1.0 - r
    if y < r:
        y = r
    if y > 1.0 - r:
        y = 1.0 - r
    return (x, y), r


def _max_radius_for_position(pos, other_centers, other_radii, exclude):
    """Max radius for a circle at pos that stays inside [0,1]^2 and is
    disjoint from all other circles (excluding index `exclude`)."""
    x, y = pos
    # distance to each boundary
    d_wall = min(x, 1.0 - x, y, 1.0 - y)
    r = d_wall
    if r <= 0:
        return 0.0
    # distance to other circles
    for k in range(len(other_centers)):
        if k == exclude:
            continue
        ox, oy = other_centers[k]
        dx = x - ox
        dy = y - oy
        dist = (dx * dx + dy * dy) ** 0.5
        cand = dist - other_radii[k]
        if cand < r:
            r = cand
    return max(0.0, r)


def _score(centers, radii):
    s = 0.0
    for i in range(len(centers)):
        x, y = centers[i]
        if x < -1e-9 or x > 1.0 + 1e-9 or y < -1e-9 or y > 1.0 + 1e-9:
            return 0.0
        if radii[i] < -1e-9:
            return 0.0
        s += radii[i]
    return s


def _valid(centers, radii):
    for i in range(len(centers)):
        x, y = centers[i]
        r = radii[i]
        if x < r - 1e-9 or x > 1.0 - r + 1e-9 or y < r - 1e-9 or y > 1.0 - r + 1e-9:
            return False
    for i in range(len(centers)):
        xi, yi = centers[i]
        ri = radii[i]
        for j in range(i + 1, len(centers)):
            xj, yj = centers[j]
            rj = radii[j]
            dx = xi - xj
            dy = yi - yj
            if (dx * dx + dy * dy) ** 0.5 < ri + rj - 1e-9:
                return False
    return True


def _grid_start(seed):
    side = int(np.ceil(np.sqrt(N)))
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(side):
        for j in range(side):
            jitter = 0.02
            cx = (i + 0.5) / side + rng.uniform(-jitter, jitter)
            cy = (j + 0.5) / side + rng.uniform(-jitter, jitter)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return cells[:N]


def _local_search(cells, deadline, max_passes=400):
    centers = [c[0] for c in cells]
    radii = [c[2] for c in cells]
    n = len(centers)
    best = _score(centers, radii)
    if not _valid(centers, radii):
        # try to repair radii
        for i in range(n):
            centers[i], radii[i] = _clamp_inside(centers[i], radii[i])
            radii[i] = _max_radius_for_position(centers[i], centers, radii, i)
        if not _valid(centers, radii):
            return centers, radii, best
        best = _score(centers, radii)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline - 5.0:
            break
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            x, y = centers[i]
            # candidate positions: current + small perturbations in a grid
            base_r = radii[i]
            cur_pos = (x, y)
            cur_r = _max_radius_for_position(cur_pos, centers, radii, i)
            best_pos, best_r = cur_pos, cur_r
            # try expanding radius first (move toward free space)
            # sample directions
            for ang in range(0, 360, 30):
                for step in (0.01, 0.02, 0.03, 0.05):
                    nx = x + step * np.cos(np.radians(ang))
                    ny = y + step * np.sin(np.radians(ang))
                    # also try larger radius by moving away from nearest neighbor
                    cand_r = _max_radius_for_position((nx, ny), centers, radii, i)
                    if cand_r > best_r + 1e-9:
                        best_r = cand_r
                        best_pos = (nx, ny)
            # also try random positions
            for _ in range(8):
                nx = np.random.uniform(0.0, 1.0)
                ny = np.random.uniform(0.0, 1.0)
                cand_r = _max_radius_for_position((nx, ny), centers, radii, i)
                if cand_r > best_r + 1e-9:
                    best_r = cand_r
                    best_pos = (nx, ny)

            if best_r > base_r + 1e-9:
                centers[i], radii[i] = _clamp_inside(best_pos, best_r)
                radii[i] = _max_radius_for_position(centers[i], centers, radii, i)
                improved = True
                cur = _score(centers, radii)
                if cur > best:
                    best = cur
    return centers, radii, best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_centers = None
    best_radii = None
    best_val = -1.0

    n_restarts = 12
    for s in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        cells = _grid_start(s)
        centers, radii, val = _local_search(cells, deadline)
        if val > best_val:
            best_val = val
            best_centers = centers[:]
            best_radii = radii[:]

    if best_centers is None:
        # fallback grid
        cells = _grid_start(0)
        best_centers = [c[0] for c in cells]
        best_radii = [c[2] for c in cells]
        best_val = _score(best_centers, best_radii)

    # Final repair pass to ensure validity
    for i in range(len(best_centers)):
        best_centers[i], best_radii[i] = _clamp_inside(best_centers[i], best_radii[i])
        best_radii[i] = _max_radius_for_position(best_centers[i], best_centers, best_radii, i)

    # If still invalid, shrink all radii uniformly
    if not _valid(best_centers, best_radii):
        for i in range(len(best_radii)):
            best_radii[i] *= 0.9
        for i in range(len(best_centers)):
            best_centers[i], best_radii[i] = _clamp_inside(best_centers[i], best_radii[i])

    out = np.array([[best_centers[i][0], best_centers[i][1], best_radii[i]] for i in range(N)])
    return out
