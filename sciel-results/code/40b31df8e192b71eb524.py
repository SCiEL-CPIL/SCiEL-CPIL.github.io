"""Circle packing in unit square via local search with random restarts."""
import numpy as np
import time

N = 26
DEADLINE = 150.0  # evaluator kills after 150s; keep a safety margin
SAFETY = 10.0     # seconds to reserve before the hard limit


def _max_radius(pos, centers, radii, i, margin=1e-9):
    """Largest radius for circle i at fixed position pos, given other circles."""
    x, y = pos
    # wall constraints
    r = min(x, 1.0 - x, y, 1.0 - y)
    if r <= 0:
        return 0.0
    # pairwise constraints: r_i <= d - r_j  =>  r_i + r_j <= d
    for j in range(len(centers)):
        if j == i:
            continue
        dx = x - centers[j][0]
        dy = y - centers[j][1]
        d = np.hypot(dx, dy)
        # need r_i + r_j <= d  =>  r_i <= d - r_j
        limit = d - radii[j]
        if limit < r:
            r = limit
    if r < 0:
        return 0.0
    return max(0.0, r - margin)


def _evaluate(pos, centers, radii, i):
    """Radius of circle i at pos given current others; also checks feasibility."""
    return _max_radius(pos, centers, radii, i)


def _local_search(centers, radii, deadline):
    """Iteratively grow each circle's radius at its position, then reposition.

    Strategy: for each circle, try to increase its radius by moving it to a
    better spot. We do a simple hill-climb: for circle i, sample candidate
    positions near its current one and keep the one that gives the largest
    radius. Iterate until no improvement or deadline.
    """
    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)
    n = len(centers)
    total = radii.sum()
    best_total = total
    best_centers = centers.copy()
    best_radii = radii.copy()

    # Recompute radii to be consistent with current positions
    for i in range(n):
        radii[i] = _max_radius(centers[i], centers, radii, i)
    total = radii.sum()
    best_total = total
    best_centers = centers.copy()
    best_radii = radii.copy()

    rng = np.random.default_rng()
    no_improve = 0
    max_no_improve = 40

    while time.time() < deadline:
        improved = False
        # Randomize order each pass
        order = rng.permutation(n)
        for i in range(n):
            if time.time() >= deadline:
                break
            ci = order[i]
            cx, cy = centers[ci]
            cur_r = radii[ci]

            # Candidate positions: current + random perturbations of varying scale
            best_r = cur_r
            best_pos = (cx, cy)
            # scale of perturbation proportional to current radius
            scale = max(0.01, cur_r * 1.5)
            for _ in range(30):
                # random direction and magnitude
                ang = rng.uniform(0, 2 * np.pi)
                mag = scale * rng.uniform(0.1, 1.0)
                nx = cx + mag * np.cos(ang)
                ny = cy + mag * np.sin(ang)
                # clamp inside [0,1]
                nx = min(max(nx, 0.0), 1.0)
                ny = min(max(ny, 0.0), 1.0)
                r = _max_radius((nx, ny), centers, radii, ci)
                if r > best_r + 1e-10:
                    best_r = r
                    best_pos = (nx, ny)

            if best_r > cur_r + 1e-10:
                centers[ci] = best_pos
                radii[ci] = best_r
                improved = True

        # Recompute all radii (positions changed, so others may shrink)
        for i in range(n):
            radii[i] = _max_radius(centers[i], centers, radii, i)

        total = radii.sum()
        if total > best_total + 1e-10:
            best_total = total
            best_centers = centers.copy()
            best_radii = radii.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break

    return best_centers, best_radii, best_total


def _make_grid_init():
    """Create a grid of N circles in the unit square."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for j in range(side):
        for i in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:N])


def _make_random_init():
    """Create N random circles in the unit square with small radii."""
    rng = np.random.default_rng()
    centers = rng.uniform(0.1, 0.9, size=(N, 2))
    # Start with small radii to avoid overlaps
    radii = np.full(N, 0.02)
    return centers, radii


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY

    best_total = -1.0
    best_centers = None
    best_radii = None

    # Phase 1: Grid initialization + local search
    grid = _make_grid_init()
    centers = grid[:, :2].copy()
    radii = grid[:, 2].copy()
    # Recompute radii properly
    for i in range(N):
        radii[i] = _max_radius(centers[i], centers, radii, i)
    total = radii.sum()
    if total > best_total:
        best_total = total
        best_centers = centers.copy()
        best_radii = radii.copy()

    if time.time() < deadline:
        centers, radii, total = _local_search(centers, radii, deadline)
        if total > best_total:
            best_total = total
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Phase 2: Multiple random restarts
    restart_count = 0
    while time.time() < deadline and restart_count < 50:
        restart_count += 1
        centers, radii = _make_random_init()
        # Recompute radii
        for i in range(N):
            radii[i] = _max_radius(centers[i], centers, radii, i)
        total = radii.sum()

        if time.time() < deadline:
            centers, radii, total = _local_search(centers, radii, deadline)
            if total > best_total:
                best_total = total
                best_centers = centers.copy()
                best_radii = radii.copy()

    # Final safety: ensure all circles are valid (inside square, non-overlapping)
    # Clamp positions and recompute radii one final time
    centers = np.clip(best_centers, 0.0, 1.0)
    for i in range(N):
        best_radii[i] = _max_radius(centers[i], centers, best_radii, i)

    result = np.column_stack([centers, best_radii])
    return result
