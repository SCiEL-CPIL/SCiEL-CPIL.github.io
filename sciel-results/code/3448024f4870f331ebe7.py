"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 150.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - _elapsed() - 8.0  # 8s safety margin


def _is_valid(centers, radii, tol=1e-5):
    """Check if packing is valid."""
    n = len(radii)
    for i in range(n):
        if radii[i] < -tol:
            return False
        if centers[i, 0] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 0] - radii[i] < -tol:
            return False
        if centers[i, 1] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 1] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                        (centers[i, 1] - centers[j, 1])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _fix_packing(centers, radii):
    """Fix small violations by shrinking radii and clipping positions."""
    centers = centers.copy()
    radii = radii.copy()
    for i in range(len(radii)):
        centers[i, 0] = np.clip(centers[i, 0], 0.0, 1.0)
        centers[i, 1] = np.clip(centers[i, 1], 0.0, 1.0)
    for i in range(len(radii)):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max(0.0, max_r))
    for _ in range(100):
        changed = False
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                            (centers[i, 1] - centers[j, 1])**2)
                needed = radii[i] + radii[j]
                if d < needed:
                    if radii[i] < radii[j]:
                        radii[i] = max(0.0, (d - radii[j]) * 0.999)
                    else:
                        radii[j] = max(0.0, (d - radii[i]) * 0.999)
                    changed = True
        if not changed:
            break
    return centers, radii


def _force_relax(centers, target_radii, n_steps=2000, dt=0.005,
                 wall_strength=50.0, repel_strength=100.0):
    """
    Force-directed relaxation:
    - Repulsive force between overlapping or near circles (Hertzian-like)
    - Wall repulsion to keep circles inside [0,1]^2
    - Velocity Verlet with damping
    """
    n = len(centers)
    centers = centers.astype(np.float64).copy()
    vel = np.zeros((n, 2))
    damping = 0.85

    for step in range(n_steps):
        # Compute forces
        forces = np.zeros((n, 2))

        # Wall forces (strong repulsion near boundaries)
        for i in range(n):
            r = target_radii[i]
            # Left wall
            dist_l = centers[i, 0] - r
            if dist_l < 0.1:
                forces[i, 0] += wall_strength * (0.1 - dist_l)
            # Right wall
            dist_r = (1.0 - centers[i, 0]) - r
            if dist_r < 0.1:
                forces[i, 0] -= wall_strength * (0.1 - dist_r)
            # Bottom wall
            dist_b = centers[i, 1] - r
            if dist_b < 0.1:
                forces[i, 1] += wall_strength * (0.1 - dist_b)
            # Top wall
            dist_t = (1.0 - centers[i, 1]) - r
            if dist_t < 0.1:
                forces[i, 1] -= wall_strength * (0.1 - dist_t)

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    d = 1e-12
                    dx = 1e-12
                    dy = 0.0
                # Desired separation
                sep = target_radii[i] + target_radii[j]
                if d < sep * 1.5:  # Only interact if reasonably close
                    # Hertzian repulsion: F ~ (sep - d)^2
                    overlap = sep - d
                    if overlap > 0:
                        f = repel_strength * overlap * overlap / d
                    else:
                        # Small attraction to maintain contact (helps packing)
                        f = -repel_strength * 0.01 * (-overlap) / d
                    fx = f * dx / d
                    fy = f * dy / d
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Velocity update with damping
        vel = vel * damping + forces * dt
        centers = centers + vel * dt

        # Hard clip to stay in bounds
        for i in range(n):
            r = target_radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r * 0.5, 1.0 - r * 0.5)
            centers[i, 1] = np.clip(centers[i, 1], r * 0.5, 1.0 - r * 0.5)

    return centers


def _compute_max_radii(centers, n):
    """Compute the maximum possible radius for each circle given current positions."""
    radii = np.zeros(n)
    for i in range(n):
        r_max = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if j == i:
                continue
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                        (centers[i, 1] - centers[j, 1])**2)
            # We want r_i + r_j <= d, but r_j is also unknown.
            # Conservative: assume r_j = 0 for max r_i estimate
            # Better: iterative
            r_max = min(r_max, d)  # will be refined
        radii[i] = max(0.0, r_max * 0.9)
    return radii


def _iterative_radius_growth(centers, n, max_growth_iters=50):
    """
    Iteratively grow radii and relax positions.
    Start with small radii, grow them, relax, repeat.
    """
    # Initial small radii
    radii = np.full(n, 0.02)

    for it in range(max_growth_iters):
        # Relax positions with current target radii
        centers = _force_relax(centers, radii, n_steps=500, dt=0.004)

        # Compute new target radii: for each circle, max radius given others
        for i in range(n):
            r_max = min(centers[i, 0], 1.0 - centers[i, 0],
                        centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if j == i:
                    continue
                d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                            (centers[i, 1] - centers[j, 1])**2)
                # r_i <= d - r_j (use current r_j as estimate)
                r_max = min(r_max, d - radii[j])
            radii[i] = max(0.001, min(r_max, radii[i] * 1.15 + 0.005))

        # Check time
        if _remaining() < 15.0:
            break

    # Final relaxation with the grown radii
    centers = _force_relax(centers, radii, n_steps=1000, dt=0.003)

    # Final radius computation
    for i in range(n):
        r_max = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if j == i:
                continue
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                        (centers[i, 1] - centers[j, 1])**2)
            r_max = min(r_max, d - radii[j])
        radii[i] = max(0.0, r_max * 0.999)

    return centers, radii


def _polish(centers, radii, n, n_iters=200):
    """
    Local polish: iteratively adjust each circle to maximize its radius
    given neighbors, then re-relax.
    """
    for it in range(n_iters):
        if _remaining() < 10.0:
            break
        # For each circle, try to increase its radius
        for i in range(n):
            r_max = min(centers[i, 0], 1.0 - centers[i, 0],
                        centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if j == i:
                    continue
                d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                            (centers[i, 1] - centers[j, 1])**2)
                r_max = min(r_max, d - radii[j])
            radii[i] = max(radii[i], min(r_max, radii[i] * 1.02))

        # Quick relaxation
        centers = _force_relax(centers, radii, n_steps=100, dt=0.002)

    return centers, radii


def _hex_initial(n_circles, jitter=0.0):
    """Generate a hexagonal packing initialization."""
    rows = 5
    cols = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * dx + (0.5 * dx if r % 2 == 1 else 0)
            y = (r + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    centers = centers[:n_circles]
    return np.array(centers)


def _grid_initial(n_circles, jitter=0.0):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(n_circles)))
    dx = 1.0 / side
    dy = 1.0 / side
    centers = []
    for j in range(side):
        for i in range(side):
            if len(centers) >= n_circles:
                break
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    return np.array(centers[:n_circles])


def _random_initial(n_circles, seed=None):
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(seed)
    centers = []
    min_dist = 0.12
    attempts = 0
    while len(centers) < n_circles and attempts < 20000:
        attempts += 1
        x = rng.uniform(0.08, 0.92)
        y = rng.uniform(0.08, 0.92)
        ok = True
        for cx, cy in centers:
            if (x - cx)**2 + (y - cy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            centers.append((x, y))
    while len(centers) < n_circles:
        x = rng.uniform(0.08, 0.92)
        y = rng.uniform(0.08, 0.92)
        centers.append((x, y))
    return np.array(centers)


def construct():
    n = N
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Multiple initializations
    inits = []

    # Hexagonal with different jitters
    for jitter in [0.0, 0.01, 0.02, 0.03]:
        c = _hex_initial(n, jitter=jitter)
        inits.append(c)

    # Grid
    for jitter in [0.0, 0.015]:
        c = _grid_initial(n, jitter=jitter)
        inits.append(c)

    # Random
    for seed in [42, 123, 7, 99, 256]:
        c = _random_initial(n, seed=seed)
        inits.append(c)

    # Optimize from each start
    for idx, c_init in enumerate(inits):
        if _remaining() < 20.0:
            break

        centers, radii = _iterative_radius_growth(c_init.copy(), n,
                                                   max_growth_iters=40)
        centers, radii = _polish(centers, radii, n, n_iters=100)

        # Fix and validate
        centers_fixed, radii_fixed = _fix_packing(centers, radii)
        if _is_valid(centers_fixed, radii_fixed, tol=1e-5):
            total = np.sum(radii_fixed)
            if total > best_sum:
                best_sum = total
                best_centers = centers_fixed
                best_radii = radii_fixed
        else:
            # Try harder fix
            centers_fixed, radii_fixed = _fix_packing(centers, radii)
            if _is_valid(centers_fixed, radii_fixed, tol=1e-4):
                total = np.sum(radii_fixed)
                if total > best_sum:
                    best_sum = total
                    best_centers = centers_fixed
                    best_radii = radii_fixed

    # If no valid result, fall back to simple grid
    if best_centers is None:
        side = int(np.ceil(np.sqrt(n)))
        dx = 1.0 / side
        dy = 1.0 / side
        centers = []
        radii = []
        for j in range(side):
            for i in range(side):
                if len(centers) >= n:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                r = 0.5 * dx - 1e-3
                centers.append((x, y))
                radii.append(r)
        best_centers = np.array(centers[:n])
        best_radii = np.array(radii[:n])

    result = np.column_stack([best_centers, best_radii])
    return result

# EVOLVE-BLOCK-END
