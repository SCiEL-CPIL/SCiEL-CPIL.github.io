"""Circle packing in a perimeter-4 rectangle.
Structural rewrite: physics-based force relaxation + hard constraint repair,
hexagonal seeding, aspect-ratio sweep, and a radius-growth phase.
"""
import time
import numpy as np

N = 21
_DEADLINE = 150.0
_TIME0 = time.time()
_TIME_BUDGET = _DEADLINE - 10.0  # safety margin


def _remaining():
    return _TIME_BUDGET - (time.time() - _TIME0)


def _is_valid(x, y, r, w, h, tol=1e-7):
    n = len(r)
    if np.any(r < -tol):
        return False
    r = np.maximum(r, 0.0)
    if np.any(x < r - tol) or np.any(x > w - r - tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r - tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    s = r[:, None] + r[None, :]
    if np.any(d2 < (s - tol) ** 2):
        return False
    return True


def _clamp_to_box(x, y, r, w, h):
    """Push circles back inside the rectangle (hard constraint)."""
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    return x, y


def _separate(x, y, r, iters=80, damping=0.5):
    """Iteratively resolve pairwise overlaps by moving circles apart."""
    n = len(r)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.hypot(dx, dy)
                s = r[i] + r[j]
                if d < s - 1e-12:
                    if d < 1e-12:
                        dx = 1e-6
                        dy = 0.0
                        d = 1e-6
                    overlap = (s - d) * 0.5 * damping
                    ux = dx / d
                    uy = dy / d
                    x[i] -= overlap * ux
                    y[i] -= overlap * uy
                    x[j] += overlap * ux
                    y[j] += overlap * uy
                    moved = True
        if not moved:
            break
    return x, y


def _generate_hex_init(w, h, rng):
    """Hexagonal lattice seeding for N circles in a w x h rectangle."""
    n = N
    # Determine a reasonable spacing
    # Hexagonal packing density ~ 0.9069
    # Estimate radius from area
    area = w * h
    r_est = np.sqrt(area * 0.9 / (np.pi * n)) * 0.8
    # Hex lattice spacing
    a = 2.0 * r_est
    dx = a
    dy = a * np.sqrt(3) / 2.0
    xs, ys = [], []
    row = 0
    while len(xs) < n:
        offset = (dx / 2.0) if (row % 2 == 1) else 0.0
        col = 0
        while len(xs) < n:
            x = offset + col * dx
            y = row * dy
            xs.append(x)
            ys.append(y)
            col += 1
        row += 1
    xs = np.array(xs[:n])
    ys = np.array(ys[:n])
    # Center the lattice in the rectangle
    xs -= xs.mean()
    ys -= ys.mean()
    xs += w / 2.0
    ys += h / 2.0
    # Add small random jitter to break symmetry
    xs += rng.uniform(-0.05 * r_est, 0.05 * r_est, n)
    ys += rng.uniform(-0.05 * r_est, 0.05 * r_est, n)
    # Clamp to box
    r0 = np.full(n, r_est * 0.9)
    xs, ys = _clamp_to_box(xs, ys, r0, w, h)
    return xs, ys, r0


def _generate_grid_init(w, h, rng):
    """Grid seeding for N circles."""
    n = N
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = [], []
    for idx in range(n):
        i = idx % side
        j = idx // side
        xs.append((i + 0.5) * w / side)
        ys.append((j + 0.5) * h / side)
    xs = np.array(xs)
    ys = np.array(ys)
    r_est = 0.5 * min(w / side, h / side) * 0.85
    r0 = np.full(n, r_est)
    xs += rng.uniform(-0.02 * r_est, 0.02 * r_est, n)
    ys += rng.uniform(-0.02 * r_est, 0.02 * r_est, n)
    xs, ys = _clamp_to_box(xs, ys, r0, w, h)
    return xs, ys, r0


def _force_relax(x, y, r, w, h, time_budget, stiffness=50.0):
    """Physics-based force relaxation with hard constraint enforcement.
    Uses a simple Verlet-like integration with repulsive forces.
    """
    n = len(r)
    t0 = time.time()
    # Initialize velocities to zero
    vx = np.zeros(n)
    vy = np.zeros(n)
    dt = 0.01
    damping = 0.95
    max_iter = 2000

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.hypot(dx, dy)
                s = r[i] + r[j]
                if d < 1e-12:
                    dx = 1e-6
                    dy = 0.0
                    d = 1e-6
                if d < s:
                    # Strong repulsion when overlapping
                    overlap = s - d
                    f = stiffness * overlap
                    ux = dx / d
                    uy = dy / d
                    fx[i] -= f * ux
                    fy[i] -= f * uy
                    fx[j] += f * ux
                    fy[j] += f * uy
        # Boundary repulsion (soft push)
        for i in range(n):
            # Left wall
            if x[i] - r[i] < 0:
                fx[i] += stiffness * (r[i] - x[i])
            # Right wall
            if w - x[i] - r[i] < 0:
                fx[i] -= stiffness * (w - x[i] - r[i])
            # Bottom wall
            if y[i] - r[i] < 0:
                fy[i] += stiffness * (r[i] - y[i])
            # Top wall
            if h - y[i] - r[i] < 0:
                fy[i] -= stiffness * (h - y[i] - r[i])

        # Update velocities and positions
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        x += vx * dt
        y += vy * dt

        # Hard constraint: clamp to box
        x, y = _clamp_to_box(x, y, r, w, h)

        # Check convergence: max velocity
        v_max = np.max(np.hypot(vx, vy))
        if v_max < 1e-6:
            break

    # Final separation pass to ensure no overlaps
    x, y = _separate(x, y, r, iters=100, damping=0.7)
    x, y = _clamp_to_box(x, y, r, w, h)
    return x, y


def _grow_radii(x, y, r, w, h, time_budget, growth_steps=20):
    """Iteratively grow radii while re-relaxing positions.
    This directly maximizes sum(r) by expanding circles.
    """
    n = len(r)
    t0 = time.time()
    # Start with current radii, try to grow them
    for step in range(growth_steps):
        if time.time() - t0 > time_budget:
            break
        # Try to grow all radii by a small factor
        growth = 1.0 + 0.02  # 2% growth per step
        r_new = r * growth
        # Relax positions with new radii
        x_new, y_new = _force_relax(x, y, r_new, w, h,
                                     time_budget=(time_budget - (time.time() - t0)) * 0.5,
                                     stiffness=80.0)
        # Check validity
        if _is_valid(x_new, y_new, r_new, w, h):
            x, y, r = x_new, y_new, r_new
        else:
            # Try smaller growth
            growth = 1.0 + 0.005
            r_new = r * growth
            x_new, y_new = _force_relax(x, y, r_new, w, h,
                                         time_budget=(time_budget - (time.time() - t0)) * 0.5,
                                         stiffness=80.0)
            if _is_valid(x_new, y_new, r_new, w, h):
                x, y, r = x_new, y_new, r_new
            else:
                # Revert and try even smaller
                growth = 1.0 + 0.001
                r_new = r * growth
                x_new, y_new = _force_relax(x, y, r_new, w, h,
                                             time_budget=(time_budget - (time.time() - t0)) * 0.5,
                                             stiffness=100.0)
                if _is_valid(x_new, y_new, r_new, w, h):
                    x, y, r = x_new, y_new, r_new
                else:
                    break
    return x, y, r


def _solve_aspect(w, h, rng, time_budget):
    """Solve the packing problem for a given aspect ratio (w, h)."""
    n = N
    best_val = -np.inf
    best = None

    # Try multiple initializations
    for init_type in ['hex', 'grid', 'random']:
        if time_budget - (time.time() - _TIME0) < 2.0:
            break
        if init_type == 'hex':
            xs, ys, r = _generate_hex_init(w, h, rng)
        elif init_type == 'grid':
            xs, ys, r = _generate_grid_init(w, h, rng)
        else:
            xs = rng.uniform(0.1, w - 0.1, n)
            ys = rng.uniform(0.1, h - 0.1, n)
            r_est = 0.08 * min(w, h)
            r = np.full(n, r_est)
            xs, ys = _clamp_to_box(xs, ys, r, w, h)

        # Phase 1: Relax positions with initial radii
        x1, y1 = _force_relax(xs, ys, r, w, h, time_budget=time_budget * 0.3,
                               stiffness=50.0)
        if not _is_valid(x1, y1, r, w, h):
            # Try to repair
            x1, y1 = _separate(x1, y1, r, iters=150, damping=0.8)
            x1, y1 = _clamp_to_box(x1, y1, r, w, h)
            if not _is_valid(x1, y1, r, w, h):
                continue

        # Phase 2: Grow radii
        time_left = time_budget - (time.time() - _TIME0)
        if time_left < 1.0:
            continue
        x2, y2, r2 = _grow_radii(x1, y1, r, w, h, time_budget=time_left * 0.7)

        if _is_valid(x2, y2, r2, w, h):
            val = np.sum(r2)
            if val > best_val:
                best_val = val
                best = (x2.copy(), y2.copy(), r2.copy())

    if best is not None:
        return best
    return None


def construct():
    n = N
    rng = np.random.default_rng(42)

    # Aspect ratios to try (w + h = 2)
    aspects = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0]:
        w = 2.0 / (1.0 + t)
        h = 2.0 - w
        if 0.3 < w < 5.0 and 0.3 < h < 5.0:
            aspects.append((w, h))

    best_val = -np.inf
    best = None

    # Time budget per aspect
    n_aspects = len(aspects)
    time_per_aspect = max(1.0, _TIME_BUDGET / (n_aspects + 5))

    for w, h in aspects:
        if _remaining() < 3.0:
            break
        result = _solve_aspect(w, h, rng, time_budget=time_per_aspect)
        if result is not None:
            x, y, r = result
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best = (x, y, r, w, h)

    if best is None:
        # Fallback: simple grid packing
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for idx in range(n):
            i = idx % side
            j = idx // side
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = 0.5 * min(w / side, h / side) * 0.85
            cells.append((x, y, r))
        return np.array(cells[:n])

    x, y, r, w, h = best
    out = np.empty((n, 3))
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = r
    return out
