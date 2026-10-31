"""Circle packing in perimeter-4 rectangle: Force-directed + Simulated Annealing."""
import numpy as np
import time
import math
import random

N = 21


def _is_valid(x, y, r, w, h, tol=1e-7):
    if w <= 0.01 or h <= 0.01:
        return False, 0.0
    if np.any(r < 0):
        return False, 0.0
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False, 0.0
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs - tol:
                return False, 0.0
    return True, float(np.sum(r))


def _max_radius_for_circle(i, x, y, r, w, h):
    """Compute the maximum radius circle i can have given current positions of others."""
    r_max = min(x[i], w - x[i], y[i], h - y[i])
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = math.sqrt(dx * dx + dy * dy)
        allowed = d - r[j]
        if allowed < r_max:
            r_max = allowed
    return max(0.0, r_max)


def _expand_radii(x, y, r, w, h, max_rounds=60):
    """Iteratively expand each radius to its geometric maximum."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_rounds):
        improved = False
        for i in range(N):
            r_new = _max_radius_for_circle(i, x, y, r, w, h)
            if r_new > r[i] + 1e-10:
                r[i] = r_new
                improved = True
        if not improved:
            break
    return x, y, r


def _force_resolve(x, y, r, w, h, iterations=200):
    """
    Iterative constraint projection: push overlapping circles apart and
    keep them within walls. This is a rigid-body relaxation step.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for _ in range(iterations):
        moved = False
        # Wall constraints
        for i in range(N):
            # Left
            if x[i] < r[i]:
                x[i] = r[i]
                moved = True
            # Right
            if x[i] > w - r[i]:
                x[i] = w - r[i]
                moved = True
            # Bottom
            if y[i] < r[i]:
                y[i] = r[i]
                moved = True
            # Top
            if y[i] > h - r[i]:
                y[i] = h - r[i]
                moved = True

        # Circle-circle constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                rs = r[i] + r[j]
                if d2 < rs * rs:
                    d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                    # Overlap: push apart along the line connecting centers
                    overlap = rs - d
                    if overlap > 0:
                        # Weight by inverse radius (smaller circles move more)
                        wi = r[j] / (r[i] + r[j] + 1e-12)
                        wj = r[i] / (r[i] + r[j] + 1e-12)
                        ux = dx / d
                        uy = dy / d
                        x[i] += wi * overlap * ux
                        y[i] += wi * overlap * uy
                        x[j] -= wj * overlap * ux
                        y[j] -= wj * overlap * uy
                        moved = True

        if not moved:
            break

    # Final wall clamp
    for i in range(N):
        x[i] = max(r[i], min(x[i], w - r[i]))
        y[i] = max(r[i], min(y[i], h - r[i]))

    return x, y, r


def _simulated_annealing(x0, y0, r0, w0, time_budget=15.0):
    """
    Simulated annealing: perturb positions, resolve constraints, expand radii,
    accept/reject based on temperature. Objective: maximize sum(r).
    """
    start = time.time()
    x = x0.copy().astype(np.float64)
    y = y0.copy().astype(np.float64)
    r = r0.copy().astype(np.float64)
    w = float(w0)
    h = 2.0 - w

    # Initial expansion
    x, y, r = _expand_radii(x, y, r, w, h, max_rounds=30)
    valid, best_val = _is_valid(x, y, r, w, h)
    if not valid:
        # Try to fix
        x, y, r = _force_resolve(x, y, r, w, h, iterations=100)
        x, y, r = _expand_radii(x, y, r, w, h, max_rounds=30)
        valid, best_val = _is_valid(x, y, r, w, h)
        if not valid:
            return x0.copy(), y0.copy(), r0.copy(), w0

    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
    current_val = best_val
    current_x, current_y, current_r, current_w = x.copy(), y.copy(), r.copy(), w
    current_h = h

    T = 0.05  # initial temperature
    T_min = 0.001
    cooling = 0.995

    rng = random.Random(42)

    while time.time() - start < time_budget and T > T_min:
        # Propose a move: perturb one or two circles
        n_perturb = rng.choice([1, 1, 1, 2, 2, 3])
        indices = rng.sample(range(N), n_perturb)

        new_x = current_x.copy()
        new_y = current_y.copy()
        new_r = current_r.copy()
        new_w = current_w
        new_h = current_h

        for idx in indices:
            # Random displacement
            dx = rng.gauss(0, 0.03)
            dy = rng.gauss(0, 0.03)
            new_x[idx] += dx
            new_y[idx] += dy
            # Clamp to walls
            new_x[idx] = max(new_r[idx], min(new_x[idx], new_w - new_r[idx]))
            new_y[idx] = max(new_r[idx], min(new_y[idx], new_h - new_r[idx]))

        # Occasionally perturb aspect ratio
        if rng.random() < 0.1:
            dw = rng.gauss(0, 0.02)
            new_w = max(0.3, min(1.7, current_w + dw))
            new_h = 2.0 - new_w
            # Re-clamp all to new walls
            for i in range(N):
                new_x[i] = max(new_r[i], min(new_x[i], new_w - new_r[i]))
                new_y[i] = max(new_r[i], min(new_y[i], new_h - new_r[i]))

        # Resolve constraints
        new_x, new_y, new_r = _force_resolve(new_x, new_y, new_r, new_w, new_h, iterations=50)
        # Expand radii
        new_x, new_y, new_r = _expand_radii(new_x, new_y, new_r, new_w, new_h, max_rounds=20)

        valid, new_val = _is_valid(new_x, new_y, new_r, new_w, new_h)
        if not valid:
            T *= cooling
            continue

        # Accept/reject
        delta = new_val - current_val
        if delta > 0 or rng.random() < math.exp(delta / T):
            current_x, current_y, current_r, current_w = new_x, new_y, new_r, new_w
            current_h = new_h
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_x, best_y, best_r, best_w = new_x.copy(), new_y.copy(), new_r.copy(), new_w

        T *= cooling

    return best_x, best_y, best_r, best_w


def _hex_init(w, h):
    """Generate a hexagonal lattice initial configuration."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    # Estimate radius for hex packing
    # For N circles in w x h, hex packing has density ~pi/(2*sqrt(3))
    area = w * h
    r_est = math.sqrt(area * math.pi / (2 * math.sqrt(3) * N * math.pi))
    r_est = min(r_est, 0.15)
    r_est = max(r_est, 0.03)

    dx = 2 * r_est
    dy = dx * math.sqrt(3) / 2

    # Fill hex lattice
    col = 0
    row = 0
    placed = 0
    while placed < N:
        cx = (col + 0.5) * dx
        cy = (row + 0.5) * dy + (0.5 * dy if col % 2 else 0)
        if cx + r_est <= w and cy + r_est <= h:
            x[placed] = cx
            y[placed] = cy
            r[placed] = r_est
            placed += 1
        col += 1
        if cx + r_est > w:
            col = 0
            row += 1

    return x, y, r


def _random_init(w, h, seed=0):
    """Random initial configuration."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        r0 = rng.uniform(0.03, 0.08)
        x[i] = rng.uniform(r0, w - r0)
        y[i] = rng.uniform(r0, h - r0)
        r[i] = r0
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    best_value = 0.0
    best_result = None

    # Generate diverse initial configurations
    configs = []
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x, y, r = _hex_init(w0, h0)
        configs.append((x, y, r, w0, h0))

    # Add random inits
    for s in range(6):
        w0 = random.uniform(0.5, 1.5)
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x, y, r = _random_init(w0, h0, seed=100 + s)
        configs.append((x, y, r, w0, h0))

    for idx, (x0, y0, r0, w0, h0) in enumerate(configs):
        if time.time() > deadline - 10:
            break

        remaining = deadline - time.time()
        sa_time = min(12.0, remaining - 5.0)
        if sa_time < 2.0:
            break

        # Run simulated annealing
        x, y, r, w = _simulated_annealing(x0, y0, r0, w0, time_budget=sa_time)
        h = 2.0 - w

        # Final expansion
        x, y, r = _expand_radii(x, y, r, w, h, max_rounds=80)

        # Validate
        valid, val = _is_valid(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Final polish: one more expansion + force resolve
    if best_result is not None and time.time() < deadline - 3:
        x, y, r, w, h = best_result
        x, y, r = _force_resolve(x, y, r, w, h, iterations=50)
        x, y, r = _expand_radii(x, y, r, w, h, max_rounds=80)
        valid, val = _is_valid(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
