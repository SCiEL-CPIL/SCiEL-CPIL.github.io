"""Circle packing in unit square – kinetic-annealing + local polish."""
import numpy as np
import time

N = 26


def _enforce_bounds(x, y, r):
    """Clamp centers so circles stay inside [0,1]^2."""
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return x, y


def _compute_forces(x, y, r, temperature):
    """
    Compute repulsive forces between overlapping circles and
    a gentle attractive force toward the center of the square.
    Returns (fx, fy, r_vec) where r_vec is the overlap vector for each pair.
    """
    n = len(r)
    fx = np.zeros(n)
    fy = np.zeros(n)

    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-12:
                # Overlap amount
                overlap = min_dist - dist
                # Force magnitude proportional to overlap, scaled by temperature
                strength = overlap * (1.0 + temperature * 2.0)
                ux = dx / dist
                uy = dy / dist
                fx[i] += strength * ux
                fy[i] += strength * uy
                fx[j] -= strength * ux
                fy[j] -= strength * uy
            elif dist <= 1e-12:
                # Nearly coincident: push apart randomly
                angle = np.random.uniform(0, 2 * np.pi)
                strength = min_dist * 0.5
                fx[i] += strength * np.cos(angle)
                fy[i] += strength * np.sin(angle)
                fx[j] -= strength * np.cos(angle)
                fy[j] -= strength * np.sin(angle)

    # Attractive force toward center (0.5, 0.5)
    cx = 0.5
    cy = 0.5
    for i in range(n):
        fx[i] += 0.02 * (cx - x[i])
        fy[i] += 0.02 * (cy - y[i])

    return fx, fy


def _kinetic_anneal(x0, y0, r0, n_iter=5000, dt=0.002, temp_start=1.0, temp_end=0.01,
                    damping=0.95, deadline=None):
    """
    Kinetic annealing: treat circles as particles with repulsive collisions
    and gentle centering. Adaptive temperature and damping.
    """
    n = len(r0)
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    vx = np.zeros(n)
    vy = np.zeros(n)

    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = -np.inf

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Temperature schedule: linear cooling
        frac = it / max(n_iter - 1, 1)
        temp = temp_start + (temp_end - temp_start) * frac

        # Compute forces
        fx, fy = _compute_forces(x, y, r, temp)

        # Symplectic-like update: velocity Verlet simplified
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt

        x = x + vx * dt
        y = y + vy * dt

        # Enforce bounds
        x, y = _enforce_bounds(x, y, r)

        # Re-zero velocities after boundary clamp (energy loss)
        # Check validity and score
        valid = True
        for i in range(n):
            if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9:
                valid = False
                break
            if y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
                valid = False
                break
        if valid:
            # Check pairwise non-overlap
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break

        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    return best_x, best_y, best_r


def _local_polish(x, y, r, n_iter=2000, deadline=None):
    """
    Local gradient-descent-style polish: push overlapping circles apart
    and nudge toward better packing. Uses a simple iterative relaxation.
    """
    n = len(r)
    x = x.copy()
    y = y.copy()

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    # Push apart along the connecting line, weighted by inverse radius
                    w_i = r[j] / (r[i] + r[j])
                    w_j = r[i] / (r[i] + r[j])
                    ux = dx / dist
                    uy = dy / dist
                    x[i] += 0.5 * overlap * w_i * ux
                    y[i] += 0.5 * overlap * w_i * uy
                    x[j] -= 0.5 * overlap * w_j * ux
                    y[j] -= 0.5 * overlap * w_j * uy
                    moved = True
                elif dist <= 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    x[i] += 0.5 * min_dist * 0.1 * np.cos(angle)
                    y[i] += 0.5 * min_dist * 0.1 * np.sin(angle)
                    x[j] -= 0.5 * min_dist * 0.1 * np.cos(angle)
                    y[j] -= 0.5 * min_dist * 0.1 * np.sin(angle)
                    moved = True

        # Enforce bounds
        x, y = _enforce_bounds(x, y, r)

        if not moved:
            break

    return x, y


def _is_valid(x, y, r, tol=1e-9):
    n = len(r)
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _random_init_with_repulsion(n, seed=None):
    """
    Generate random positions, then relax overlaps to produce a valid
    initial configuration with non-overlapping circles of equal radius.
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with equal radii
    r_base = 0.5 / np.sqrt(n) * 0.95  # slightly smaller to fit
    r = np.full(n, r_base)

    # Random initial positions
    x = np.random.uniform(r, 1.0 - r, size=n)
    y = np.random.uniform(r, 1.0 - r, size=n)

    # Relax overlaps
    for _ in range(500):
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    max_overlap = max(max_overlap, overlap)
                    ux = dx / dist
                    uy = dy / dist
                    x[i] += 0.5 * overlap * ux
                    y[i] += 0.5 * overlap * uy
                    x[j] -= 0.5 * overlap * ux
                    y[j] -= 0.5 * overlap * uy
        x, y = _enforce_bounds(x, y, r)
        if max_overlap < 1e-10:
            break

    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_result = None
    best_score = -np.inf

    n_restarts = 8

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        # Generate initial configuration
        x0, y0, r0 = _random_init_with_repulsion(N, seed=42 + restart * 1000)

        # Kinetic annealing phase
        x1, y1, r1 = _kinetic_anneal(
            x0, y0, r0,
            n_iter=3000,
            dt=0.002,
            temp_start=0.8,
            temp_end=0.02,
            damping=0.96,
            deadline=deadline
        )

        # Local polish phase
        x2, y2 = _local_polish(x1, y1, r1, n_iter=1500, deadline=deadline)

        # Check validity
        if _is_valid(x2, y2, r1):
            score = np.sum(r1)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x2, y2, r1])

        # If we haven't found a valid result yet, try a simpler approach
        if best_result is None and time.time() < deadline:
            # Fallback: grid-based
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            fallback = np.array(cells[:N])
            if _is_valid(fallback[:, 0], fallback[:, 1], fallback[:, 2]):
                best_result = fallback
                best_score = np.sum(fallback[:, 2])

    if best_result is None:
        # Absolute fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
