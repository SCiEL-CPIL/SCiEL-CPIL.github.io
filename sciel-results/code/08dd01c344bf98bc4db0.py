"""Physics-based (position-based dynamics) circle packing in the unit square."""
import numpy as np
import time

N = 32


def _init_positions(n):
    """Hexagonal-ish initial layout with overlap."""
    positions = np.zeros((n, 2))
    col = 0
    row = 0
    dx = 0.30
    dy = 0.28
    x0 = 0.30
    y0 = 0.28
    idx = 0
    while idx < n:
        x = x0 + col * dx + (0.15 if row % 2 else 0.0)
        y = y0 + row * dy
        positions[idx, 0] = x
        positions[idx, 1] = y
        idx += 1
        col += 1
        if col > 5:
            col = 0
            row += 1
    # Shuffle deterministically to break symmetry
    rng = np.random.default_rng(12345)
    perm = rng.permutation(n)
    return positions[perm]


def _init_radii(n):
    """Initial radii: start small, grow during simulation."""
    # Start with a radius that allows significant overlap
    return np.full(n, 0.025)


def _project_constraints(pos, rad):
    """
    Project positions to resolve circle-circle overlaps and wall penetrations.
    Returns updated positions.
    """
    n = len(pos)
    pos = pos.copy()

    # Wall constraints: for each circle, x - r >= 0, x + r <= 1, etc.
    for i in range(n):
        r = rad[i]
        if pos[i, 0] < r:
            pos[i, 0] = r
        elif pos[i, 0] > 1.0 - r:
            pos[i, 0] = 1.0 - r
        if pos[i, 1] < r:
            pos[i, 1] = r
        elif pos[i, 1] > 1.0 - r:
            pos[i, 1] = 1.0 - r

    # Circle-circle constraints: iterative projection
    for _iter in range(20):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        dx, dy = 0.01, 0.01
                        dist = np.sqrt(dx * dx + dy * dy)
                    # Push apart along the line connecting centers
                    overlap = (min_dist - dist) / 2.0
                    ux = dx / dist
                    uy = dy / dist
                    # Weight by inverse radius (smaller circles move more)
                    wi = 1.0 / (rad[i] + 1e-8)
                    wj = 1.0 / (rad[j] + 1e-8)
                    total_w = wi + wj
                    pos[i, 0] -= overlap * ux * (wi / total_w)
                    pos[i, 1] -= overlap * uy * (wi / total_w)
                    pos[j, 0] += overlap * ux * (wj / total_w)
                    pos[j, 1] += overlap * uy * (wj / total_w)
                    moved = True

        # Re-apply wall constraints after circle-circle projection
        for i in range(n):
            r = rad[i]
            if pos[i, 0] < r:
                pos[i, 0] = r
            elif pos[i, 0] > 1.0 - r:
                pos[i, 0] = 1.0 - r
            if pos[i, 1] < r:
                pos[i, 1] = r
            elif pos[i, 1] > 1.0 - r:
                pos[i, 1] = 1.0 - r

        if not moved:
            break

    return pos


def _is_valid(pos, rad):
    """Check if the configuration is valid (disjoint interiors, inside square)."""
    n = len(pos)
    eps = 1e-9
    for i in range(n):
        if pos[i, 0] - rad[i] < -eps or pos[i, 0] + rad[i] > 1.0 + eps:
            return False
        if pos[i, 1] - rad[i] < -eps or pos[i, 1] + rad[i] > 1.0 + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rad[i] + rad[j] - eps:
                return False
    return True


def _objective(pos, rad):
    """Sum of radii (the objective to maximize)."""
    return np.sum(rad)


def _grow_and_project(pos, rad, max_radius, time_limit):
    """
    Iteratively grow radii and project to resolve constraints.
    Uses a multiplicative growth factor.
    """
    start_time = time.time()
    n = len(pos)
    growth_factor = 1.005  # small growth per step

    # Try to grow all radii uniformly
    for step in range(2000):
        if time.time() - start_time > time_limit:
            break

        # Check current validity
        if not _is_valid(pos, rad):
            # Try to fix by projecting
            pos = _project_constraints(pos, rad)
            if not _is_valid(pos, rad):
                # Shrink slightly and retry
                rad *= 0.999
                pos = _project_constraints(pos, rad)
                continue

        # Try growing
        new_rad = rad * growth_factor
        # Cap at max_radius
        new_rad = np.minimum(new_rad, max_radius)

        # Project with new radii
        new_pos = _project_constraints(pos, new_rad)

        # Check if the projected configuration is valid
        if _is_valid(new_pos, new_rad):
            pos = new_pos
            rad = new_rad
        else:
            # Try a smaller growth
            smaller_growth = 1.001
            new_rad2 = rad * smaller_growth
            new_rad2 = np.minimum(new_rad2, max_radius)
            new_pos2 = _project_constraints(pos, new_rad2)
            if _is_valid(new_pos2, new_rad2):
                pos = new_pos2
                rad = new_rad2
            else:
                # Can't grow more with current positions
                # Try perturbing positions to find a better packing
                # Random perturbation of positions
                rng = np.random.default_rng(step)
                perturbation = rng.normal(0, 0.01, size=(n, 2))
                perturbed_pos = pos + perturbation
                # Keep within bounds
                for i in range(n):
                    r = rad[i]
                    perturbed_pos[i, 0] = np.clip(perturbed_pos[i, 0], r, 1.0 - r)
                    perturbed_pos[i, 1] = np.clip(perturbed_pos[i, 1], r, 1.0 - r)
                projected_pos = _project_constraints(perturbed_pos, rad)
                if _is_valid(projected_pos, rad):
                    pos = projected_pos

    return pos, rad


def _polish(pos, rad, time_limit):
    """
    Final polish: use scipy.optimize to maximize sum of radii
    with constraints.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return pos, rad

    start_time = time.time()
    n = len(pos)

    # Variables: positions (2n) and radii (n)
    x0 = np.concatenate([pos.flatten(), rad])

    def objective(x):
        pos_x = x[:2 * n].reshape(n, 2)
        rad_x = x[2 * n:]
        return -np.sum(rad_x)  # minimize negative sum

    # Constraints:
    # 1. Wall constraints: x_i - r_i >= 0, x_i + r_i <= 1, etc.
    # 2. Circle-circle: ||p_i - p_j|| >= r_i + r_j
    # 3. r_i >= 0

    constraints = []

    # Wall constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2 * i] - x[2 * n + i]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[2 * i] - x[2 * n + i]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2 * i + 1] - x[2 * n + i]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[2 * i + 1] - x[2 * n + i]
        })

    # Circle-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[2 * j] - x[2 * i]) ** 2 +
                    (x[2 * j + 1] - x[2 * i + 1]) ** 2 -
                    (x[2 * n + i] + x[2 * n + j]) ** 2
                )
            })

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.0, 0.5))  # radius

    result = None
    # Try SLSQP
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        if time.time() - start_time > time_limit:
            return pos, rad
    except Exception:
        pass

    # Try COBYLA as fallback
    if result is None or not result.success:
        try:
            # COBYLA uses inequality constraints as g(x) >= 0
            cobyla_constraints = []
            for i in range(n):
                cobyla_constraints.append(
                    lambda x, i=i: x[2 * i] - x[2 * n + i]
                )
                cobyla_constraints.append(
                    lambda x, i=i: 1.0 - x[2 * i] - x[2 * n + i]
                )
                cobyla_constraints.append(
                    lambda x, i=i: x[2 * i + 1] - x[2 * n + i]
                )
                cobyla_constraints.append(
                    lambda x, i=i: 1.0 - x[2 * i + 1] - x[2 * n + i]
                )
            for i in range(n):
                for j in range(i + 1, n):
                    cobyla_constraints.append(
                        lambda x, i=i, j=j: (
                            (x[2 * j] - x[2 * i]) ** 2 +
                            (x[2 * j + 1] - x[2 * i + 1]) ** 2 -
                            (x[2 * n + i] + x[2 * n + j]) ** 2
                        )
                    )

            result = minimize(
                objective, x0, method='COBYLA',
                bounds=bounds, constraints=cobyla_constraints,
                options={'maxiter': 1000, 'rhobeg': 1e-4}
            )
        except Exception:
            pass

    if result is not None and result.success:
        new_pos = result.x[:2 * n].reshape(n, 2)
        new_rad = result.x[2 * n:]
        if _is_valid(new_pos, new_rad):
            return new_pos, new_rad

    return pos, rad


def construct():
    """
    Pack 32 circles in the unit square, maximizing sum of radii.
    Returns array of shape (32, 3): rows (x, y, radius).
    """
    start_time = time.time()
    time_limit = 140.0  # safety margin before 150s kill

    pos = _init_positions(N)
    rad = _init_radii(N)

    # Phase 1: Grow and project
    max_radius = 0.15  # reasonable upper bound for individual radius
    pos, rad = _grow_and_project(pos, rad, max_radius, time_limit - 20.0)

    # Phase 2: Polish with scipy
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        pos, rad = _polish(pos, rad, remaining_time)

    # Final validity check
    if not _is_valid(pos, rad):
        # Fall back to a valid grid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return np.column_stack([pos, rad])
