"""Circle packing in unit square - force-directed simulated annealing."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 143.0  # safety margin before 150s kill


def _force_energy(params, n, stiffness=50.0, boundary_stiff=100.0, center_pull=0.05):
    """Compute (energy, forces) for the force-directed system.
    params: flat array of length 3n.
    Returns energy (scalar) and forces (flat array, to be added to params).
    """
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]

    # Pairwise interactions (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 1.0)  # avoid div by zero
    min_d = r[:, None] + r[None, :]

    # Overlap amount (positive when overlapping)
    overlap = np.maximum(0.0, min_d - d)
    # Force magnitude for overlap (repulsive)
    # Use a strong spring: F = stiffness * overlap
    fmag = stiffness * overlap
    np.fill_diagonal(fmag, 0.0)

    # Unit vectors
    inv_d = 1.0 / d
    inv_d = np.where(d > 1e-12, inv_d, 0.0)
    ux = dx * inv_d
    uy = dy * inv_d

    fx = np.sum(fmag * ux, axis=1)
    fy = np.sum(fmag * uy, axis=1)

    # Boundary forces: push inward if circle too close to wall
    # For x: if r - x > 0, push in +x direction
    bfx = np.maximum(0.0, r - x) * boundary_stiff + np.maximum(0.0, r - (1 - x)) * boundary_stiff
    bfy = np.maximum(0.0, r - y) * boundary_stiff + np.maximum(0.0, r - (1 - y)) * boundary_stiff

    # Center pull: gentle attraction toward center to keep cluster compact
    cfx = center_pull * (0.5 - x)
    cfy = center_pull * (0.5 - y)

    # Radius: encourage larger radii (negative force on r means increase)
    # We want to maximize sum of radii, so apply a constant "growth" force
    # but limited by constraints. Use a small constant push.
    fr = -0.1  # negative = increase radius (since we add force to params)

    # Total forces
    Fx = fx + bfx + cfx
    Fy = fy + bfy + cfy
    Fr = np.full(n, fr)

    forces = np.zeros(3 * n)
    forces[0::3] = Fx
    forces[1::3] = Fy
    forces[2::3] = Fr

    # Energy: sum of overlap^2 + boundary violations^2 - sum(r)
    energy = np.sum(overlap ** 2) * stiffness * 0.5
    energy += np.sum(np.maximum(0.0, r - x) ** 2) * boundary_stiff * 0.5
    energy += np.sum(np.maximum(0.0, r - (1 - x)) ** 2) * boundary_stiff * 0.5
    energy += np.sum(np.maximum(0.0, r - y) ** 2) * boundary_stiff * 0.5
    energy += np.sum(np.maximum(0.0, r - (1 - y)) ** 2) * boundary_stiff * 0.5
    energy -= np.sum(r)

    return energy, forces


def _sa_optimize(params, n, time_limit=10.0, seed=42):
    """Simulated annealing optimization."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    params = params.copy()

    # Schedule: start with large kicks, cool down
    T_start = 0.02
    T_end = 0.0001
    n_steps = 5000
    # Adaptive: run until time limit
    step_size = 0.01

    for step in range(n_steps):
        if time.time() - t0 > time_limit:
            break

        # Cooling schedule
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        current_step = step_size * (1.0 - 0.8 * progress)

        # Compute forces
        energy, forces = _force_energy(params, n, stiffness=50.0, boundary_stiff=100.0, center_pull=0.05)

        # Propose move: forces + random noise
        noise = rng.normal(0, current_step, size=3 * n)
        new_params = params + forces * 0.01 + noise

        # Clip to reasonable bounds
        for i in range(n):
            new_params[3*i] = np.clip(new_params[3*i], 0.01, 0.99)
            new_params[3*i+1] = np.clip(new_params[3*i+1], 0.01, 0.99)
            new_params[3*i+2] = np.clip(new_params[3*i+2], 0.01, 0.49)

        new_energy, _ = _force_energy(new_params, n, stiffness=50.0, boundary_stiff=100.0, center_pull=0.05)

        # Metropolis acceptance
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            params = new_params

    return params


def _penalty_objective(params, n, lam=500.0):
    """Smooth penalty objective for L-BFGS polish."""
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]

    # Boundary violations
    pen = np.sum(np.maximum(0, r - x) ** 2)
    pen += np.sum(np.maximum(0, r - (1 - x)) ** 2)
    pen += np.sum(np.maximum(0, r - y) ** 2)
    pen += np.sum(np.maximum(0, r - (1 - y)) ** 2)

    # Non-overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    min_d = r[:, None] + r[None, :]
    overlap = np.maximum(0, min_d - d)
    pen += np.sum(overlap ** 2)

    return -np.sum(r) + lam * pen


def _local_polish(params, n, time_limit=5.0):
    """L-BFGS-B polish."""
    t0 = time.time()
    try:
        res = minimize(
            lambda p: _penalty_objective(p, n),
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() - t0 < time_limit:
            if res.fun < _penalty_objective(params, n):
                return res.x
    except Exception:
        pass
    return params


def _is_valid(pts, tol=1e-6):
    """Check validity."""
    n = len(pts)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol) or np.any(y - r < -tol) or np.any(1 - y - r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    min_d = r[:, None] + r[None, :]
    if np.any(d < min_d - tol):
        return False
    return True


def _post_process(pts, n):
    """Enforce constraints by shrinking radii and adjusting positions."""
    pts = pts.copy()
    for _ in range(100):
        changed = False
        for i in range(n):
            x, y, r = pts[i]
            r = min(r, x, 1 - x, y, 1 - y)
            if r < 0.001:
                r = 0.001
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            if abs(pts[i, 0] - x) > 1e-10 or abs(pts[i, 1] - y) > 1e-10 or abs(pts[i, 2] - r) > 1e-10:
                changed = True
            pts[i] = [x, y, r]

        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2)
                min_d = pts[i, 2] + pts[j, 2]
                if d < min_d - 1e-7:
                    shrink = (min_d - d) / 2 + 1e-6
                    pts[i, 2] = max(0.001, pts[i, 2] - shrink)
                    pts[j, 2] = max(0.001, pts[j, 2] - shrink)
                    changed = True
        if not changed:
            break
    return pts


def _make_hex_initial(n):
    """Create a hexagonal close-packed initial layout."""
    pts = []
    # Hexagonal packing: rows offset
    # For 26 circles, try a 5x6 or similar arrangement
    # Use hexagonal grid with spacing ~ 0.2
    spacing = 0.18
    row_h = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(8):
            if len(pts) >= n:
                break
            x = 0.1 + col * spacing + (row % 2) * spacing / 2
            y = 0.1 + row * row_h
            if x < 0.95 and y < 0.95:
                pts.append([x, y, spacing * 0.45])
        if len(pts) >= n:
            break
    # Fill remaining with random positions
    rng = np.random.RandomState(0)
    while len(pts) < n:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        pts.append([x, y, 0.08])
    return np.array(pts[:n]).flatten()


def _make_random_initial(n, seed):
    """Random initial layout."""
    rng = np.random.RandomState(seed)
    pts = np.array([
        [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.05, 0.15)]
        for _ in range(n)
    ])
    return pts.flatten()


def construct():
    n = N
    t_start = time.time()

    best_pts = None
    best_score = -np.inf

    # Strategy 1: Hexagonal initial + SA + polish
    if time.time() - t_start < _DEADLINE - 30:
        params = _make_hex_initial(n)
        params = _sa_optimize(params, n, time_limit=25.0, seed=42)
        if time.time() - t_start < _DEADLINE - 5:
            params = _local_polish(params, n, time_limit=10.0)
        pts = params.reshape(n, 3)
        pts = _post_process(pts, n)
        if _is_valid(pts):
            score = np.sum(pts[:, 2])
            if score > best_score:
                best_score = score
                best_pts = pts.copy()

    # Strategy 2: Multiple random restarts with SA
    if time.time() - t_start < _DEADLINE - 10:
        for seed in [123, 456, 789, 101112, 202020]:
            if time.time() - t_start > _DEADLINE - 5:
                break
            params = _make_random_initial(n, seed)
            params = _sa_optimize(params, n, time_limit=15.0, seed=seed)
            if time.time() - t_start < _DEADLINE - 5:
                params = _local_polish(params, n, time_limit=5.0)
            pts = params.reshape(n, 3)
            pts = _post_process(pts, n)
            if _is_valid(pts):
                score = np.sum(pts[:, 2])
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()

    # Fallback
    if best_pts is None or not _is_valid(best_pts):
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.85
                pts.append([x, y, r])
        best_pts = _post_process(np.array(pts[:n]), n)

    return best_pts
# EVOLVE-BLOCK-END
