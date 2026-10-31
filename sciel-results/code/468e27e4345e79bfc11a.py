"""Circle packing in unit square - 26 circles maximizing sum of radii.
Completely rewritten with force-directed relaxation, constraint-based SLSQP,
and multi-restart local search.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _pairwise_distances(pos):
    """Compute pairwise distance matrix (N, N)."""
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    return np.sqrt(dx * dx + dy * dy)

def _force_directed(pos, radii, n_iters=5000, dt_init=0.005, dt_end=0.0001, seed=42):
    """
    Force-directed relaxation: circles repel each other and are attracted
    toward the center. Radii are also adjusted via gradient of objective.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for it in range(n_iters):
        frac = it / max(n_iters - 1, 1)
        dt = dt_init * (dt_end / dt_init) ** frac

        # Compute forces
        forces = np.zeros_like(pos)
        radius_grad = np.zeros(n)

        # Wall forces: push circles inward if they touch/exceed boundary
        for i in range(n):
            x, y = pos[i]
            r = radii[i]
            # Left wall
            if x - r < 0:
                forces[i, 0] += 10.0 * (r - x)
                radius_grad[i] -= 1.0
            # Right wall
            if x + r > 1:
                forces[i, 0] -= 10.0 * (x + r - 1)
                radius_grad[i] -= 1.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += 10.0 * (r - y)
                radius_grad[i] -= 1.0
            # Top wall
            if y + r > 1:
                forces[i, 1] -= 10.0 * (y + r - 1)
                radius_grad[i] -= 1.0

        # Pairwise repulsion
        dist = _pairwise_distances(pos)
        np.fill_diagonal(dist, np.inf)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]

        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                md = min_dist[i, j]
                if d < md and d > 1e-10:
                    overlap = md - d
                    # Force along direction from j to i
                    fx = (pos[i, 0] - pos[j, 0]) / d
                    fy = (pos[i, 1] - pos[j, 1]) / d
                    forces[i, 0] += 5.0 * overlap * fx
                    forces[i, 1] += 5.0 * overlap * fy
                    forces[j, 0] -= 5.0 * overlap * fx
                    forces[j, 1] -= 5.0 * overlap * fy
                    # Shrink radii slightly when overlapping
                    radius_grad[i] -= 0.5 * overlap
                    radius_grad[j] -= 0.5 * overlap

        # Attraction toward center (weak)
        for i in range(n):
            forces[i, 0] += 0.01 * (0.5 - pos[i, 0])
            forces[i, 1] += 0.01 * (0.5 - pos[i, 1])

        # Update positions
        for i in range(n):
            pos[i, 0] += dt * forces[i, 0]
            pos[i, 1] += dt * forces[i, 1]
            # Clamp to [0,1]
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)

        # Update radii
        for i in range(n):
            radii[i] += dt * radius_grad[i] * 0.5
            # Enforce max radius from walls
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            radii[i] = np.clip(radii[i], 1e-4, max(0.001, max_r))

        # Add small noise to escape local minima
        if it < n_iters * 0.7:
            noise_scale = 0.001 * (1 - frac)
            for i in range(n):
                pos[i, 0] += rng.normal(0, noise_scale)
                pos[i, 1] += rng.normal(0, noise_scale)
                pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
                pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)

        # Periodic check
        if it % 200 == 0:
            pp, rr = _post_process(pos, radii)
            if _is_valid(pp, rr):
                s = np.sum(rr)
                if s > best_score:
                    best_score = s
                    best_pos = pp.copy()
                    best_radii = rr.copy()

    return best_pos, best_radii

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)

    for _ in range(300):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r * 0.99999:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r * 0.99999:
                    k = (dist / (sum_r + 1e-12)) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False

        if valid:
            break

    return pos, radii

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dist = _pairwise_distances(pos)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
    return True

def _init_grid(seed=0):
    """Grid-based initialization with slight jitter."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 5x5 grid = 25, plus one extra
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N - 1:
                pos[k] = [(i + 0.5) / 5.0 + rng.normal(0, 0.02), (j + 0.5) / 5.0 + rng.normal(0, 0.02)]
                radii[k] = 0.095
                k += 1
    pos[k] = [0.5 + rng.normal(0, 0.05), 0.5 + rng.normal(0, 0.05)]
    radii[k] = 0.05
    return pos, radii

def _init_hex(seed=0):
    """Hexagonal packing initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.115
    dx = r * np.sqrt(3) * 1.01
    dy = r * 1.5 * 1.01
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r + 0.01
        for col in range(5):
            if k >= N:
                break
            x = r + offset + col * dx + 0.01
            if x + r <= 1.0 and y + r <= 1.0 and x - r >= 0 and y - r >= 0:
                pos[k] = [x + rng.normal(0, 0.005), y + rng.normal(0, 0.005)]
                radii[k] = r * 0.95
                k += 1
        if k >= N:
            break
    while k < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        max_r = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            max_r = min(max_r, d - radii[j])
        if max_r > 0.02:
            pos[k] = [x, y]
            radii[k] = max_r * 0.9
            k += 1
        else:
            pos[k] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
            radii[k] = 0.02
            k += 1
    return pos, radii

def _init_circles_26(seed=0):
    """Specific layout: 4 large corner circles + 4 edge + 18 small."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 4 corner circles
    corners = [(0.18, 0.18), (0.82, 0.18), (0.18, 0.82), (0.82, 0.82)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.17
    # 4 edge midpoints
    edges = [(0.5, 0.12), (0.5, 0.88), (0.12, 0.5), (0.88, 0.5)]
    for i, (ex, ey) in enumerate(edges):
        pos[i + 4] = [ex, ey]
        radii[i + 4] = 0.11
    # Center circle
    pos[8] = [0.5, 0.5]
    radii[8] = 0.14
    # 17 remaining: scattered
    for i in range(9, N):
        found = False
        for _ in range(500):
            x = rng.uniform(0.08, 0.92)
            y = rng.uniform(0.08, 0.92)
            max_r = min(x, 1 - x, y, 1 - y)
            for j in range(i):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                max_r = min(max_r, d - radii[j])
            if max_r > 0.025:
                pos[i] = [x, y]
                radii[i] = max_r * 0.92
                found = True
                break
        if not found:
            pos[i] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
            radii[i] = 0.02
    return pos, radii

def _slsqp_constrained(x0, deadline):
    """SLSQP with hard non-overlap and boundary constraints."""
    x = x0.copy()
    pos = x[:2 * N].reshape(N, 2)
    radii = x[2 * N:]

    def c_boundary(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        return np.concatenate([p[:, 0] - r, 1.0 - p[:, 0] - r, p[:, 1] - r, 1.0 - p[:, 1] - r])

    def c_overlap(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                d2 = dx * dx + dy * dy
                md = r[i] + r[j]
                cons.append(d2 - md * md)
        return np.array(cons)

    constraints = [
        {'type': 'ineq', 'fun': c_boundary},
        {'type': 'ineq', 'fun': c_overlap}
    ]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

    def obj(x):
        return -np.sum(x[2 * N:])

    try:
        result = minimize(obj, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success or (not result.success and result.fun < obj(x)):
            x = result.x
    except Exception:
        pass
    return x[:2 * N].reshape(N, 2), np.maximum(x[2 * N:], 1e-4)

def _penalty_objective(x, w=1000.0):
    """Penalty-based objective for L-BFGS-B."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv ** 2)
    dist = _pairwise_distances(pos)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, min_dist - dist)
    overlap_pen = np.sum(viol[mask] ** 2)
    return -np.sum(radii) + w * (boundary_pen + overlap_pen)

def _local_search(pos, radii, n_restarts=3, steps=2000, deadline=None):
    """Local search with random perturbations and force-directed refinement."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        p = pos.copy()
        r = radii.copy()
        # Small random perturbation
        rng = np.random.RandomState(1000 + restart)
        p += rng.normal(0, 0.005, p.shape)
        p = np.clip(p, 0.0, 1.0)
        r = np.clip(r * (1 + rng.normal(0, 0.05, N)), 1e-4, 0.5)
        p, r = _force_directed(p, r, n_iters=steps, dt_init=0.003, dt_end=0.00005,
                               seed=2000 + restart)
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin

    inits = []
    for seed in range(3):
        p, r = _init_grid(seed)
        inits.append((p, r))
    for seed in range(2):
        p, r = _init_hex(seed)
        inits.append((p, r))
    for seed in range(2):
        p, r = _init_circles_26(seed)
        inits.append((p, r))

    best_pos = inits[0][0].copy()
    best_radii = inits[0][1].copy()
    best_score = np.sum(best_radii)

    # Pass 1: force-directed relaxation on all inits
    for idx, (p0, r0) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        p, r = _force_directed(p0, r0, n_iters=4000, dt_init=0.004, dt_end=0.0001,
                               seed=idx * 7 + 1)
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()

    # Pass 2: SLSQP with hard constraints on top candidates
    for idx in range(min(len(inits), 4)):
        if time.time() > deadline - 20:
            break
        p0, r0 = inits[idx]
        x0 = np.concatenate([p0.flatten(), r0])
        p, r = _slsqp_constrained(x0, deadline)
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()

    # Also SLSQP on the current best
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        p, r = _slsqp_constrained(x0, deadline)
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()

    # Pass 3: L-BFGS-B penalty optimization
    for idx in range(min(len(inits), 3)):
        if time.time() > deadline - 15:
            break
        p0, r0 = inits[idx]
        x0 = np.concatenate([p0.flatten(), r0])
        bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.5)] * N
        try:
            result = minimize(_penalty_objective, x0, method='L-BFGS-B',
                              bounds=bounds, args=(2000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
            p = result.x[:2 * N].reshape(N, 2)
            r = np.maximum(result.x[2 * N:], 1e-4)
            pp, rr = _post_process(p, r)
            if _is_valid(pp, rr):
                s = np.sum(rr)
                if s > best_score:
                    best_score = s
                    best_pos = pp.copy()
                    best_radii = rr.copy()
        except Exception:
            pass

    # Pass 4: local search / multi-restart refinement
    if time.time() < deadline - 10:
        p, r = _local_search(best_pos, best_radii, n_restarts=4, steps=1500,
                             deadline=deadline)
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()

    # Final SLSQP polish
    if time.time() < deadline - 5:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        p, r = _slsqp_constrained(x0, deadline)
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
