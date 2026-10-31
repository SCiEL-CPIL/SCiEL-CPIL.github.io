"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: constraint-based local search (grow-until-blocked) + SLSQP polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(radii < 0):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

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
            if r > max_r + 1e-12:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    k = (dist / (sum_r + 1e-12)) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _grow_one(i, pos, radii):
    """Grow circle i as much as possible given current positions and other radii."""
    x, y = pos[i]
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(N):
        if j == i:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - radii[j])
    return max(max_r, 0.0)

def _grow_all(pos, radii, n_passes=80):
    """Iteratively grow each circle until blocked (coordinate-wise local search)."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    for _ in range(n_passes):
        changed = False
        for i in range(N):
            new_r = _grow_one(i, pos, radii)
            if new_r > radii[i] + 1e-10:
                radii[i] = new_r
                changed = True
        if not changed:
            break
    return pos, radii

def _random_nudge(pos, radii, rng, step=0.02):
    """Nudge one circle randomly, then regrow all."""
    pos = pos.copy()
    radii = radii.copy()
    i = rng.randint(N)
    pos[i, 0] += rng.normal(0, step)
    pos[i, 1] += rng.normal(0, step)
    pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
    pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
    # After moving, shrink to validity then regrow
    pos_pp, radii_pp = _post_process(pos, radii)
    pos_pp, radii_pp = _grow_all(pos_pp, radii_pp, n_passes=30)
    return pos_pp, radii_pp

def _local_search(pos, radii, n_iter=800, seed=0, deadline=None):
    """Constraint-based local search: grow + random nudges."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    # Initial grow
    pos, radii = _grow_all(pos, radii, n_passes=50)
    score = np.sum(radii)
    if score > best_score:
        best_score = score
        best_pos = pos.copy()
        best_radii = radii.copy()

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        step = 0.03 * (1.0 - it / max(n_iter, 1)) + 0.005
        new_pos, new_radii = _random_nudge(pos, radii, rng, step=step)
        new_score = np.sum(new_radii)
        if new_score > np.sum(radii) - 1e-6:
            pos, radii = new_pos, new_radii
            if new_score > best_score:
                best_score = new_score
                best_pos = pos.copy()
                best_radii = radii.copy()
    return best_pos, best_radii

def _slsqp_polish(pos, radii, deadline):
    """SLSQP with explicit nonlinear constraints for circle-circle and walls."""
    n = len(radii)
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))

    # Linear constraints: x_i - r_i >= 0, etc.
    # We'll use nonlinear constraints for all.

    def obj(x):
        return -np.sum(x[2 * n:])

    def grad_obj(x):
        g = np.zeros_like(x)
        g[2 * n:] = -1.0
        return g

    constraints = []
    # Wall constraints (linear, but express as nonlinear for simplicity)
    for i in range(n):
        xi, yi, ri = 2 * i, 2 * i + 1, 2 * n + i
        constraints.append({'type': 'ineq', 'fun': lambda x, a=xi, b=ri: x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=xi, b=ri: 1.0 - x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=yi, b=ri: x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=yi, b=ri: 1.0 - x[a] - x[b]})

    # Circle-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = 2 * i, 2 * i + 1
            xj, yj = 2 * j, 2 * j + 1
            ri, rj = 2 * n + i, 2 * n + j
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, a=xi, b=yi, c=xj, d=yj, e=ri, f=rj:
                    np.sqrt((x[a] - x[c]) ** 2 + (x[b] - x[d]) ** 2) - x[e] - x[f]
            })

    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, jac=grad_obj,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if result.success or result.fun < obj(x0) - 1e-8:
            x = result.x
            pos_out = x[:2 * n].reshape(n, 2)
            radii_out = np.maximum(x[2 * n:], 1e-6)
            return pos_out, radii_out
    except Exception:
        pass
    return pos, radii

def _init_grid():
    """5x5 grid minus center, plus center small - deterministic."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            if k >= N - 1:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.09
            k += 1
    pos[k] = [0.5, 0.5]
    radii[k] = 0.03
    return pos, radii

def _init_corners():
    """Corner + edge biased initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    rng = np.random.RandomState(42)
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.15
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5),
             (0.3, 0.08), (0.7, 0.08), (0.3, 0.92), (0.7, 0.92),
             (0.08, 0.3), (0.08, 0.7), (0.92, 0.3), (0.92, 0.7)]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos[i] = edges[i - 4]
            radii[i] = 0.10
        else:
            pos[i] = [rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)]
            radii[i] = 0.05
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    inits = [
        _init_grid(),
        _init_corners(),
    ]

    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        pos_init, rad_init = _post_process(pos_init, rad_init)
        pos_init, rad_init = _grow_all(pos_init, rad_init, n_passes=50)
        score = np.sum(rad_init)
        if score > best_score:
            best_score = score
            best_pos = pos_init.copy()
            best_radii = rad_init.copy()

        # Local search
        if time.time() < deadline - 15:
            pos_ls, rad_ls = _local_search(pos_init, rad_init, n_iter=600, seed=idx * 1000 + 1, deadline=deadline)
            pos_pp, rad_pp = _post_process(pos_ls, rad_ls)
            score = np.sum(rad_pp)
            if _is_valid(pos_pp, rad_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()

    # SLSQP polish on best
    if time.time() < deadline - 10:
        pos_pol, rad_pol = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_pol, rad_pol)
        score = np.sum(rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
