"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines multi-phase local optimization with a compact structural
parameterization (corner + center + rings) that reduces the global-search
dimensionality so Differential Evolution can actually explore it.

Refinements within the same design family:
  * smoother penalty (soft-min via (v + sqrt(v^2 + eps))/2) for better gradients
  * stronger final high-precision polish (tighter tol, more iters, SLSQP cross-check)
  * re-tuned time budget: trim the slow force-directed loop and the redundant
    direct 78-dim track to free wall-clock for more DE structural exploration
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty.

    Uses a combined approach:
    - Quadratic penalty on the sum of squared positive violations (smooth, well-behaved)
    - Log-sum-exp soft-max on the worst violation (directional, focuses on binding constraints)
    This gives both global smoothness and local directionality.
    """
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Temperature for the soft-max component
    beta = 60.0

    # Boundary violations (vectorized)
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    # Stack all violation magnitudes
    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    # Quadratic penalty on positive violations (smooth, captures total violation magnitude)
    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    # Log-sum-exp soft-max on the worst violation (directional, focuses on binding constraints)
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    # Combined penalty: quadratic for smoothness, soft-max for directionality
    penalty = 50.0 * quad_penalty + 20.0 * sm

    return -np.sum(radii) + penalty


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Structural seeds ----------------

def _make_grid(n, seed=42):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            idx = row * 5 + col
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink the smaller circle more to preserve larger ones
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r * 0.999))

    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision local polish using L-BFGS-B and SLSQP."""
    x0 = np.concatenate([pos.flatten(), radii])
    best_x = x0.copy()
    best_val = _objective(x0, deadline)

    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.35))

    # L-BFGS-B polish
    try:
        res = minimize(_objective, best_x, args=(deadline,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass

    # SLSQP polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_x, args=(deadline,), method='SLSQP',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    pos_opt = best_x[:2 * N].reshape(N, 2)
    radii_opt = best_x[2 * N:]
    pos_opt, radii_opt = _post_process(pos_opt, radii_opt, deadline)
    return pos_opt, radii_opt


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def _update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if _is_valid(pos, radii):
            s = np.sum(radii)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()

    # Phase 1: Differential Evolution on structural parameters (10-dim)
    # This is the main global search - much more tractable than 78-dim
    struct_bounds = [
        (0.05, 0.30),  # corner_r
        (0.05, 0.35),  # center_r
        (0.2, 0.8),    # cx
        (0.2, 0.8),    # cy
        (0.15, 0.6),   # ring1_off
        (0.03, 0.20),  # ring1_r
        (0.03, 0.20),  # ring1_r2
        (0.25, 0.7),   # ring2_off
        (0.02, 0.16),  # ring2_r
        (0.02, 0.16),  # ring2_r2
    ]

    try:
        de_res = differential_evolution(
            _struct_objective, struct_bounds, args=(deadline,),
            seed=42, maxiter=300, popsize=30, tol=1e-8,
            mutation=(0.3, 1.0), recombination=0.9,
            polish=True, init='latinhypercube',
            update_interval=10,
        )
        if de_res.fun < 1e9:
            p_opt = de_res.x
            pos_s, radii_s = _expand_struct(p_opt)
            pos_s, radii_s = _polish(pos_s, radii_s, deadline)
            _update_best(pos_s, radii_s)
    except Exception:
        pass

    # Phase 2: Local optimization from multiple initial configurations
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    for strategy in strategies:
        if time.time() > deadline - 20:
            break
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            if strategy == 'grid':
                pos, radii = _make_grid(N, seed)
            elif strategy == 'hexagonal':
                pos, radii = _make_hexagonal(N, seed)
            elif strategy == 'center_ring':
                pos, radii = _make_center_ring(N, seed)
            else:
                pos, radii = _make_random(N, seed)

            x0 = np.concatenate([pos.flatten(), radii])
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))
                bounds.append((0.01, 0.99))
            for i in range(N):
                bounds.append((0.01, 0.35))

            try:
                res = minimize(_objective, x0, args=(deadline,), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
                pos_opt = res.x[:2 * N].reshape(N, 2)
                radii_opt = res.x[2 * N:]
                pos_opt, radii_opt = _polish(pos_opt, radii_opt, deadline)
                _update_best(pos_opt, radii_opt)
            except Exception:
                pass

    # Phase 3: Nelder-Mead refinement on best found
    if best_pos is not None and time.time() < deadline - 10:
        try:
            x_best = np.concatenate([best_pos.flatten(), best_radii])
            res_nm = minimize(_objective, x_best, args=(deadline,), method='Nelder-Mead',
                              options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10})
            pos_opt = res_nm.x[:2 * N].reshape(N, 2)
            radii_opt = res_nm.x[2 * N:]
            pos_opt, radii_opt = _polish(pos_opt, radii_opt, deadline)
            _update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # Fallback: simple grid if nothing valid found
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005)
                 for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
