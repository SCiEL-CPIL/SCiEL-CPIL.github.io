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
    """Objective: minimize negative sum of radii + smooth soft-min penalty."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Smooth soft-min: approximates max(0, v) with a smooth surrogate
    eps = 1e-6

    def _soft(v):
        return (v + np.sqrt(v * v + eps)) * 0.5

    # Boundary violations (vectorized) - smooth
    left = _soft(radii - pos[:, 0])
    right = _soft(pos[:, 0] - (1.0 - radii))
    bot = _soft(radii - pos[:, 1])
    top = _soft(pos[:, 1] - (1.0 - radii))
    bpen = np.sum(left * left) + np.sum(right * right) + np.sum(bot * bot) + np.sum(top * top)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)
    viol_s = _soft(viol)
    np.fill_diagonal(viol_s, 0.0)
    open_pen = 0.5 * np.sum(viol_s * viol_s)

    return -np.sum(radii) + 100.0 * (bpen + open_pen)


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
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2500, 'ftol': 1e-18, 'gtol': 1e-16})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """Constraint-based SLSQP polish as a cross-check / alternative local solve.
    Maximizes sum of radii subject to explicit non-overlap and boundary
    inequality constraints; often escapes shallow L-BFGS-B local minima."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    # Bounds: positions in [0,1], radii in [1e-4, 0.35]
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.35)] * N

    # Inequality constraints g(x) >= 0
    cons = []
    # Boundary: pos >= r and 1 - pos >= r
    for i in range(N):
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: (1.0 - x[2 * i]) - x[2 * N + i]})
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: x[2 * i + 1] - x[2 * N + i]})
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: (1.0 - x[2 * i + 1]) - x[2 * N + i]})
    # Non-overlap: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, j=j:
                         np.sqrt((x[2 * i] - x[2 * j]) ** 2 +
                                 (x[2 * i + 1] - x[2 * j + 1]) ** 2)
                         - x[2 * N + i] - x[2 * N + j]})

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    try:
        result = minimize(neg_sum_rad, x0, method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': 400, 'ftol': 1e-12})
        if result.x is not None:
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _force_directed_anneal(pos, radii, deadline, max_iter=2000, rng_seed=123):
    """Force-directed annealing with position-dependent step sizes and
    smarter radius growth to explore topological rearrangements."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(rng_seed)

    T = 0.05  # initial temperature
    cooling = 0.999

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T < 1e-6:
            break

        # Compute forces
        forces = np.zeros_like(pos)

        # Boundary forces: push circles back inside
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 10.0
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 10.0
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 10.0
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 10.0

        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    # Overlap: strong repulsion
                    overlap = min_dist - dist
                    f = overlap * 50.0 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                elif dist > min_dist and dist < min_dist * 1.5:
                    # Near-contact: mild attraction to maximize packing
                    f = (min_dist * 1.5 - dist) * 0.1 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy

        # Apply forces with position-dependent step size
        for i in range(N):
            norm_f = np.linalg.norm(forces[i]) + 1e-12
            # Smaller circles can move more; larger circles move less
            scale = 0.01 * T * (1.0 / (0.5 + radii[i]))
            delta = scale * (forces[i] / norm_f + rng.randn(2) * 0.5)
            pos[i] += delta
            # Clamp to unit square with margin
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)

        # Try to grow radii slightly if no overlaps
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
            # Grow radii: smaller circles grow faster (they have more slack)
            for i in range(N):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                grow_factor = 1.0 + 0.002 * (1.0 - radii[i])
                radii[i] = min(radii[i] * grow_factor, max_r * 0.999)

        T *= cooling

    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    struct_bounds = [
        (0.06, 0.28),   # corner_r
        (0.06, 0.32),   # center_r
        (0.3, 0.7),     # cx
        (0.3, 0.7),     # cy
        (0.18, 0.45),   # ring1 offset
        (0.04, 0.16),   # ring1 r
        (0.04, 0.16),   # ring1 r2
        (0.30, 0.55),   # ring2 offset
        (0.03, 0.12),   # ring2 r
        (0.03, 0.12),   # ring2 r2
    ]

    # ---------- Track A: compact structural parameterization ----------
    # DE over 10-dim structural space, then expand + polish.

    # First DE run (larger population, more iterations, smoother objective helps)
    if time.time() < deadline - 70:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=120, popsize=22, seed=7,
                                            tol=1e-9, polish=False,
                                            mutation=(0.4, 1.0), recombination=0.7,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                if time.time() < deadline - 40:
                    pos_p, radii_p = _polish(pos_s, radii_s, deadline)
                    if _is_valid(pos_p, radii_p):
                        s = np.sum(radii_p)
                        if s > best_score:
                            best_score = s
                            best_pos = pos_p.copy()
                            best_radii = radii_p.copy()
        except Exception:
            pass

    # Second DE run with different parameters to explore more broadly
    if time.time() < deadline - 50:
        try:
            de_res2 = differential_evolution(_struct_objective, struct_bounds,
                                             args=(deadline,),
                                             maxiter=80, popsize=18, seed=123,
                                             tol=1e-8, polish=False,
                                             mutation=(0.6, 1.2), recombination=0.8,
                                             updating='immediate')
            if de_res2.x is not None:
                pos_s2, radii_s2 = _expand_struct(de_res2.x)
                update_best(pos_s2, radii_s2)
                if time.time() < deadline - 30:
                    pos_p2, radii_p2 = _polish(pos_s2, radii_s2, deadline)
                    if _is_valid(pos_p2, radii_p2):
                        s2 = np.sum(radii_p2)
                        if s2 > best_score:
                            best_score = s2
                            best_pos = pos_p2.copy()
                            best_radii = radii_p2.copy()
        except Exception:
            pass

    # L-BFGS-B on structural vector from a few good seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.105, 0.105, 0.43, 0.065, 0.065],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 45:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B',
                           bounds=struct_bounds,
                           options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: direct 78-dim from structural seeds (safety net) ----------
    # Only run if we still have comfortable time; it is comparatively expensive.
    if time.time() < deadline - 35:
        configs = []
        pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
        configs.append((pos_s0, radii_s0))
        configs.append(_make_center_ring(N))

        for pos, radii in configs:
            if time.time() > deadline - 25:
                break
            x0 = np.concatenate([pos.flatten(), radii])
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))
                bounds.append((0.01, 0.99))
            for i in range(N):
                bounds.append((0.01, 0.32))
            try:
                result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 400, 'ftol': 1e-11})
                if result.success or result.fun < _objective(x0, deadline):
                    pos_opt = result.x[:2 * N].reshape(N, 2)
                    radii_opt = result.x[2 * N:]
                    update_best(pos_opt, radii_opt)
            except Exception:
                pass

    # ---------- Track C: Force-directed annealing on best configuration ----------
    if best_pos is not None and time.time() < deadline - 20:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline,
                                                       max_iter=1000, rng_seed=456)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    # ---------- Final high-precision polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 8:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # ---------- Final SLSQP cross-check polish ----------
    if best_pos is not None and time.time() < deadline - 3:
        pos_q, radii_q = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_q, radii_q):
            s = np.sum(radii_q)
            if s > best_score:
                best_score = s
                best_pos = pos_q.copy()
                best_radii = radii_q.copy()

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
