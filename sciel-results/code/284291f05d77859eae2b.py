"""Pack 50 equal circles in a unit square, maximize common radius.

Strategy:
  1. Multi-initialization (7x7+1, hexagonal, jittered 8x8, random)
  2. Hard-disk radius expansion
  3. L-BFGS-B soft-min polish (multi-temperature, sparse via Delaunay)
  4. SLSQP epigraph on sparse contact graph
  5. Active-set Newton solve on binding constraints
  6. High-precision final polish
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0


def _radius_fast(c):
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _contact_graph(c, r, margin=0.15):
    """Return (pairs, walls) of constraints near the minimum radius."""
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    tol = 2 * r + margin * r
    pairs = np.argwhere(d < tol)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    wall_dist = 0.5 - np.abs(c)
    walls = []
    for i in range(N):
        for k in range(2):
            if wall_dist[i, k] < r + margin * r:
                walls.append((i, k))
    return pairs, walls


def _soft_min_radius(c, t, pairs, walls):
    """Soft-min of critical constraints only."""
    wall = 0.5 - np.abs(c)
    terms = []
    for i, k in walls:
        terms.append(wall[i, k])
    for i, j in pairs:
        terms.append(np.linalg.norm(c[i] - c[j]) / 2.0)
    if len(terms) == 0:
        return 0.0
    arr = np.array(terms)
    m = arr.min()
    s = np.exp(-(arr - m) / t)
    return m - t * np.log(s.sum())


def _soft_min_neg(x, t, pairs, walls):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t, pairs, walls)


def _soft_min_grad(x, t, pairs, walls):
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    terms = []
    wall_idx = []
    for i, k in walls:
        terms.append(wall[i, k])
        wall_idx.append((i, k))
    pair_terms = []
    for i, j in pairs:
        pair_terms.append(np.linalg.norm(c[i] - c[j]) / 2.0)
    if len(terms) == 0 and len(pair_terms) == 0:
        return np.zeros(2 * N)
    all_terms = np.concatenate([np.array(terms), np.array(pair_terms)]) if pair_terms else np.array(terms)
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()
    grad = np.zeros(2 * N)
    # Wall terms
    for idx, (i, k) in enumerate(wall_idx):
        w = exps[idx] / Z
        grad[2 * i + k] -= w * np.sign(c[i, k])
    # Pair terms
    for idx, (i, j) in enumerate(pairs):
        w = exps[len(terms) + idx] / Z
        dij = c[i] - c[j]
        norm = np.linalg.norm(dij)
        if norm > 1e-14:
            g = dij / (2.0 * norm)
            grad[2 * i:2 * i + 2] += w * g
            grad[2 * j:2 * j + 2] -= w * g
    return grad


def _relax_hard_disk(c, R, iters, eta0, eta1):
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 15:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=20.0):
    """Multi-temperature L-BFGS-B with sparse contact graph."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    temps = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        c_cur = x.reshape(N, 2)
        r_cur = _radius_fast(c_cur)
        pairs, walls = _contact_graph(c_cur, r_cur, margin=0.2)
        try:
            res = minimize(
                lambda xx: _soft_min_neg(xx, t, pairs, walls),
                x,
                jac=lambda xx: _soft_min_grad(xx, t, pairs, walls),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_x = res.x.copy()
                x = res.x.copy()
        except Exception:
            pass
    return best_x.reshape(N, 2), best_r


def _slsqp_sparse(c, t0, time_budget=30.0):
    """SLSQP epigraph on sparse contact graph."""
    c = c.copy()
    r_init = _radius_fast(c)
    pairs, walls = _contact_graph(c, r_init, margin=0.3)

    def objective(x):
        return -x[-1]

    constraints = []
    for i, j in pairs:
        def _c(x, i=i, j=j):
            ci = x[2 * i:2 * i + 2]
            cj = x[2 * j:2 * j + 2]
            r = x[-1]
            return np.sqrt(np.sum((ci - cj) ** 2)) - 2 * r
        constraints.append({'type': 'ineq', 'fun': _c})
    for i, k in walls:
        def _c(x, i=i, k=k):
            return 0.5 - np.abs(x[2 * i + k]) - x[-1]
        constraints.append({'type': 'ineq', 'fun': _c})

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]
    best_x = x0.copy()
    best_r = r_init

    for maxiter in [200, 400, 600]:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success or res.fun < objective(x0):
                r_new = res.x[-1]
                c_new = res.x[:2 * N].reshape(N, 2)
                r_verified = _radius_fast(c_new)
                if r_verified > best_r:
                    best_r = r_verified
                    best_x = res.x.copy()
                    x0 = res.x.copy()
                    # Update contact graph
                    pairs, walls = _contact_graph(c_new, r_verified, margin=0.3)
                    constraints = []
                    for i, j in pairs:
                        def _c(x, i=i, j=j):
                            ci = x[2 * i:2 * i + 2]
                            cj = x[2 * j:2 * j + 2]
                            r = x[-1]
                            return np.sqrt(np.sum((ci - cj) ** 2)) - 2 * r
                        constraints.append({'type': 'ineq', 'fun': _c})
                    for i, k in walls:
                        def _c(x, i=i, k=k):
                            return 0.5 - np.abs(x[2 * i + k]) - x[-1]
                        constraints.append({'type': 'ineq', 'fun': _c})
        except Exception:
            pass
    return best_x[:2 * N].reshape(N, 2), best_r


def _active_set_newton(c, r, t0, time_budget=15.0):
    """Active-set Newton: solve for displacement that increases min radius."""
    c = c.copy()
    r_cur = _radius_fast(c)
    for _ in range(30):
        if (time.time() - t0) > t0 + time_budget:
            break
        pairs, walls = _contact_graph(c, r_cur, margin=0.05)
        if len(pairs) == 0 and len(walls) == 0:
            break
        # Build system: for each active constraint, direction of increase
        # We want to find displacement d that maximizes the minimum constraint
        # Use a simple approach: sum of unit vectors pointing away from constraints
        disp = np.zeros_like(c)
        for i, j in pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm > 1e-14:
                u = dij / norm
                disp[i] += u
                disp[j] -= u
        for i, k in walls:
            s = -np.sign(c[i, k])
            if c[i, k] == 0:
                s = 1.0
            disp[i, k] += s
        max_d = np.max(np.linalg.norm(disp, axis=1))
        if max_d < 1e-14:
            break
        disp = disp / max_d * min(0.1 * r_cur, 1e-4)
        c_try = c + disp
        c_try = np.clip(c_try, -0.5, 0.5)
        r_try = _radius_fast(c_try)
        if r_try > r_cur + 1e-13:
            c = c_try
            r_cur = r_try
        else:
            # Try smaller step
            for frac in [0.5, 0.1, 0.01]:
                c_try2 = c + frac * disp
                c_try2 = np.clip(c_try2, -0.5, 0.5)
                r_try2 = _radius_fast(c_try2)
                if r_try2 > r_cur + 1e-13:
                    c = c_try2
                    r_cur = r_try2
                    break
            else:
                break
    return c, r_cur


def _high_precision_polish(c, r, t0, time_budget=10.0):
    c = c.copy()
    r_cur = _radius_fast(c)
    for step_scale in [5e-6, 2e-6, 1e-6, 5e-7]:
        if (time.time() - t0) > t0 + time_budget:
            break
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=float)
        directions /= np.sqrt(2)
        for _ in range(30):
            if (time.time() - t0) > t0 + time_budget:
                break
            improved = False
            for i in range(N):
                for d in directions:
                    c_try = c.copy()
                    c_try[i] += step_scale * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-13:
                        c = c_try
                        r_cur = r_try
                        improved = True
            if not improved:
                break
    return c, r_cur


def _init_7x7_plus():
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.02, 0.02]])
    return np.vstack([g, extra])


def _init_hexagonal(rng):
    s = 0.14
    c = []
    for row in range(8):
        y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2
        if y > 0.5 - s / 2:
            break
        offset = 0 if row % 2 == 0 else s / 2
        for col in range(8):
            x = -0.5 + s / 2 + col * s + offset
            if x <= 0.5 - s / 2:
                c.append([x, y])
    c = np.array(c[:N])
    if len(c) < N:
        while len(c) < N:
            c = np.vstack([c, rng.uniform(-0.4, 0.4, 2)])
    c = c[:N]
    c += rng.uniform(-0.01, 0.01, c.shape)
    c = np.clip(c, -0.45, 0.45)
    return c


def _init_jittered_8x8(rng):
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.02, 0.02, c.shape)
    return c


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
    ]

    best_c = None
    best_r = -1.0

    # Phase 1: Hard-disk radius expansion
    for start in inits:
        if (time.time() - t0) > DEADLINE - 50:
            break
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 40:
        budget = min(15.0, DEADLINE - 40 - (time.time() - t0))
        if budget > 5:
            c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=budget)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished

    # Phase 3: SLSQP sparse epigraph
    if (time.time() - t0) < DEADLINE - 25:
        budget = min(20.0, DEADLINE - 25 - (time.time() - t0))
        if budget > 5:
            c_slsqp, r_slsqp = _slsqp_sparse(best_c, t0, time_budget=budget)
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_c = c_slsqp

    # Phase 4: Active-set Newton
    if (time.time() - t0) < DEADLINE - 10:
        c_ns, r_ns = _active_set_newton(best_c, best_r, t0, time_budget=10.0)
        if r_ns > best_r:
            best_r = r_ns
            best_c = c_ns

    # Phase 5: High-precision polish
    if (time.time() - t0) < DEADLINE - 3:
        budget = min(8.0, DEADLINE - 3 - (time.time() - t0))
        if budget > 2:
            c_hp, r_hp = _high_precision_polish(best_c, best_r, t0, time_budget=budget)
            if r_hp > best_r:
                best_r = r_hp
                best_c = c_hp

    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_c
