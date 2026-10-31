"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    bv = (np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) +
          np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii)))
    boundary_pen = np.sum(bv * bv)

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    boundary_pen = (np.sum(_soft_max(radii - pos[:, 0], beta)) +
                    np.sum(_soft_max(pos[:, 0] - (1.0 - radii), beta)) +
                    np.sum(_soft_max(radii - pos[:, 1], beta)) +
                    np.sum(_soft_max(pos[:, 1] - (1.0 - radii), beta)))

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    overlap_pen = np.sum(_soft_max((min_dist_sq - dist_sq)[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Vectorized feasibility repair: shrink radii to satisfy boundary & pairwise constraints."""
    pos = np.asarray(pos, dtype=float).copy()
    radii = np.asarray(radii, dtype=float).copy()
    # Boundary clamp
    max_b = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                       np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(radii, max_b * 0.99999)
    radii = np.maximum(radii, 1e-6)
    # Pairwise: iterate shrinkage (vectorized)
    for _ in range(80):
        changed = False
        # Compute all pair distances
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        # For each pair, max allowed sum = dist
        # Greedy: for each circle i, max_r_i = min over j of (dist[i,j] - radii[j])
        # Do a few passes of shrinking
        for _ in range(2):
            # For each i: r_i <= dist[i,j] - r_j for all j != i
            # max allowed for i = min_j (dist[i,j] - radii[j])
            allowed = dist - radii[np.newaxis, :]  # allowed[i,j] = dist[i,j] - radii[j]
            np.fill_diagonal(allowed, np.inf)
            max_allowed = np.min(allowed, axis=1)
            max_allowed = np.maximum(max_allowed, 1e-6)
            shrink = np.minimum(radii, max_allowed * 0.99999)
            if np.any(shrink < radii - 1e-12):
                radii = shrink
                changed = True
        if not changed:
            break
    # Final safety
    radii = np.minimum(radii, max_b * 0.99999)
    radii = np.maximum(radii, 1e-6)
    return pos, radii

def _initial_best_fit():
    pos = np.zeros((N, 2)); radii = np.zeros(N)
    pos[0] = [0.5, 0.5]; radii[0] = 0.25
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98); y = rng.uniform(0.02, 0.98)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                max_r = min(max_r, d - radii[j])
            if max_r > best_r and max_r > 0.003:
                best_r = max_r; best_x = x; best_y = y; found = True
        if not found or best_r < 0.003:
            best_x, best_y, best_r = 0.5, 0.5, 0.003
        pos[i] = [best_x, best_y]; radii[i] = best_r * 0.97
    return pos, radii

def _initial_hexagonal():
    pos = np.zeros((N, 2)); radii = np.zeros(N)
    r = 0.11; dx = r * np.sqrt(3); dy = r * 1.5; k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset; y = row * dy + r
        for col in range(5):
            if k >= N: break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]; radii[k] = r; k += 1
        if k >= N: break
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        placed = False
        for _ in range(200):
            x = rng.uniform(0.1, 0.9); y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]; radii[k] = min_dist * 0.95; k += 1; placed = True; break
        if not placed:
            pos[k] = [0.5, 0.5]; radii[k] = 0.01; k += 1
    return pos, radii

def _initial_corner_biased(seed=7):
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2)); radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4); t = rng.uniform(0.1, 0.9)
            if edge == 0: x, y = t, 0.08
            elif edge == 1: x, y = t, 0.92
            elif edge == 2: x, y = 0.08, t
            else: x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            max_r = min(max_r, d - radii[j])
        if max_r < 0.01:
            for _ in range(200):
                x2 = rng.uniform(0.05, 0.95); y2 = rng.uniform(0.05, 0.95)
                m2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    d = np.sqrt((x2 - pos[j, 0]) ** 2 + (y2 - pos[j, 1]) ** 2)
                    m2 = min(m2, d - radii[j])
                if m2 > max_r:
                    max_r = m2; x, y = x2, y2
        pos[i] = [x, y]; radii[i] = max(max_r, 0.01) * 0.95
    return pos, radii

def _initial_grid():
    pos = np.zeros((N, 2)); radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N - 1:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]; radii[k] = 0.095; k += 1
    pos[k] = [0.5, 0.02]; radii[k] = 0.01
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    return _post_process(pos, radii)

def _slsqp_hard_constraints(x0, deadline, maxiter=400, ftol=1e-12):
    x = x0.copy()

    def c_boundary(x):
        pos = x[:2 * N].reshape(N, 2); radii = x[2 * N:]
        return np.concatenate([pos[:, 0] - radii, 1.0 - pos[:, 0] - radii,
                               pos[:, 1] - radii, 1.0 - pos[:, 1] - radii])

    def c_overlap(x):
        pos = x[:2 * N].reshape(N, 2); radii = x[2 * N:]
        out = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2 + 1e-16)
                out.append(d - radii[i] - radii[j])
        return np.array(out)

    constraints = [{'type': 'ineq', 'fun': c_boundary}, {'type': 'ineq', 'fun': c_overlap}]
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N + [(0.0, 0.5)] * N

    def neg(x):
        return -np.sum(x[2 * N:])

    try:
        result = minimize(neg, x, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
        if result.success or (not result.success and result.fun < neg(x)):
            x = result.x
    except Exception:
        pass
    return x[:2 * N].reshape(N, 2), np.maximum(x[2 * N:], 1e-4)

def _slsqp_grow_radii(x0, deadline, maxiter=400, ftol=1e-12):
    """Fix positions, grow radii via LP: max sum(r) s.t. r_i + r_j <= d_ij, r_i <= b_i."""
    pos = x0[:2 * N].reshape(N, 2)
    r0 = np.maximum(x0[2 * N:], 1e-4)

    # Boundary upper bounds
    b = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                   np.minimum(pos[:, 1], 1.0 - pos[:, 1]))

    # Pairwise distances
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    # Build LP: max 1^T r s.t. A_ub r <= b_ub, 0 <= r <= b
    # Constraints: r_i + r_j <= dist[i,j] for i<j
    n_pairs = N * (N - 1) // 2
    A = np.zeros((n_pairs, N))
    b_ub = np.zeros(n_pairs)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b_ub[idx] = dist[i, j]
            idx += 1

    bounds_lp = [(1e-6, min(b[i], 0.5)) for i in range(N)]

    try:
        from scipy.optimize import linprog
        res = linprog(-np.ones(N), A_ub=A, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res.success:
            r = np.maximum(res.x, 1e-6)
        else:
            r = r0.copy()
    except Exception:
        # Fallback: greedy shrink
        r = r0.copy()
        r = np.minimum(r, b * 0.9999)
        for _ in range(50):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    if r[i] + r[j] > dist[i, j]:
                        excess = r[i] + r[j] - dist[i, j]
                        if r[i] >= r[j]:
                            r[i] = max(r[i] - excess * 0.51, 1e-6)
                        else:
                            r[j] = max(r[j] - excess * 0.51, 1e-6)
                        changed = True
            if not changed:
                break
        r = np.minimum(r, b * 0.9999)
    return pos, r

def _expand_circles(pos, radii, deadline):
    pos = pos.copy(); radii = radii.copy()
    for _ in range(60):
        if time.time() > deadline - 3: break
        improved = False
        for i in range(N):
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(N):
                if i == j: continue
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                max_r = min(max_r, d - radii[j])
            if max_r > radii[i] * 1.001:
                new_r = min(max_r * 0.9999, radii[i] * 1.05)
                if new_r > radii[i]:
                    old_r = radii[i]; radii[i] = new_r
                    if _is_valid(pos, radii, tol=1e-9):
                        improved = True
                    else:
                        radii[i] = old_r
        if not improved: break
    return pos, radii

def _force_directed(pos, radii, deadline, n_iters=250):
    """Force-directed repulsion + radius growth to find a dense collision-free packing."""
    pos = pos.copy().astype(float); radii = radii.copy().astype(float)
    best_pos = pos.copy(); best_radii = radii.copy(); best_score = np.sum(radii)
    for it in range(n_iters):
        if time.time() > deadline - 5: break
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if i == j: continue
                d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
                max_r = min(max_r, d - radii[j])
            radii[i] = min(max(max_r, 0.001) * 0.9999, radii[i] * 1.02)
        for _ in range(60):
            if time.time() > deadline - 5: break
            any_overlap = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]; dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy + 1e-16)
                    sr = radii[i] + radii[j]
                    if d < sr:
                        any_overlap = True
                        nx, ny = dx / d, dy / d
                        shift = (sr - d) * 0.51
                        pos[i, 0] += nx * shift; pos[i, 1] += ny * shift
                        pos[j, 0] -= nx * shift; pos[j, 1] -= ny * shift
            if not any_overlap: break
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], radii[i] * 0.9999, 1.0 - radii[i] * 0.9999)
            pos[i, 1] = np.clip(pos[i, 1], radii[i] * 0.9999, 1.0 - radii[i] * 0.9999)
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.9999
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score; best_pos = pos_pp.copy(); best_radii = radii_pp.copy()
            pos, radii = pos_pp, radii_pp
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    x = np.concatenate([pos.flatten(), radii])
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.5)] * N
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds, args=(2000.0,),
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 2000.0) - 1e-12:
            x = result.x
    except Exception:
        pass
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds, args=(5000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 5000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    return x[:2 * N].reshape(N, 2), np.maximum(x[2 * N:], 1e-4)

def _final_refinement(pos, radii, deadline):
    x = np.concatenate([pos.flatten(), radii])
    best_x = x.copy(); best_score = np.sum(radii)

    def consider(p, r):
        nonlocal best_x, best_score
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s; best_x = np.concatenate([pp.flatten(), rr])

    # Multi-stage SLSQP with decreasing tolerances
    for ftol in [1e-10, 1e-12, 1e-14]:
        if time.time() > deadline - 4: break
        p, r = _slsqp_hard_constraints(best_x, deadline, maxiter=500, ftol=ftol)
        consider(p, r)
    # LP-based radius growth (optimal for fixed positions)
    if time.time() < deadline - 4:
        p, r = _slsqp_grow_radii(best_x, deadline)
        consider(p, r)
    # Expand circles (greedy radius growth)
    if time.time() < deadline - 4:
        p, r = _expand_circles(best_x[:2 * N].reshape(N, 2), np.maximum(best_x[2 * N:], 1e-4), deadline)
        consider(p, r)
    # High-penalty L-BFGS-B polish
    if time.time() < deadline - 4:
        bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.5)] * N
        try:
            result = minimize(_objective, best_x, method='L-BFGS-B', bounds=bounds, args=(10000.0,),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < _objective(best_x, 10000.0) - 1e-12:
                consider(result.x[:2 * N].reshape(N, 2), np.maximum(result.x[2 * N:], 1e-4))
        except Exception:
            pass
    # Second round: SLSQP + LP growth + expand
    if time.time() < deadline - 3:
        p, r = _slsqp_hard_constraints(best_x, deadline, maxiter=600, ftol=1e-14)
        consider(p, r)
    if time.time() < deadline - 3:
        p, r = _slsqp_grow_radii(best_x, deadline)
        consider(p, r)
    if time.time() < deadline - 3:
        p, r = _expand_circles(best_x[:2 * N].reshape(N, 2), np.maximum(best_x[2 * N:], 1e-4), deadline)
        consider(p, r)
    # Final LP squeeze
    if time.time() < deadline - 2:
        p, r = _slsqp_grow_radii(best_x, deadline)
        consider(p, r)
    return best_x[:2 * N].reshape(N, 2), np.maximum(best_x[2 * N:], 1e-4)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    inits = []
    inits.append(_initial_best_fit())
    inits.append(_initial_hexagonal())
    inits.append(_initial_corner_biased(seed=7))
    inits.append(_initial_grid())
    for s in (100, 101):
        inits.append(_initial_random(s))

    best_pos = inits[0][0].copy(); best_radii = inits[0][1].copy()
    best_score = np.sum(best_radii)

    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.5)] * N
    penalty_stages = [50.0, 500.0, 2000.0]

    def consider(p, r):
        nonlocal best_pos, best_radii, best_score
        pp, rr = _post_process(p, r)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s; best_pos = pp.copy(); best_radii = rr.copy()

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25: break
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()

        for p_w in penalty_stages:
            if time.time() > deadline - 12: break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds, args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            consider(current_x[:2 * N].reshape(N, 2), np.maximum(current_x[2 * N:], 1e-4))

        if time.time() < deadline - 12:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 6: break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0), options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                consider(current_x[:2 * N].reshape(N, 2), np.maximum(current_x[2 * N:], 1e-4))

        if time.time() < deadline - 12:
            p, r = _slsqp_hard_constraints(current_x, deadline)
            consider(p, r)
            current_x = np.concatenate([p.flatten(), r])

        if time.time() < deadline - 10:
            p, r = _slsqp_grow_radii(current_x, deadline)
            consider(p, r)
            current_x = np.concatenate([p.flatten(), r])

        if time.time() < deadline - 10:
            p, r = _force_directed(current_x[:2 * N].reshape(N, 2), np.maximum(current_x[2 * N:], 1e-4), deadline)
            consider(p, r)
            current_x = np.concatenate([p.flatten(), r])

    if time.time() < deadline - 12:
        p, r = _polish(best_pos, best_radii, deadline)
        consider(p, r)
    if time.time() < deadline - 12:
        p, r = _slsqp_hard_constraints(np.concatenate([best_pos.flatten(), best_radii]), deadline, maxiter=400, ftol=1e-12)
        consider(p, r)
    if time.time() < deadline - 10:
        p, r = _slsqp_grow_radii(np.concatenate([best_pos.flatten(), best_radii]), deadline)
        consider(p, r)
    if time.time() < deadline - 10:
        p, r = _expand_circles(best_pos, best_radii, deadline)
        consider(p, r)
    if time.time() < deadline - 10:
        p, r = _force_directed(best_pos, best_radii, deadline, n_iters=200)
        consider(p, r)
    if time.time() < deadline - 8:
        p, r = _final_refinement(best_pos, best_radii, deadline)
        consider(p, r)

    best_pos, best_radii = _post_process(best_pos, best_radii)

    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2)); radii = np.zeros(N); k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]; radii[k] = 0.095; k += 1
        pos[k] = [0.5, 0.02]; radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
