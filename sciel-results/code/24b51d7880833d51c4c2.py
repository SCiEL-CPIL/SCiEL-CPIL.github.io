"""Circle packing in perimeter-4 rectangle: Smooth SLSQP + DE + systematic polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _smooth_objective(z, penalty_weight=5000.0, beta=200.0):
    """Smooth penalty-based objective using soft-ReLU for differentiability."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty with smooth soft-ReLU
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap_raw = r_sum_pairs - d_pairs

    soft_relu = np.logaddexp(0, beta * overlap_raw) / beta
    obj += penalty_weight * np.sum(soft_relu)

    # Boundary penalty with smooth soft-ReLU
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    soft_left = np.logaddexp(0, beta * left_viol) / beta
    soft_right = np.logaddexp(0, beta * right_viol) / beta
    soft_bottom = np.logaddexp(0, beta * bottom_viol) / beta
    soft_top = np.logaddexp(0, beta * top_viol) / beta

    obj += penalty_weight * (
        np.sum(soft_left) +
        np.sum(soft_right) +
        np.sum(soft_bottom) +
        np.sum(soft_top)
    )

    # Aspect ratio constraint (smooth)
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _force_relax(z, max_iter=300, time_limit=None):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    start = time.time()
    iu = np.triu_indices(n, k=1)

    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break

        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-10:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < 1e-9:
            break

        # Adaptive step
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(0.001, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        z[0::3][:n] = x
        z[1::3][:n] = y

    return z


def _expand_radii(x, y, r, w, h, time_limit=None):
    """Iteratively expand each radius to its maximum valid value."""
    n = N
    start = time.time()
    improved = True

    while improved:
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False

        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed

            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-8:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

    return r


def _polish_positions(x, y, r, w, h, time_limit=None):
    """Polish positions using force-directed method with adaptive step."""
    n = N
    start = time.time()
    dt = 0.002
    iu = np.triu_indices(n, k=1)

    for it in range(3000):
        if time_limit is not None and time.time() > time_limit:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.001
        elif max_force > 0.1:
            dt = 0.002
        else:
            dt = 0.003

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _uniform_expand(x, y, r, w, h, time_limit=None):
    """Try to uniformly scale all radii up via binary search."""
    start = time.time()
    for _ in range(40):
        if time_limit is not None and time.time() > time_limit:
            break
        lo, hi = 1.0, 1.015
        for _ in range(18):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break
    return r


def _constraint_refine(z, time_budget):
    """High-precision SLSQP with explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    maxiter = int(min(1000, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        val = float(np.sum(r))
        return True, val

    def try_update(z):
        nonlocal best_value, best_result
        valid, val = check_validity(z)
        if valid and val > best_value:
            best_value = val
            best_result = z.copy()

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 75:
        de_bounds = [(0, 2)] * (3 * N) + [(0.5, 1.5)]
        de_obj = lambda zv: _smooth_objective(zv, 5000.0, 200.0)
        try:
            de_res = differential_evolution(
                de_obj, de_bounds, seed=42, maxiter=40,
                tol=1e-8, popsize=12, mutation=(0.3, 0.8),
                recombination=0.7, init='latinhypercube'
            )
            if de_res.success or de_res.fun < 0:
                z_de = de_res.x
                valid, val = check_validity(z_de)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_de.copy()
        except Exception:
            pass

    # Phase 2: Local optimization from multiple starts with smooth objective
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        z0_relaxed = _force_relax(z0, max_iter=200, time_limit=deadline - 35)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(80, remaining * 30)))
        try:
            res = minimize(_smooth_objective, z0_relaxed, method='SLSQP',
                           args=(5000.0, 200.0),
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            try_update(res.x)
        except Exception:
            continue

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Phase 3: Systematic polish pipeline
    if best_result is not None and time.time() < deadline - 20:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Step 1: Polish positions
        if time.time() < deadline - 15:
            x, y = _polish_positions(x, y, r, w, h, time_limit=deadline - 15)

        # Step 2: Expand radii individually
        if time.time() < deadline - 10:
            r = _expand_radii(x, y, r, w, h, time_limit=deadline - 10)

        # Step 3: Uniform expansion
        if time.time() < deadline - 8:
            r = _uniform_expand(x, y, r, w, h, time_limit=deadline - 8)

        # Reconstruct and check
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w
        try_update(z_new)

        # Step 4: Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 5:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 3)
            if refined is not None:
                try_update(refined)

        # Step 5: Final position polish + expand
        if best_result is not None and time.time() < deadline - 2:
            x2 = best_result[0::3][:N].copy()
            y2 = best_result[1::3][:N].copy()
            r2 = best_result[2::3][:N].copy()
            w2 = best_result[-1]
            h2 = 2.0 - w2
            x2, y2 = _polish_positions(x2, y2, r2, w2, h2, time_limit=deadline - 2)
            r2 = _expand_radii(x2, y2, r2, w2, h2, time_limit=deadline - 2)
            r2 = _uniform_expand(x2, y2, r2, w2, h2, time_limit=deadline - 2)
            z_final = np.empty(3 * N + 1)
            z_final[0::3][:N] = x2
            z_final[1::3][:N] = y2
            z_final[2::3][:N] = r2
            z_final[-1] = w2
            try_update(z_final)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output with safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
