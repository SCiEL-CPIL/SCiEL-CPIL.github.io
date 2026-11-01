"""Circle packing in perimeter-4 rectangle: Differential Evolution + Newton Radius Max."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21


def _lse_penalty(x, y, r, w, h, beta=50.0):
    """Smooth log-sum-exp overlap penalty (differentiable, multi-modal friendly)."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    # Upper triangle only
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

    # Log-sum-exp for smoothness
    if np.max(overlap) < 1e-12:
        return 0.0
    max_ov = np.max(overlap)
    lse = np.log(np.sum(np.exp(beta * (overlap - max_ov)))) / beta + max_ov
    return lse


def _de_objective(z, beta=50.0):
    """Objective for differential evolution: minimize -sum(r) + penalty."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15 or np.any(r < 0.001):
        return 1e6

    # Overlap penalty
    overlap_pen = 50.0 * _lse_penalty(x, y, r, w, h, beta)

    # Boundary penalty (smooth)
    left_v = np.maximum(r - x, 0.0)
    right_v = np.maximum(x + r - w, 0.0)
    bot_v = np.maximum(r - y, 0.0)
    top_v = np.maximum(y + r - h, 0.0)
    bound_pen = 50.0 * (np.sum(left_v * left_v) + np.sum(right_v * right_v) +
                        np.sum(bot_v * bot_v) + np.sum(top_v * top_v))

    return -np.sum(r) + overlap_pen + bound_pen


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
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


def _max_radius_given_others(x, y, r, w, h, i):
    """Max radius for circle i given all other positions fixed."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _newton_radius_max(x, y, r, w, h, time_budget):
    """
    Newton-style radius maximization:
    1. Per-circle binary search for max radius given others.
    2. Global uniform scaling via binary search.
    3. Iterate a few times for convergence.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for iteration in range(3):
        if time.time() > time_budget:
            break

        # Per-circle max radius
        for i in range(N):
            r_max = _max_radius_given_others(x, y, r, w, h, i)
            if r_max > r[i] + 1e-10:
                # Binary search for valid max
                lo, hi = r[i], r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r[i] = mid
                    if _is_valid(x, y, r, w, h):
                        lo = mid
                    else:
                        hi = mid
                r[i] = lo

        # Global uniform scaling
        for _ in range(50):
            if time.time() > time_budget:
                break
            r_trial = r * 1.002
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
            else:
                break

    return x, y, r


def _slsqp_polish(z, time_budget):
    """Quick SLSQP polish using penalty objective."""
    def obj(zv):
        x = zv[0::3][:N]
        y = zv[1::3][:N]
        r = zv[2::3][:N]
        w = zv[-1]
        h = 2.0 - w
        if w < 0.15 or h < 0.15:
            return 1e6
        return _de_objective(zv, beta=100.0)

    bounds = [(0.001, 2.0)] * (3 * N) + [(0.2, 1.8)]
    maxiter = int(min(400, max(50, time_budget * 40)))

    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 60})
        z_opt = res.x
        x = z_opt[0::3][:N]
        y = z_opt[1::3][:N]
        r = z_opt[2::3][:N]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    best_value = 0.0
    best_result = None

    def _evaluate(z):
        nonlocal best_value, best_result
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = z.copy()

    # --- Phase 1: Differential Evolution global search ---
    # Bounds: x in [0.01, 1.99], y in [0.01, 1.99], r in [0.01, 0.5], w in [0.3, 1.7]
    bounds = [(0.01, 1.99)] * N + [(0.01, 1.99)] * N + [(0.01, 0.5)] * N + [(0.3, 1.7)]
    # Reorder: x[0..N-1], y[0..N-1], r[0..N-1], w
    bounds_de = []
    for i in range(N):
        bounds_de.append((0.01, 1.99))  # x_i
    for i in range(N):
        bounds_de.append((0.01, 1.99))  # y_i
    for i in range(N):
        bounds_de.append((0.01, 0.5))   # r_i
    bounds_de.append((0.3, 1.7))        # w

    # DE objective expects flat array: [x0..x20, y0..y20, r0..r20, w]
    # But our _de_objective expects interleaved [x0,y0,r0, x1,y1,r1, ..., w]
    # Let's make a wrapper
    def de_obj_flat(z_flat):
        # z_flat = [x0..x20, y0..y20, r0..r20, w]
        x = z_flat[0:N]
        y = z_flat[N:2*N]
        r = z_flat[2*N:3*N]
        w = z_flat[3*N]
        h = 2.0 - w
        if w < 0.15 or h < 0.15 or np.any(r < 0.001):
            return 1e6
        overlap_pen = 50.0 * _lse_penalty(x, y, r, w, h, beta=50.0)
        left_v = np.maximum(r - x, 0.0)
        right_v = np.maximum(x + r - w, 0.0)
        bot_v = np.maximum(r - y, 0.0)
        top_v = np.maximum(y + r - h, 0.0)
        bound_pen = 50.0 * (np.sum(left_v * left_v) + np.sum(right_v * right_v) +
                            np.sum(bot_v * bot_v) + np.sum(top_v * top_v))
        return -np.sum(r) + overlap_pen + bound_pen

    remaining = deadline - (time.time() - start_time)
    maxiter_de = int(min(300, max(80, remaining * 3)))

    try:
        de_res = differential_evolution(
            de_obj_flat, bounds_de,
            maxiter=maxiter_de, popsize=15, tol=1e-10,
            mutation=(0.3, 1.0), recombination=0.8,
            seed=42, polish=False, init='sobol'
        )
        z_flat = de_res.x
        x = z_flat[0:N]
        y = z_flat[N:2*N]
        r = z_flat[2*N:3*N]
        w = z_flat[3*N]

        # Convert to interleaved format
        z = np.zeros(3 * N + 1)
        for i in range(N):
            z[3*i] = x[i]
            z[3*i+1] = y[i]
            z[3*i+2] = r[i]
        z[-1] = w

        _evaluate(z)
    except Exception:
        pass

    # --- Phase 1b: Multi-start with perturbation around DE result ---
    if best_result is not None and time.time() < deadline - 30:
        base = best_result.copy()
        rng = np.random.default_rng(123)
        for trial in range(5):
            if time.time() > deadline - 25:
                break
            z_pert = base.copy()
            # Small random perturbation
            for i in range(N):
                z_pert[3*i] += rng.uniform(-0.02, 0.02)
                z_pert[3*i+1] += rng.uniform(-0.02, 0.02)
                z_pert[3*i+2] *= rng.uniform(0.9, 1.1)
            z_pert[-1] = base[-1] + rng.uniform(-0.05, 0.05)
            z_pert[-1] = np.clip(z_pert[-1], 0.3, 1.7)

            # Quick SLSQP polish
            polished = _slsqp_polish(z_pert, deadline - (time.time() - start_time) - 15)
            if polished is not None:
                _evaluate(polished)

    # --- Phase 2: Newton radius maximization on best ---
    if best_result is not None and time.time() < deadline - 15:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        x_new, y_new, r_new = _newton_radius_max(x, y, r, w, h, deadline - (time.time() - start_time) - 5)

        z_new = np.zeros(3 * N + 1)
        for i in range(N):
            z_new[3*i] = x_new[i]
            z_new[3*i+1] = y_new[i]
            z_new[3*i+2] = r_new[i]
        z_new[-1] = w

        _evaluate(z_new)

    # --- Phase 3: Final SLSQP + Newton combo ---
    if best_result is not None and time.time() < deadline - 8:
        polished = _slsqp_polish(best_result, deadline - (time.time() - start_time) - 5)
        if polished is not None:
            _evaluate(polished)

        if best_result is not None and time.time() < deadline - 4:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x_new, y_new, r_new = _newton_radius_max(x, y, r, w, h, deadline - (time.time() - start_time))
            z_new = np.zeros(3 * N + 1)
            for i in range(N):
                z_new[3*i] = x_new[i]
                z_new[3*i+1] = y_new[i]
                z_new[3*i+2] = r_new[i]
            z_new[-1] = w
            _evaluate(z_new)

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
