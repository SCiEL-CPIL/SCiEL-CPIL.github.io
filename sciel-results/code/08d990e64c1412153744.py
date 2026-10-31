"""Circle packing in perimeter-4 rectangle: vectorized local search + SLSQP + tight polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 148.0


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    iu = np.triu_indices(N, 1)
    if np.any(d2[iu] < rs[iu] ** 2 - tol):
        return False
    return True


def _max_radius_i(cx, cy, r, W, H, i):
    """Max radius for circle i given others and walls fixed."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    others = np.arange(N) != i
    dx = cx[i] - cx[others]
    dy = cy[i] - cy[others]
    d = np.sqrt(dx * dx + dy * dy)
    r_allowed = d - r[others]
    r_max = min(r_wall, np.min(r_allowed))
    return max(0.0, r_max)


def _is_valid_move(cx, cy, r, W, H, i, new_x, new_y, new_r):
    if new_x < new_r - 1e-12 or new_x > W - new_r + 1e-12:
        return False
    if new_y < new_r - 1e-12 or new_y > H - new_r + 1e-12:
        return False
    others = np.arange(N) != i
    dx = new_x - cx[others]
    dy = new_y - cy[others]
    d2 = dx * dx + dy * dy
    rs = new_r + r[others]
    if np.any(d2 < rs * rs - 1e-12):
        return False
    return True


def _force_relax(cx, cy, r, W, H, max_iter=3000, time_budget=None):
    n = N
    dt = 0.004
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall forces
        for i in range(n):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 80
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 80
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 80
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 80
        # Pairwise repulsion (vectorized)
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e10)
        rs = r[:, None] + r[None, :]
        overlap = np.maximum(0, rs - d)
        ux = np.where(d > 1e-10, dx / d, 0.0)
        uy = np.where(d > 1e-10, dy / d, 0.0)
        f = overlap * 25.0
        fx += np.sum(f * ux, axis=1)
        fy += np.sum(f * uy, axis=1)
        if np.max(overlap) < 1e-10:
            break
        cx = np.clip(cx + dt * fx, r, W - r)
        cy = np.clip(cy + dt * fy, r, H - r)
    return cx, cy


def _expand_radii(cx, cy, r, W, H, time_budget, max_rounds=30, bisection=30):
    n = N
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _max_radius_i(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-9:
                lo, hi = r[i], r_max
                for _ in range(bisection):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    return cx, cy, r


def _local_search(cx, cy, r, W, H, time_budget):
    """Vectorized 8-direction position search with radius expansion."""
    n = N
    best_val = np.sum(r)
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    step_sizes = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                           [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    directions /= np.sqrt(2)
    # Normalize cardinal to unit
    directions[:4] /= 1.0

    for step in step_sizes:
        if time.time() > time_budget:
            break
        improved = True
        max_rounds = 15
        for rnd in range(max_rounds):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                r_max = _max_radius_i(cx, cy, r, W, H, i)
                # Try to expand radius
                if r_max > r[i] + 1e-9:
                    lo, hi = r[i], r_max
                    for _ in range(25):
                        mid = (lo + hi) / 2.0
                        r_trial = r.copy()
                        r_trial[i] = mid
                        if _is_valid(cx, cy, r_trial, W, H):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-10:
                        r[i] = lo
                        improved = True
                        if np.sum(r) > best_val:
                            best_val = np.sum(r)
                            best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
                        continue

                # Try position moves
                found = False
                for dx_dir, dy_dir in directions:
                    for s in [step, step * 0.5, step * 0.25]:
                        new_x = cx[i] + dx_dir * s
                        new_y = cy[i] + dy_dir * s
                        if new_x < r[i] or new_x > W - r[i] or new_y < r[i] or new_y > H - r[i]:
                            continue
                        others = np.arange(n) != i
                        dx = new_x - cx[others]
                        dy = new_y - cy[others]
                        d2 = dx * dx + dy * dy
                        rs = r[i] + r[others]
                        if np.any(d2 < rs * rs - 1e-12):
                            continue
                        # Check if this position allows larger radius
                        old_x, old_y = cx[i], cy[i]
                        cx[i], cy[i] = new_x, new_y
                        r_max_new = _max_radius_i(cx, cy, r, W, H, i)
                        cx[i], cy[i] = old_x, old_y
                        if r_max_new > r[i] + 1e-8:
                            cx[i], cy[i] = new_x, new_y
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Try small random moves
                    for _ in range(3):
                        if time.time() > time_budget:
                            break
                        ang = np.random.uniform(0, 2 * np.pi)
                        s = step * np.random.uniform(0.3, 1.0)
                        new_x = cx[i] + np.cos(ang) * s
                        new_y = cy[i] + np.sin(ang) * s
                        if new_x < r[i] or new_x > W - r[i] or new_y < r[i] or new_y > H - r[i]:
                            continue
                        others = np.arange(n) != i
                        dx = new_x - cx[others]
                        dy = new_y - cy[others]
                        d2 = dx * dx + dy * dy
                        rs = r[i] + r[others]
                        if np.any(d2 < rs * rs - 1e-12):
                            continue
                        old_x, old_y = cx[i], cy[i]
                        cx[i], cy[i] = new_x, new_y
                        r_max_new = _max_radius_i(cx, cy, r, W, H, i)
                        cx[i], cy[i] = old_x, old_y
                        if r_max_new > r[i] + 1e-8:
                            cx[i], cy[i] = new_x, new_y
                            improved = True
                            break
            if not improved:
                break

    if np.sum(r) > best_val:
        best_val = np.sum(r)
        best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    return best_cx, best_cy, best_r


def _slsqp_optimize(cx, cy, r, W, H, time_budget, maxiter=400):
    n = N
    z0 = np.concatenate([cx, cy, r, [W]])
    h0 = H

    def objective(zv):
        return -np.sum(zv[2*n:3*n])

    constraints = []
    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*n+i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*n+i+1]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3*i] - p[3*j])**2 +
                    (p[3*i+1] - p[3*j+1])**2 -
                    (p[3*n+i] + p[3*n+j])**2
                )
            })

    try:
        res = minimize(objective, z0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 80})
        z_opt = res.x
        cx_new = z_opt[0:n]
        cy_new = z_opt[n:2*n]
        r_new = z_opt[2*n:3*n]
        W_new = z_opt[-1]
        H_new = 2.0 - W_new
        if _is_valid(cx_new, cy_new, r_new, W_new, H_new):
            return cx_new, cy_new, r_new, W_new, H_new
    except Exception:
        pass
    return None


def _tight_polish(cx, cy, r, W, H, time_budget):
    """Very tight final polish with many small steps."""
    n = N
    best_val = np.sum(r)
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()

    # Multiple passes with decreasing step sizes
    step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    for step in step_sizes:
        if time.time() > time_budget:
            break
        improved = True
        max_rounds = 10
        for rnd in range(max_rounds):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                r_max = _max_radius_i(cx, cy, r, W, H, i)
                if r_max > r[i] + 1e-10:
                    lo, hi = r[i], r_max
                    for _ in range(35):
                        mid = (lo + hi) / 2.0
                        r_trial = r.copy()
                        r_trial[i] = mid
                        if _is_valid(cx, cy, r_trial, W, H):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-11:
                        r[i] = lo
                        improved = True
                        if np.sum(r) > best_val:
                            best_val = np.sum(r)
                            best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
                        continue

                # Try 16 directions
                for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    for s in [step, step * 0.5]:
                        new_x = cx[i] + np.cos(ang) * s
                        new_y = cy[i] + np.sin(ang) * s
                        if new_x < r[i] or new_x > W - r[i] or new_y < r[i] or new_y > H - r[i]:
                            continue
                        others = np.arange(n) != i
                        dx = new_x - cx[others]
                        dy = new_y - cy[others]
                        d2 = dx * dx + dy * dy
                        rs = r[i] + r[others]
                        if np.any(d2 < rs * rs - 1e-12):
                            continue
                        old_x, old_y = cx[i], cy[i]
                        cx[i], cy[i] = new_x, new_y
                        r_max_new = _max_radius_i(cx, cy, r, W, H, i)
                        cx[i], cy[i] = old_x, old_y
                        if r_max_new > r[i] + 1e-9:
                            cx[i], cy[i] = new_x, new_y
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    # Final uniform expansion
    for _ in range(1000):
        if time.time() > time_budget:
            break
        r_trial = r * 1.00005
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break

    if np.sum(r) > best_val:
        best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    return best_cx, best_cy, best_r


def _expansion_polish(cx, cy, r, W, H, time_budget):
    """Shift circles toward their tightest constraint to create space for expansion."""
    n = N
    for rnd in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _max_radius_i(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-8:
                lo, hi = r[i], r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True
                continue

            # Find tightest constraint and move away
            min_gap = 1e9
            move_dx, move_dy = 0.0, 0.0
            gaps = [
                (cx[i] - r[i], 1, 0),
                (W - cx[i] - r[i], -1, 0),
                (cy[i] - r[i], 0, 1),
                (H - cy[i] - r[i], 0, -1),
            ]
            for gap, dx_dir, dy_dir in gaps:
                if gap < min_gap:
                    min_gap = gap
                    move_dx, move_dy = dx_dir, dy_dir
            for j in range(n):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-10:
                        move_dx = dx / d
                        move_dy = dy / d

            if min_gap < 0.05:
                for step in [0.03, 0.015, 0.008, 0.004, 0.002]:
                    new_x = cx[i] + move_dx * step
                    new_y = cy[i] + move_dy * step
                    if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y, r[i]):
                        cx[i] = new_x
                        cy[i] = new_y
                        improved = True
                        break
                else:
                    # Try perpendicular
                    perp_dx, perp_dy = -move_dy, move_dx
                    for step in [0.02, 0.01, 0.005]:
                        for s in [1.0, -1.0]:
                            new_x = cx[i] + s * perp_dx * step
                            new_y = cy[i] + s * perp_dy * step
                            if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y, r[i]):
                                cx[i] = new_x
                                cy[i] = new_y
                                improved = True
                                break
                        if improved:
                            break
        if not improved:
            break

    # Final uniform expansion
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0002
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break
    return cx, cy, r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_value = 0.0
    best_cx = best_cy = best_r = None
    best_W = best_H = None

    def _update_best(cx, cy, r, W, H):
        nonlocal best_value, best_cx, best_cy, best_r, best_W, best_H
        if _is_valid(cx, cy, r, W, H):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
                best_W, best_H = W, H

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.95, 1.05]
    side = 5

    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h0 = 2.0 - w0
        if h0 < 0.25 or w0 < 0.25:
            continue

        cx_t = np.zeros(N)
        cy_t = np.zeros(N)
        r_t = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            cx_t[i] = (col + 0.5) / side * w0
            cy_t[i] = (row + 0.5) / side * h0
            r_t[i] = min(w0, h0) / side * 0.85

        cx_t, cy_t = _force_relax(cx_t, cy_t, r_t, w0, h0, max_iter=500, time_budget=deadline - 25)
        cx_t, cy_t, r_t = _expand_radii(cx_t, cy_t, r_t, w0, h0, deadline - 25)
        _update_best(cx_t, cy_t, r_t, w0, h0)

        # Local search
        if time.time() < deadline - 20:
            cx_t, cy_t, r_t = _local_search(cx_t, cy_t, r_t, w0, h0, deadline - 15)
            _update_best(cx_t, cy_t, r_t, w0, h0)

        # SLSQP
        if time.time() < deadline - 20:
            result = _slsqp_optimize(cx_t, cy_t, r_t, w0, h0, deadline - 15, maxiter=250)
            if result is not None:
                cx_n, cy_n, r_n, W_n, H_n = result
                cx_n, cy_n, r_n = _expand_radii(cx_n, cy_n, r_n, W_n, H_n, deadline - 10)
                _update_best(cx_n, cy_n, r_n, W_n, H_n)

    # Polish the best solution
    if best_cx is not None and time.time() < deadline - 15:
        cx_p, cy_p, r_p = _local_search(best_cx, best_cy, best_r, best_W, best_H, deadline - 10)
        _update_best(cx_p, cy_p, r_p, best_W, best_H)

    if best_cx is not None and time.time() < deadline - 10:
        cx_p, cy_p, r_p = _expansion_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 8)
        _update_best(cx_p, cy_p, r_p, best_W, best_H)

    if best_cx is not None and time.time() < deadline - 8:
        cx_p, cy_p, r_p = _tight_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 3)
        _update_best(cx_p, cy_p, r_p, best_W, best_H)

    # SLSQP on the best solution
    if best_cx is not None and time.time() < deadline - 12:
        result = _slsqp_optimize(best_cx, best_cy, best_r, best_W, best_H, deadline - 8, maxiter=300)
        if result is not None:
            cx_n, cy_n, r_n, W_n, H_n = result
            cx_n, cy_n, r_n = _expand_radii(cx_n, cy_n, r_n, W_n, H_n, deadline - 5)
            _update_best(cx_n, cy_n, r_n, W_n, H_n)

    # Final tight polish
    if best_cx is not None and time.time() < deadline - 5:
        cx_p, cy_p, r_p = _tight_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 1)
        _update_best(cx_p, cy_p, r_p, best_W, best_H)

    # Safety: shrink radii slightly
    if best_cx is not None:
        for _ in range(10):
            r_trial = best_r * (1 - 1e-9)
            if _is_valid(best_cx, best_cy, r_trial, best_W, best_H):
                best_r = r_trial
            else:
                break

    if best_cx is None:
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, 5)
            cx[i] = (col + 0.5) / 5
            cy[i] = (row + 0.5) / 5
            r[i] = 0.09
        return np.column_stack((cx, cy, r))

    return np.column_stack((best_cx, best_cy, best_r))
