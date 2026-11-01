"""Circle packing in perimeter-4 rectangle: Multi-start SLSQP + LSE Refine + Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21


def _lse_refine(z, time_budget, n_iter=2000):
    """
    Log-sum-exp based soft refinement: jointly adjust positions and grow radii
    using a smooth surrogate for the max-overlap constraint.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Precompute pair indices
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])

    def _smooth_violation(x, y, r, w, h):
        """Compute smooth violation using LSE for pairwise and boundary constraints."""
        # Pairwise
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        # violation = r_sum - d (positive if overlapping)
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        viol_pair = r_sum - d

        # Boundary
        viol_left = r - x
        viol_right = x + r - w
        viol_bottom = r - y
        viol_top = y + r - h

        # LSE approximation of max(violations)
        beta = 50.0
        all_viol = np.concatenate([viol_pair, viol_left, viol_right, viol_bottom, viol_top])
        # Clip for numerical stability
        all_viol = np.clip(all_viol, -5.0, 5.0)
        lse = (1.0 / beta) * np.log(np.sum(np.exp(beta * all_viol)))
        return lse

    def _grad_violation(x, y, r, w, h):
        """Compute gradient of smooth violation w.r.t. x, y, r."""
        beta = 50.0
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        viol_pair = np.clip(r_sum - d, -5.0, 5.0)

        viol_left = np.clip(r - x, -5.0, 5.0)
        viol_right = np.clip(x + r - w, -5.0, 5.0)
        viol_bottom = np.clip(r - y, -5.0, 5.0)
        viol_top = np.clip(y + r - h, -5.0, 5.0)

        all_viol = np.concatenate([viol_pair, viol_left, viol_right, viol_bottom, viol_top])
        weights = np.exp(beta * all_viol) / np.sum(np.exp(beta * all_viol))

        # Gradient w.r.t. x
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        # Pairwise contributions
        for idx in range(n_pairs):
            wt = weights[idx] * beta
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            if d_sq[idx] > 1e-12:
                gx[i_idx] += wt * (dx[idx] / d[idx])
                gx[j_idx] -= wt * (dx[idx] / d[idx])
                gy[i_idx] += wt * (dy[idx] / d[idx])
                gy[j_idx] -= wt * (dy[idx] / d[idx])
            gr[i_idx] += wt
            gr[j_idx] += wt

        # Boundary contributions
        gr += beta * weights[n_pairs:n_pairs + n]  # left
        gx -= beta * weights[n_pairs:n_pairs + n]
        gr += beta * weights[n_pairs + n:n_pairs + 2 * n]  # right
        gx += beta * weights[n_pairs + n:n_pairs + 2 * n]
        gr += beta * weights[n_pairs + 2 * n:n_pairs + 3 * n]  # bottom
        gy -= beta * weights[n_pairs + 2 * n:n_pairs + 3 * n]
        gr += beta * weights[n_pairs + 3 * n:n_pairs + 4 * n]  # top
        gy += beta * weights[n_pairs + 3 * n:n_pairs + 4 * n]

        return gx, gy, gr

    # Iterative refinement
    lr_pos = 0.001
    lr_rad = 0.0001

    for it in range(n_iter):
        if time.time() > time_budget:
            break

        gx, gy, gr = _grad_violation(x, y, r, w, h)

        # Move positions to reduce violation
        x -= lr_pos * gx
        y -= lr_pos * gy

        # Grow radii (opposite direction of gradient)
        r += lr_rad * gr

        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        r = np.maximum(r, 0.001)

        # Periodic validity check
        if it % 100 == 0:
            if not _is_valid(x, y, r, w, h):
                # Revert radii slightly
                r *= 0.999
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)

    # Final uniform expansion
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0002
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _coordinate_descent_polish(z, time_budget):
    """Coordinate descent: optimize each circle's position and radius independently."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Optimize position of circle i (fix others)
            best_x, best_y, best_r = x[i], y[i], r[i]

            # Try grid of positions
            for dx in [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]:
                for dy in [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]:
                    nx = x[i] + dx
                    ny = y[i] + dy
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    # Check validity
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Max radius from this position
                    r_max = min(nx, w - nx, ny, h - ny)
                    for j in range(n):
                        if i == j:
                            continue
                        d = np.sqrt((nx - x[j]) ** 2 + (ny - y[j]) ** 2)
                        r_max = min(r_max, d - r[j])
                    r_max = max(0.0, r_max)
                    if r_max > best_r + 1e-6:
                        best_x, best_y, best_r = nx, ny, r_max

            if best_r > r[i] + 1e-6:
                x[i] = best_x
                y[i] = best_y
                r[i] = best_r
                improved = True

        if not improved:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


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


def _force_relax(z, max_iter=300):
    """Simple force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        fx -= np.maximum(r - x, 0) * 50
        fx += np.maximum(x + r - w, 0) * 50
        fy -= np.maximum(r - y, 0) * 50
        fy += np.maximum(y + r - h, 0) * 50

        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-10
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    f = overlap * 10
                    fx[i] += dx / d * f
                    fy[i] += dy / d * f
                    fx[j] -= dx / d * f
                    fy[j] -= dy / d * f

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _vectorized_objective(z, penalty_weight=500.0):
    """Penalty-based objective for SLSQP."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    r_sum = r[iu[0]] + r[iu[1]]
    overlap = np.maximum(r_sum - d, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary
    obj += penalty_weight * (np.sum(np.maximum(r - x, 0)**2) +
                             np.sum(np.maximum(x + r - w, 0)**2) +
                             np.sum(np.maximum(r - y, 0)**2) +
                             np.sum(np.maximum(y + r - h, 0)**2))

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _constraint_refine(z, time_budget):
    """SLSQP with explicit constraints for high-precision refinement."""
    n = N

    def obj(zv):
        return -np.sum(zv[2::3][:n])

    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def mk(i=i, j=j):
                return lambda p: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            constraints.append({'type': 'ineq', 'fun': mk()})

    maxiter = int(min(500, max(100, time_budget * 80)))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})
        if _is_valid(res.x[0::3][:n], res.x[1::3][:n], res.x[2::3][:n], res.x[-1], 2.0 - res.x[-1]):
            return res.x
    except Exception:
        pass
    return None


def _expand_radii(x, y, r, w, h, time_budget):
    """Try to expand all radii uniformly, then individually."""
    # Uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Individual expansion with position adjustment
    for _ in range(10):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            # Try to grow circle i
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                r_max = min(r_max, d - r[j])
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-6:
                r[i] = r_max
                improved = True
        if not improved:
            break

    return r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    def check_and_update(z):
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
                return True
        return False

    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    guesses = []

    # Grid-based starts with various aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
    side = int(np.ceil(np.sqrt(N)))
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

    # Random starts
    for _ in range(8):
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

    # SLSQP optimization from each guess
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        z0_relaxed = _force_relax(z0, max_iter=200)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(_vectorized_objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            check_and_update(res.x)
        except Exception:
            continue

    # Polish phases
    if best_result is not None and time.time() < deadline - 20:
        # LSE refinement
        refined = _lse_refine(best_result, deadline - (time.time() - start_time) - 10)
        if refined is not None:
            check_and_update(refined)

        # Coordinate descent
        if time.time() < deadline - 15:
            cd = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 5)
            if cd is not None:
                check_and_update(cd)

        # Constraint refinement
        if time.time() < deadline - 10:
            cr = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if cr is not None:
                check_and_update(cr)

        # Final expansion
        if best_result is not None and time.time() < deadline - 5:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            r_expanded = _expand_radii(x, y, r, w, h, deadline - (time.time() - start_time))
            if _is_valid(x, y, r_expanded, w, h):
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:N] = x
                z_new[1::3][:N] = y
                z_new[2::3][:N] = r_expanded
                z_new[-1] = w
                check_and_update(z_new)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
