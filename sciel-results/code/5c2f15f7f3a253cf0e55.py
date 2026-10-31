"""Circle packing in perimeter-4 rectangle.

Combines:
  * SLSQP global optimization over (x, y, r, w) with smooth quadratic penalties
  * Smooth gap-aware force polish (creates space for radius growth)
  * NEW: Sequential radial growth via a smooth "pressure" field
      - each circle's local clearance is approximated with a log-sum-exp
        soft-min over neighbor/wall distances
      - circles are grown iteratively by their admissible radius increase,
        most-constrained first, interleaved with force relaxation
  * Contact-graph polish (encourages exact tangencies)
  * Constraint-based SLSQP refinement
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


# ---------------------------------------------------------------------------
# Objective (penalty form) for SLSQP
# ---------------------------------------------------------------------------
def _vectorized_objective(z, penalty_weight=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]

    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------
def _is_valid(cx, cy, r, W, H, tol=1e-8):
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
            return False
    return True


def _max_radius_at(x, y, r, w, h, i):
    """Max admissible radius for circle i given all others fixed in place."""
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


# ---------------------------------------------------------------------------
# Force-directed relaxation (unified parameter space)
# ---------------------------------------------------------------------------
def _force_relax_unified(z, max_iter=500):
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
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

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


# ---------------------------------------------------------------------------
# Smooth gap-aware polish (creates space, then grows radii)
# ---------------------------------------------------------------------------
def _polish_result_smooth(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0008
    max_iter = int(min(4000, time_budget * 180))
    gap_smooth_len = 0.04

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 45
        fx += right_viol * 45
        fy -= bottom_viol * 45
        fy += top_viol * 45

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        r_sum_pairs = r_sum[iu]
        gap = d_pairs - r_sum_pairs

        overlap_force = np.maximum(-gap, 0.0) * 22.0
        soft_force = 0.6 * np.exp(-gap / gap_smooth_len)
        f_pairs = overlap_force + soft_force

        active = f_pairs > 0.005
        if np.any(active):
            idx_active = np.where(active)[0]
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            for idx in idx_active:
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                d = d_pairs[idx]
                if d > 1e-6:
                    ux = dx_pairs[idx] / d
                    uy = dy_pairs[idx] / d
                    f = f_pairs[idx]
                    fx[i_idx] += ux * f
                    fy[i_idx] += uy * f
                    fx[j_idx] -= ux * f
                    fy[j_idx] -= uy * f

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0003
        elif max_force > 0.01:
            dt = 0.0006
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    gap_eps = 0.05
    for _ in range(400):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        any_force = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if 0 < gap < gap_eps:
                    force = (gap_eps - gap) / gap_eps
                    force *= force
                    if d > 1e-10:
                        ux = dx / d
                        uy = dy / d
                        fx[i] += ux * force * 0.4
                        fy[i] += uy * force * 0.4
                        fx[j] -= ux * force * 0.4
                        fy[j] -= uy * force * 0.4
                        any_force = True
        if not any_force:
            break
        step = 0.001
        x += step * fx
        y += step * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        if _is_valid(x, y, r, w, h):
            r_trial = r * 1.0001
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
        else:
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


# ---------------------------------------------------------------------------
# NEW: Sequential radial growth via a smooth pressure field
# ---------------------------------------------------------------------------
def _softmin_clearances(x, y, r, w, h, temperature=0.01):
    """Log-sum-exp soft-min of each circle's clearance to neighbors and walls.

    For circle i, the true clearance is min over {walls, other circles} of
    (distance - r_other). We approximate the min with a soft-min so the
    pressure field is smooth. Returns array of soft-clearances.
    """
    n = N
    # Wall clearances (4 per circle)
    wall = np.stack([
        x - r,                 # left
        w - x - r,             # right
        y - r,                 # bottom
        h - y - r,             # top
    ], axis=1)  # shape (n, 4)

    # Neighbor clearances: for circle i, d_ij - r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    # clearance[i, j] = d[i,j] - r[j]
    clr = d - r[None, :]
    np.fill_diagonal(clr, np.inf)  # exclude self

    # Combine walls + neighbors per circle
    # walls: (n,4), neighbors: (n, n) -> (n, n+4)
    all_clr = np.concatenate([clr, wall], axis=1)  # (n, n+4)

    # soft-min via -logsumexp(-x)/1  (temperature controls sharpness)
    neg = -all_clr / temperature
    neg_max = np.max(neg, axis=1, keepdims=True)
    # softmax over exp(neg - neg_max)
    exp_neg = np.exp(neg - neg_max)
    lse = np.log(np.sum(exp_neg, axis=1)) + neg_max.squeeze(1)
    soft_min = -lse * temperature

    return soft_min


def _pressure_growth(z, time_budget):
    """Sequential radial growth: expand each circle by its admissible increase,
    most-constrained first, interleaved with force relaxation.

    Uses the smooth soft-min clearance as a "pressure" indicator: circles with
    the smallest clearance are the most constrained and get priority for
    repositioning, while the growth amount per circle is its exact local
    clearance (max radius - current radius).
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False

        # Compute exact local clearance (max admissible radius - current)
        clearance = np.zeros(n)
        for i in range(n):
            r_max = _max_radius_at(x, y, r, w, h, i)
            clearance[i] = r_max - r[i]

        # Smooth pressure field (for ordering)
        soft_clr = _softmin_clearances(x, y, r, w, h, temperature=0.01)

        # Order: most-constrained (smallest soft clearance) first
        order = np.argsort(soft_clr)

        for i in order:
            if time.time() > time_budget:
                break

            if clearance[i] > 1e-7:
                # Binary search for max growable radius
                lo = r[i]
                hi = r[i] + clearance[i]
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

        # Interleaved relaxation to escape local traps after growth
        z_temp = np.empty(3 * N + 1)
        z_temp[0::3][:n] = x
        z_temp[1::3][:n] = y
        z_temp[2::3][:n] = r
        z_temp[-1] = w
        z_relaxed = _force_relax_unified(z_temp, max_iter=150)
        x = z_relaxed[0::3][:n]
        y = z_relaxed[1::3][:n]
        r = z_relaxed[2::3][:n]
        w = z_relaxed[-1]
        h = 2.0 - w

    # Final uniform growth
    grow_factor = 1.001
    for _ in range(80):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
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


# ---------------------------------------------------------------------------
# Expansion / relocation polish (keep from current program)
# ---------------------------------------------------------------------------
def _expansion_relaxation_polish(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
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

            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True

            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]

            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j

            if min_dist < 0.02:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d

                directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
                found = False
                for dx_dir, dy_dir in directions:
                    norm = np.sqrt(dx_dir ** 2 + dy_dir ** 2)
                    if norm > 0:
                        dx_dir /= norm
                        dy_dir /= norm
                    for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                        new_x = x[i] + dx_dir * step
                        new_y = y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if not found:
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

        if time.time() < time_budget:
            z_temp = np.empty(3 * N + 1)
            z_temp[0::3][:n] = x
            z_temp[1::3][:n] = y
            z_temp[2::3][:n] = r
            z_temp[-1] = w
            z_relaxed = _force_relax_unified(z_temp, max_iter=100)
            x = z_relaxed[0::3][:n]
            y = z_relaxed[1::3][:n]
            r = z_relaxed[2::3][:n]
            w = z_relaxed[-1]
            h = 2.0 - w

    grow_factor = 1.001
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
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


# ---------------------------------------------------------------------------
# Contact graph optimization
# ---------------------------------------------------------------------------
def _contact_graph_optimize(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    epsilon = 1e-3
    contact_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] + epsilon:
                contact_pairs.append((i, j))

    def objective_contact(zv):
        x_v = zv[0::3][:n]
        y_v = zv[1::3][:n]
        r_v = zv[2::3][:n]
        w_v = zv[-1]
        h_v = 2.0 - w_v

        if w_v < 0.2 or h_v < 0.2 or np.any(r_v < 0):
            return 1e6

        obj = -np.sum(r_v)

        for i, j in contact_pairs:
            dx = x_v[i] - x_v[j]
            dy = y_v[i] - y_v[j]
            d = np.sqrt(dx * dx + dy * dy)
            target_d = r_v[i] + r_v[j]
            gap = d - target_d
            if gap > 0:
                obj += 10.0 * gap * gap

        dx_all = x_v[:, None] - x_v[None, :]
        dy_all = y_v[:, None] - y_v[None, :]
        d_sq_all = dx_all * dx_all + dy_all * dy_all
        r_sum_all = r_v[:, None] + r_v[None, :]

        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq_all[iu], 0.0))
        overlap = np.maximum(r_sum_all[iu] - d_pairs, 0.0)
        obj += 1000.0 * np.sum(overlap * overlap)

        left_viol = np.maximum(r_v - x_v, 0.0)
        right_viol = np.maximum(x_v + r_v - w_v, 0.0)
        bottom_viol = np.maximum(r_v - y_v, 0.0)
        top_viol = np.maximum(y_v + r_v - h_v, 0.0)
        obj += 1000.0 * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )
        return obj

    try:
        maxiter = int(min(500, time_budget * 100))
        res = minimize(objective_contact, z, method='SLSQP',
                       bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt
        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            if np.sum(r_opt) > np.sum(r):
                return z_opt
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Constraint-based SLSQP refinement
# ---------------------------------------------------------------------------
def _constraint_refine(z, time_budget):
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(800, time_budget * 150))
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
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Initial guesses
# ---------------------------------------------------------------------------
def _get_initial_guesses():
    guesses = []

    # Hexagonal lattice initializations
    aspect_ratios_hex = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    for w0 in aspect_ratios_hex:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        z[-1] = w0
        r_est = np.sqrt(w0 * h0 / (N * np.pi * 0.9))

        points = []
        y = r_est
        row_idx = 0
        while y < h0 - r_est * 0.5:
            x_offset = 0 if row_idx % 2 == 0 else r_est
            x = r_est + x_offset
            while x < w0 - r_est * 0.5:
                if r_est <= x <= w0 - r_est and r_est <= y <= h0 - r_est:
                    points.append((x, y))
                x += 2 * r_est
            y += r_est * np.sqrt(3)
            row_idx += 1

        if len(points) >= N:
            for i in range(N):
                cx, cy = points[i]
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r_est
            guesses.append(z)
        else:
            for scale in [0.9, 0.8, 0.7]:
                r_est_new = r_est * scale
                points = []
                y = r_est_new
                row_idx = 0
                while y < h0 - r_est_new * 0.5:
                    x_offset = 0 if row_idx % 2 == 0 else r_est_new
                    x = r_est_new + x_offset
                    while x < w0 - r_est_new * 0.5:
                        if r_est_new <= x <= w0 - r_est_new and r_est_new <= y <= h0 - r_est_new:
                            points.append((x, y))
                        x += 2 * r_est_new
                    y += r_est_new * np.sqrt(3)
                    row_idx += 1
                if len(points) >= N:
                    for i in range(N):
                        cx, cy = points[i]
                        z[3 * i] = cx
                        z[3 * i + 1] = cy
                        z[3 * i + 2] = r_est_new
                    guesses.append(z)
                    break

    # Square grid initializations
    side = int(np.ceil(np.sqrt(N)))
    aspect_ratios_sq = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
    for w0 in aspect_ratios_sq:
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

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

    guesses = _get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Smooth gap-aware polish
        polished = _polish_result_smooth(best_result, deadline - (time.time() - start_time) - 18)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. NEW: Pressure-field sequential radial growth
        if best_result is not None and time.time() < deadline - 18:
            grown = _pressure_growth(best_result, deadline - (time.time() - start_time) - 10)
            if grown is not None:
                valid, val = check_validity(grown)
                if valid and val > best_value:
                    best_value = val
                    best_result = grown

        # 3. Expansion / relocation polish
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

            if best_result is not None and time.time() < deadline - 10:
                expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
                if expanded2 is not None:
                    valid, val = check_validity(expanded2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = expanded2

        # 4. Contact graph optimization
        if best_result is not None and time.time() < deadline - 12:
            contact_res = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 6)
            if contact_res is not None:
                valid, val = check_validity(contact_res)
                if valid and val > best_value:
                    best_value = val
                    best_result = contact_res

        # 5. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 6. Final smooth polish pass
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result_smooth(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

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

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
