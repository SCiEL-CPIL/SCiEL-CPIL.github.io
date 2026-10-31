"""Circle packing in perimeter-4 rectangle: PBD + SLSQP + LSE Epigraph + Constraint Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y, tol=1e-10):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] - tol or new_x > w - r[i] + tol:
        return False
    if new_y < r[i] - tol or new_y > h - r[i] + tol:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - tol:
            return False
    return True


def _max_radius_for_circle(x, y, r, w, h, i):
    """Compute maximum admissible radius for circle i given current positions."""
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


def _sequential_radial_growth(x, y, r, w, h, max_rounds=50, time_budget=None):
    """Iteratively grow each circle to its maximum admissible radius."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for rnd in range(max_rounds):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return x, y, r


def _pbd_solve(x, y, r, w, h, target_r, n_iters=200, time_budget=None):
    """
    Position-Based Dynamics: project positions to satisfy non-overlap and wall constraints
    while trying to achieve target radii. Returns (x, y, achieved_r).
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    tr = target_r.copy().astype(float)
    
    for it in range(n_iters):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # 1. Wall projection: clamp positions so circles stay inside
        x = np.clip(x, tr, w - tr)
        y = np.clip(y, tr, h - tr)
        
        # 2. Pairwise non-overlap projection
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = tr[i] + tr[j]
                if d < min_d - 1e-12:
                    if d < 1e-12:
                        # Overlapping centers: push apart arbitrarily
                        dx = 1e-6
                        dy = 0.0
                        d = 1e-6
                    correction = (min_d - d) / 2.0
                    nx = dx / d
                    ny = dy / d
                    x[i] -= nx * correction
                    y[i] -= ny * correction
                    x[j] += nx * correction
                    y[j] += ny * correction
        
        # Re-clamp after pairwise correction
        x = np.clip(x, tr, w - tr)
        y = np.clip(y, tr, h - tr)
    
    # Compute achieved radii: for each circle, the max radius that fits at its position
    achieved = np.zeros(N)
    for i in range(N):
        achieved[i] = _max_radius_for_circle(x, y, np.full(N, 0.0), w, h, i)
        # But we also need to account for other circles' radii
        # Actually, recompute properly:
        achieved[i] = min(x[i], w - x[i], y[i], h - y[i])
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            # The constraint is d >= achieved[i] + achieved[j], but we're computing sequentially
            # For a proper PBD, we'd iterate. For now, use the current r as proxy:
            # Actually, let's just return the positions and let the caller grow radii.
    
    return x, y


def _pbd_grow(x, y, w, h, target_r, n_iters=100, time_budget=None):
    """
    PBD-style iterative radius growth with position adjustment.
    Tries to grow radii toward target_r while keeping circles non-overlapping and in-bounds.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = np.full(N, 0.001)
    tr = target_r.copy().astype(float)
    
    for it in range(n_iters):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # Try to grow each radius toward target
        improved = False
        for i in range(N):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max < tr[i]:
                # Can't reach target; set to max possible
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True
            else:
                # Grow toward target gradually
                new_r = min(tr[i], r[i] + (tr[i] - r[i]) * 0.1 + 0.001)
                if new_r > r[i] + 1e-9:
                    r[i] = new_r
                    improved = True
        
        if not improved:
            break
        
        # PBD position correction: resolve any overlaps created by growth
        for _ in range(20):
            any_overlap = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-10:
                        any_overlap = True
                        if d < 1e-12:
                            dx, dy = 1e-6, 0.0
                            d = 1e-6
                        corr = (min_d - d) / 2.0
                        nx, ny = dx / d, dy / d
                        x[i] -= nx * corr
                        y[i] -= ny * corr
                        x[j] += nx * corr
                        y[j] += ny * corr
            
            # Wall correction
            for i in range(N):
                if x[i] < r[i]:
                    x[i] = r[i]
                if x[i] > w - r[i]:
                    x[i] = w - r[i]
                if y[i] < r[i]:
                    y[i] = r[i]
                if y[i] > h - r[i]:
                    y[i] = h - r[i]
            
            if not any_overlap:
                break
    
    # Final growth pass
    for _ in range(50):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
        if not improved:
            break
    
    return x, y, r


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
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
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=400, time_budget=None):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003

    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
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
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-9:
            break

        dx_pairs = dx[iu]
        dy_pairs = dy[iu]

        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            if overlap[idx] > 0:
                d = d_pairs[idx]
                if d > 1e-8:
                    ux = dx_pairs[idx] / d
                    uy = dy_pairs[idx] / d
                    f = overlap[idx] * 10.0
                    fx[i_idx] += ux * f
                    fy[i_idx] += uy * f
                    fx[j_idx] -= ux * f
                    fy[j_idx] -= uy * f

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_smooth(z, time_budget):
    """Smooth polish with gap-aware repulsion."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0005
    max_iter = int(min(2000, max(100, (time_budget - time.time()) * 100)))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 30
        fx += right_viol * 30
        fy -= bottom_viol * 30
        fy += top_viol * 30

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        gap = d_pairs - r_sum_pairs
        overlap_force = np.maximum(-gap, 0.0) * 15.0
        soft_force = 0.3 * np.exp(-gap / 0.05)
        f_pairs = overlap_force + soft_force

        active = f_pairs > 0.01
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

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0002
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
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

    maxiter = int(min(800, max(50, (time_budget - time.time()) * 150)))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 200})

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


def _lse_refine(z, time_budget):
    """
    Log-sum-exp epigraph refinement: use a smooth surrogate for the non-convex
    distance constraints to guide SLSQP to a better local optimum.
    """
    z = z.copy().astype(float)
    n = N
    w0 = z[-1]
    h0 = 2.0 - w0

    def lse_obj(p):
        x = p[0::3][:n]
        y = p[1::3][:n]
        r = p[2::3][:n]
        w = p[-1]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        obj = -np.sum(r)
        # Smooth pairwise penalty via log-sum-exp of squared overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        # Soft penalty: for each pair, penalty = max(0, r_sum - d)^2
        # Approximate with smooth max
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        viol = r_sum_pairs - d_pairs
        # Soft max via LSE
        beta = 50.0
        m = np.max(viol, axis=0) if viol.ndim > 1 else viol.max()
        lse = m + np.log(np.sum(np.exp(beta * (viol - m)))) / beta
        obj += 500.0 * lse
        # Boundary
        bv = np.maximum(r - x, 0) + np.maximum(x + r - w, 0) + np.maximum(r - y, 0) + np.maximum(y + r - h, 0)
        obj += 500.0 * np.sum(bv * bv)
        return obj

    maxiter = int(min(600, max(30, (time_budget - time.time()) * 100)))
    try:
        res = minimize(lse_obj, z, method='SLSQP',
                       bounds=[(0, 2)] * (3 * n) + [(0.3, 1.7)],
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
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


def _local_search_perturb(z, time_budget):
    """Try small perturbations of individual circles to escape local optima."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())

    for i in range(n):
        if time.time() > time_budget:
            break
        for dx_dir in [-1, 0, 1]:
            for dy_dir in [-1, 0, 1]:
                if dx_dir == 0 and dy_dir == 0:
                    continue
                norm = np.sqrt(dx_dir * dx_dir + dy_dir * dy_dir)
                for step in [0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + dx_dir / norm * step
                    new_y = y[i] + dy_dir / norm * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        r_max = _max_radius_for_circle(x, y, r, w, h, i)
                        if r_max > r[i] + 1e-8:
                            r[i] = r_max
                        x2, y2, r2 = _sequential_radial_growth(x, y, r, w, h, max_rounds=5)
                        val = np.sum(r2)
                        if val > best_val:
                            best_val = val
                            best_state = (x2.copy(), y2.copy(), r2.copy())
                        x[i] = best_state[0][i]
                        y[i] = best_state[1][i]
                        r[i] = best_state[2][i]

    x, y, r = best_state
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _get_hex_initialization(w0, h0, n_circles=N):
    """Generate hexagonal lattice initialization."""
    z = np.zeros(3 * N + 1)
    z[-1] = w0
    r_est = np.sqrt(w0 * h0 / (n_circles * np.pi * 0.95))
    points = []
    row_idx = 0
    y = r_est
    while y < h0 - r_est * 0.3:
        x_offset = 0 if row_idx % 2 == 0 else r_est * 0.9
        x = r_est + x_offset
        while x < w0 - r_est * 0.3:
            if x >= r_est and x <= w0 - r_est and y >= r_est and y <= h0 - r_est:
                points.append((x, y))
            x += 2 * r_est * 0.95
        y += r_est * np.sqrt(3) * 0.95
        row_idx += 1
    if len(points) >= n_circles:
        for i in range(n_circles):
            cx, cy = points[i]
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r_est * 0.9
        return z
    return None


def _get_grid_initialization(w0, h0, n_circles=N):
    """Generate grid initialization."""
    side = int(np.ceil(np.sqrt(n_circles)))
    z = np.zeros(3 * N + 1)
    for i in range(n_circles):
        row, col = divmod(i, side)
        cx = (col + 0.5) / side * w0
        cy = (row + 0.5) / side * h0
        r0 = min(w0 / side, h0 / side) * 0.45
        z[3 * i] = cx
        z[3 * i + 1] = cy
        z[3 * i + 2] = r0
    z[-1] = w0
    return z


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

    def get_initial_guesses():
        guesses = []
        aspect_ratios_hex = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
        for w0 in aspect_ratios_hex:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = _get_hex_initialization(w0, h0)
            if z is not None:
                guesses.append(z)
        aspect_ratios_sq = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
        for w0 in aspect_ratios_sq:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = _get_grid_initialization(w0, h0)
            guesses.append(z)
        rng = np.random.default_rng(42)
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
        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force-directed relaxation
        z0_relaxed = _force_relax(z0, max_iter=400, time_budget=deadline - 28)

        x = z0_relaxed[0::3][:N]
        y = z0_relaxed[1::3][:N]
        r = z0_relaxed[2::3][:N]
        w = z0_relaxed[-1]
        h = 2.0 - w

        # PBD-based growth: use target radii from area estimate
        r_target = np.full(N, np.sqrt(w * h / (N * np.pi * 0.92)))
        x_pbd, y_pbd, r_pbd = _pbd_grow(x, y, w, h, r_target, n_iters=80, time_budget=deadline - 25)

        # Sequential radial growth
        x_gr, y_gr, r_gr = _sequential_radial_growth(x_pbd, y_pbd, r_pbd, w, h, max_rounds=30, time_budget=deadline - 22)

        # SLSQP optimization
        z_opt = np.empty(3 * N + 1)
        z_opt[0::3][:N] = x_gr
        z_opt[1::3][:N] = y_gr
        z_opt[2::3][:N] = r_gr
        z_opt[-1] = w

        remaining = deadline - (time.time() - start_time)
        if remaining > 18:
            maxiter = int(min(500, max(100, remaining * 35)))
            try:
                res = minimize(_vectorized_objective, z_opt, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
                z_slsqp = res.x
                valid, val = check_validity(z_slsqp)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_slsqp.copy()
            except Exception:
                pass

        valid, val = check_validity(z_opt)
        if valid and val > best_value:
            best_value = val
            best_result = z_opt.copy()

    # --- Phase 2: Advanced Polish ---
    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_smooth(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    if best_result is not None and time.time() < deadline - 18:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 10)
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    if best_result is not None and time.time() < deadline - 12:
        lse_res = _lse_refine(best_result, deadline - (time.time() - start_time) - 6)
        if lse_res is not None:
            valid, val = check_validity(lse_res)
            if valid and val > best_value:
                best_value = val
                best_result = lse_res

    if best_result is not None and time.time() < deadline - 8:
        perturbed = _local_search_perturb(best_result, deadline - (time.time() - start_time) - 4)
        if perturbed is not None:
            valid, val = check_validity(perturbed)
            if valid and val > best_value:
                best_value = val
                best_result = perturbed

    if best_result is not None and time.time() < deadline - 5:
        refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
        if refined2 is not None:
            valid, val = check_validity(refined2)
            if valid and val > best_value:
                best_value = val
                best_result = refined2

    if best_result is not None and time.time() < deadline - 2:
        polished2 = _polish_smooth(best_result, deadline - (time.time() - start_time))
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

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
