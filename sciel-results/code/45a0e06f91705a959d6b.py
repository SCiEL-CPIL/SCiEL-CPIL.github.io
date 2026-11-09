"""Circle packing in unit square: 26 circles, maximize sum of radii.
Combined approach:
1. Vectorized force-directed relaxation (from current)
2. Constraint-aware local search with radius maximization (from current)
3. Robust repair that guarantees validity (from inspiration)
4. Scipy-based refinement for final polish (from inspiration)
5. Random perturbation local search (from inspiration)
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 118.0  # safety margin before 150s kill


def _validate(x, y, r, tol=1e-9):
    """Check validity: boundary and no overlaps."""
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or \
       np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    if np.any(d < r_sum - tol):
        return False
    return True


def _repair_config(x, y, r):
    """Robustly repair a configuration to be valid by shrinking radii."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Clamp to boundary
    r_max_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    r = np.minimum(r, r_max_bound)
    r = np.maximum(r, 0.0)
    
    # Iteratively fix overlaps by shrinking the larger circle in the worst pair
    for _ in range(500):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum - 1e-9
        np.fill_diagonal(overlaps, False)
        
        if not np.any(overlaps):
            break
        
        # Find the most overlapping pair
        max_overlap = 0
        max_i, max_j = 0, 0
        for i in range(N):
            for j in range(i + 1, N):
                if overlaps[i, j]:
                    ov = r_sum[i, j] - d[i, j]
                    if ov > max_overlap:
                        max_overlap = ov
                        max_i, max_j = i, j
        
        # Shrink the larger circle by the overlap amount plus a small buffer
        shrink_amount = max_overlap + 1e-5
        if r[max_i] >= r[max_j]:
            r[max_i] = max(0, r[max_i] - shrink_amount)
        else:
            r[max_j] = max(0, r[max_j] - shrink_amount)
    
    # Final safety check
    if not _validate(x, y, r):
        r *= 0.9
    
    return x, y, r


def _max_radius_for_position(x, y, r):
    """For each circle i, compute max radius allowed by boundary and
    distance to all other circles (given their current radii). Vectorized."""
    # Boundary constraint
    r_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Pairwise constraints: for each i, r_i <= d(i,j) - r_j for all j != i
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    # For each i, the constraint is r_i <= min_j (d[i,j] - r[j]) for j != i
    # d[i,j] - r[j] is a matrix; take min over j (axis=1)
    r_constraint = d - r[None, :]  # shape (N, N), entry [i,j] = d[i,j] - r[j]
    np.fill_diagonal(r_constraint, np.inf)
    r_pair = np.min(r_constraint, axis=1)
    
    r_max = np.minimum(r_max, r_pair)
    r_max = np.maximum(r_max, 1e-6)
    return r_max


def _vectorized_relax(x, y, r, lr=0.005, lam_ov=5.0, lam_bd=5.0,
                      max_steps=4000, deadline=None):
    """Fast vectorized force-directed relaxation. Returns (x, y, r)."""
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        d_safe = np.where(d < 1e-8, 1e-8, d)

        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        np.fill_diagonal(overlap, 0)

        gx_ov = -2 * lam_ov * np.sum(overlap * (dx / d_safe), axis=1)
        gy_ov = -2 * lam_ov * np.sum(overlap * (dy / d_safe), axis=1)
        gr_ov = 2 * lam_ov * np.sum(overlap, axis=1)

        bxl = np.maximum(0, r - x)
        bxr = np.maximum(0, r - (1 - x))
        byl = np.maximum(0, r - y)
        byr = np.maximum(0, r - (1 - y))

        gx_bd = -2 * lam_bd * bxl + 2 * lam_bd * bxr
        gy_bd = -2 * lam_bd * byl + 2 * lam_bd * byr
        gr_bd = 2 * lam_bd * (bxl + bxr + byl + byr)

        gr_obj = -np.ones(N)

        gx = gx_ov + gx_bd
        gy = gy_ov + gy_bd
        gr = gr_ov + gr_bd + gr_obj

        grad_mag = np.sqrt(np.sum(gx**2) + np.sum(gy**2) + np.sum(gr**2))
        eff_lr = lr * (10.0 / grad_mag) if grad_mag > 10 else lr

        x -= eff_lr * gx
        y -= eff_lr * gy
        r -= eff_lr * gr

        r = np.clip(r, 0.01, 0.4)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        lr *= 0.9995

    return x, y, r


def _local_search(x, y, r, deadline):
    """Constraint-aware local search:
    - Iteratively maximize radii given positions.
    - Perform pairwise separation moves to increase the bottleneck radius.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()

    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 2a: Radius maximization given positions
    for _ in range(50):
        if time.time() > deadline:
            break
        r_max = _max_radius_for_position(x, y, r)
        r = r_max
        r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, r_max_b)
        r = np.maximum(r, 1e-6)

        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 2b: Pairwise separation moves
    for iteration in range(200):
        if time.time() > deadline:
            break

        r_max = _max_radius_for_position(x, y, r)
        r = np.minimum(r_max, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
        r = np.maximum(r, 1e-6)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        tight = np.abs(d - r_sum) < 0.01
        np.fill_diagonal(tight, False)

        if not np.any(tight):
            break

        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if not tight[i, j]:
                    continue
                dirx = x[i] - x[j]
                diry = y[i] - y[j]
                norm = np.sqrt(dirx**2 + diry**2)
                if norm < 1e-8:
                    continue
                dirx /= norm
                diry /= norm

                delta = 0.005
                new_xi = x[i] + delta * dirx
                new_yi = y[i] + delta * diry
                new_xj = x[j] - delta * dirx
                new_yj = y[j] - delta * diry

                if (new_xi < 0 or new_xi > 1 or new_yi < 0 or new_yi > 1 or
                        new_xj < 0 or new_xj > 1 or new_yj < 0 or new_yj > 1):
                    continue

                old_xi, old_yi = x[i], y[i]
                old_xj, old_yj = x[j], y[j]
                x[i], y[i] = new_xi, new_yi
                x[j], y[j] = new_xj, new_yj

                r_new = _max_radius_for_position(x, y, r)
                r_new = np.minimum(r_new, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
                r_new = np.maximum(r_new, 1e-6)

                s_new = np.sum(r_new)
                if s_new > np.sum(r) + 1e-8:
                    r = r_new
                    moved = True
                else:
                    x[i], y[i] = old_xi, old_yi
                    x[j], y[j] = old_xj, old_yj

        if not moved:
            break

        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    if _validate(best_x, best_y, best_r):
        return best_x, best_y, best_r
    return x, y, r


def _random_perturbation_search(x, y, r, deadline, max_iter=500):
    """Random perturbation local search (from inspiration)."""
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = -np.sum(best_r)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        idx = np.random.randint(N)
        candidate_x = best_x.copy()
        candidate_y = best_y.copy()
        candidate_r = best_r.copy()
        
        progress = it / max_iter
        scale = 0.03 * (1 - progress) + 0.002
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        dr = np.random.normal(0, scale * 0.4)
        
        candidate_x[idx] += dx
        candidate_y[idx] += dy
        candidate_r[idx] += dr
        
        candidate_x[idx] = np.clip(candidate_x[idx], 0, 1)
        candidate_y[idx] = np.clip(candidate_y[idx], 0, 1)
        candidate_r[idx] = np.clip(candidate_r[idx], 0.001, 0.5)
        
        if _validate(candidate_x, candidate_y, candidate_r):
            val = -np.sum(candidate_r)
            if val < best_val:
                best_x, best_y, best_r = candidate_x, candidate_y, candidate_r
                best_val = val
    
    return best_x, best_y, best_r


def _scipy_refine(x, y, r, deadline):
    """Use scipy to optimize the configuration (from inspiration)."""
    def objective(params):
        x_ = params[:N]
        y_ = params[N:2*N]
        r_ = params[2*N:3*N]
        
        # Boundary violations
        bound_pen = 0.0
        for i in range(N):
            if r_[i] > x_[i]:
                bound_pen += (r_[i] - x_[i])**2
            if r_[i] > 1 - x_[i]:
                bound_pen += (r_[i] - (1 - x_[i]))**2
            if r_[i] > y_[i]:
                bound_pen += (r_[i] - y_[i])**2
            if r_[i] > 1 - y_[i]:
                bound_pen += (r_[i] - (1 - y_[i]))**2
        
        # Overlap violations
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x_[i] - x_[j]
                dy = y_[i] - y_[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r_[i] + r_[j]:
                    overlap_pen += (r_[i] + r_[j] - d)**2
        
        # Objective: minimize penalties, maximize sum of radii
        return bound_pen + overlap_pen - np.sum(r_)
    
    x0 = np.concatenate([x, y, r])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          bounds=[(0, 1)]*N + [(0, 1)]*N + [(0.001, 0.5)]*N,
                          options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-6})
        
        if time.time() > deadline:
            return x, y, r
            
        x_new = result.x[:N]
        y_new = result.x[N:2*N]
        r_new = result.x[2*N:3*N]
        
        if _validate(x_new, y_new, r_new):
            if np.sum(r_new) > np.sum(r):
                return x_new, y_new, r_new
    except Exception:
        pass
    
    return x, y, r


def _hex_init(n, scale=1.0):
    """Place n circles on a hexagonal lattice scaled to fit in unit square."""
    s = scale
    pts = []
    cols = int(np.ceil(1.0 / s)) + 2
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    for j in range(rows):
        y = j * s * np.sqrt(3) / 2
        offset = 0 if j % 2 == 0 else s / 2
        for i in range(cols):
            x = i * s + offset
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.rand(n - len(pts), 2) * 0.8 + 0.1
        pts = np.vstack([pts, extra])
    return pts[:n]


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    best_score = -np.inf
    best_config = None

    def _try_config(x, y, r):
        nonlocal best_score, best_config
        # Quick validity check
        if _validate(x, y, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_config = (x.copy(), y.copy(), r.copy())
        else:
            # Try to repair
            x_rep, y_rep, r_rep = _repair_config(x, y, r)
            if _validate(x_rep, y_rep, r_rep):
                s = np.sum(r_rep)
                if s > best_score:
                    best_score = s
                    best_config = (x_rep.copy(), y_rep.copy(), r_rep.copy())

    # Multiple diverse initializations
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        np.random.seed(1000 + restart * 137)

        if restart % 5 == 0:
            # Hex lattice
            scale = 0.18 - 0.01 * restart
            pts = _hex_init(N, scale)
            x = pts[:, 0].copy()
            y = pts[:, 1].copy()
            r = np.full(N, 0.03)
        elif restart % 5 == 1:
            # Jittered grid
            side = int(np.ceil(np.sqrt(N)))
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        px = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                        py = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                        px = np.clip(px, 0.1, 0.9)
                        py = np.clip(py, 0.1, 0.9)
                        xs.append(px)
                        ys.append(py)
                        rs.append(0.05)
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        elif restart % 5 == 2:
            # Random
            x = np.random.uniform(0.15, 0.85, N)
            y = np.random.uniform(0.15, 0.85, N)
            r = np.random.uniform(0.03, 0.07, N)
        elif restart % 5 == 3:
            # Clustered: few large, many small
            xs, ys, rs = [], [], []
            for _ in range(6):
                xs.append(np.random.uniform(0.2, 0.8))
                ys.append(np.random.uniform(0.2, 0.8))
                rs.append(np.random.uniform(0.07, 0.11))
            for _ in range(N - 6):
                xs.append(np.random.uniform(0.1, 0.9))
                ys.append(np.random.uniform(0.1, 0.9))
                rs.append(np.random.uniform(0.02, 0.05))
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        else:
            # Uniform grid with jitter
            side = 5
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        px = (i + 0.5) / side + np.random.uniform(-0.03, 0.03)
                        py = (j + 0.5) / side + np.random.uniform(-0.03, 0.03)
                        px = np.clip(px, 0.1, 0.9)
                        py = np.clip(py, 0.1, 0.9)
                        xs.append(px)
                        ys.append(py)
                        rs.append(0.04)
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)

        # Phase 1: Force-directed relaxation
        x, y, r = _vectorized_relax(x, y, r, lr=0.005, lam_ov=5.0, lam_bd=5.0,
                                    max_steps=2000, deadline=deadline)
        _try_config(x, y, r)

        if time.time() > deadline:
            break

        # Phase 2: Local search with radius maximization
        x2, y2, r2 = _local_search(x, y, r, deadline)
        _try_config(x2, y2, r2)

        if time.time() > deadline:
            break

        # Phase 3: Random perturbation search
        x3, y3, r3 = _random_perturbation_search(x2, y2, r2, deadline, max_iter=300)
        _try_config(x3, y3, r3)

        if time.time() > deadline:
            break

        # Phase 4: Scipy refinement (only if time allows)
        if time.time() < deadline - 10:
            x4, y4, r4 = _scipy_refine(x3, y3, r3, deadline)
            _try_config(x4, y4, r4)

    # Fallback: simple grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_config = (np.array([c[0] for c in cells[:N]]),
                       np.array([c[1] for c in cells[:N]]),
                       np.array([c[2] for c in cells[:N]]))

    x, y, r = best_config
    return np.column_stack((x, y, r))
