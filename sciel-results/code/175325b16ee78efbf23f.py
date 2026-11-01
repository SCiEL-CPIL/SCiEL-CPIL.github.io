import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 138.0

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _objective(params, start, beta=30.0, lambda_eq=0.0):
    """
    Objective: maximize sum of radii with smooth penalty for violations.
    """
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    
    obj = -np.sum(r)
    
    # Pairwise constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    violations = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    v_pairs = violations[mask]
    
    v_max = np.max(v_pairs)
    if v_max > 0:
        lse = v_max + np.log(np.sum(np.exp(beta * (v_pairs - v_max)))) / beta
    else:
        lse = 0.0
    
    # Boundary constraints
    bviol = np.maximum(0, r - x) + np.maximum(0, x - (1.0 - r)) + \
            np.maximum(0, r - y) + np.maximum(0, y - (1.0 - r))
    b_max = np.max(bviol)
    if b_max > 0:
        lse_b = b_max + np.log(np.sum(np.exp(beta * (bviol - b_max)))) / beta
    else:
        lse_b = 0.0
    
    mu = 200.0
    result = obj + mu * (lse + lse_b)
    
    # Optional: encourage equal radii
    if lambda_eq > 0:
        r_mean = np.mean(r)
        r_var = np.sum((r - r_mean)**2)
        result += lambda_eq * r_var
    
    return result

def _objective_grad(params, start, beta=30.0, lambda_eq=0.0):
    """Gradient of the objective function."""
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = params[2*N:3*N].copy()
    r = np.maximum(r, 0.0)
    
    grad = np.zeros(3*N)
    
    # Gradient of -sum(r)
    grad[2*N:3*N] = -1.0
    
    # Pairwise constraints gradient
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    violations = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    v_pairs = violations[mask]
    
    if np.max(v_pairs) > 0:
        v_max = np.max(v_pairs)
        # Softmax weights
        exps = np.exp(beta * (v_pairs - v_max))
        weights = exps / np.sum(exps)
        
        # Reconstruct full matrix of weights (symmetric)
        w_mat = np.zeros((N, N))
        idx = np.where(mask)
        w_mat[idx] = weights
        w_mat = w_mat + w_mat.T
        
        # Gradient of log-sum-exp
        # d/dx_i of lse = sum_j w_ij * (r_i + r_j - dist_ij) * (-dx_ij/dist_ij)
        for i in range(N):
            for j in range(i+1, N):
                w = w_mat[i, j]
                if w < 1e-12:
                    continue
                d = dist[i, j]
                if d < 1e-12:
                    continue
                # Gradient w.r.t x_i
                gx = w * (-dx[i, j] / d)
                gy = w * (-dy[i, j] / d)
                # Gradient w.r.t r_i
                gr_i = w
                # Gradient w.r.t r_j
                gr_j = w
                
                grad[i] += gx
                grad[N + i] += gy
                grad[2*N + i] += gr_i
                grad[j] -= gx
                grad[N + j] -= gy
                grad[2*N + j] += gr_j
    
    # Boundary constraints gradient
    bviol = np.maximum(0, r - x) + np.maximum(0, x - (1.0 - r)) + \
            np.maximum(0, r - y) + np.maximum(0, y - (1.0 - r))
    b_max = np.max(bviol)
    if b_max > 0:
        exps = np.exp(beta * (bviol - b_max))
        weights = exps / np.sum(exps)
        
        for i in range(N):
            w = weights[i]
            if w < 1e-12:
                continue
            # d/dx_i: from max(0, r-x) -> -1 if r>x, from max(0, x-(1-r)) -> +1 if x>1-r
            if r[i] > x[i]:
                grad[i] += -w
            if x[i] > 1.0 - r[i]:
                grad[i] += w
            if r[i] > y[i]:
                grad[N+i] += -w
            if y[i] > 1.0 - r[i]:
                grad[N+i] += w
            # d/dr_i: from max(0, r-x) -> +1 if r>x, from max(0, x-(1-r)) -> +1 if x>1-r
            if r[i] > x[i]:
                grad[2*N+i] += w
            if x[i] > 1.0 - r[i]:
                grad[2*N+i] += w
            if r[i] > y[i]:
                grad[2*N+i] += w
            if y[i] > 1.0 - r[i]:
                grad[2*N+i] += w
    
    # Equalization gradient
    if lambda_eq > 0:
        r_mean = np.mean(r)
        grad[2*N:3*N] += lambda_eq * 2 * (r - r_mean)
    
    # Scale by mu
    mu = 200.0
    grad = grad * mu + grad[2*N:3*N] * 0  # already included
    
    return grad

def _push_apart(x, y, r, iters=300):
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = (dist < need) & (np.abs(np.triu(np.ones((N, N), dtype=bool), k=1).astype(float)) > 0)
        
        if not np.any(overlaps):
            break
        
        disp_x = np.zeros(N)
        disp_y = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not overlaps[i, j]:
                    continue
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                d = np.hypot(dx_ij, dy_ij)
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx_ij / d, dy_ij / d
                    gap = need_ij - d
                    step = 0.5 * gap
                    disp_x[i] += step * ux
                    disp_y[i] += step * uy
                    disp_x[j] -= step * ux
                    disp_y[j] -= step * uy
        
        x += disp_x
        y += disp_y
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    return x, y

def _make_valid(x, y, r):
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(30):
        x, y = _push_apart(x, y, r, iters=150)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if _is_valid(x, y, r):
            break
    
    if not _is_valid(x, y, r):
        for _ in range(300):
            for i in range(N):
                mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > mr * 0.999:
                    r[i] = max(0.0, mr * 0.998)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx*dx + dy*dy + 1e-14)
            need = r[:, None] + r[None, :]
            overlaps = dist < need - 1e-10
            
            if not np.any(overlaps):
                break
            
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99
    
    return x, y, r

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.98)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, 0.07)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _voronoi_init():
    rng = np.random.RandomState(42)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    dists = np.sqrt((x[:,None]-x[None,:])**2 + (y[:,None]-y[None,:])**2)
    np.fill_diagonal(dists, np.inf)
    r = 0.45 * np.min(dists, axis=1)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _phyllotaxis_init():
    phi = np.pi * (3.0 - np.sqrt(5.0))
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        r = np.sqrt((i + 0.5) / N) * 0.45
        theta = i * phi
        x[i] = 0.5 + r * np.cos(theta)
        y[i] = 0.5 + r * np.sin(theta)
    
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _lbfgsb_optimize(x, y, r, start, max_iter=3000, beta=30.0, lambda_eq=0.0):
    """L-BFGS-B with smooth penalty."""
    if _elapsed(start) > TIME_LIMIT - 10:
        return x, y, r
    
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    p0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.001, 0.45))
    
    try:
        res = minimize(
            _objective, p0, args=(start, beta, lambda_eq),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-8, 'maxfun': 10000}
        )
        vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
        if _is_valid(vx, vy, vr):
            s = _sum_r(vr)
            if s > best_score:
                best = (vx, vy, vr)
                best_score = s
    except Exception:
        pass
    
    return best

def _coordinate_descent(x, y, r, start, max_rounds=80):
    """
    Refine positions using coordinate descent.
    For each circle, move it to maximize the minimum distance to other circles and boundaries.
    """
    x = x.copy()
    y = y.copy()
    best_score = _sum_r(r)
    
    for _round in range(max_rounds):
        if _elapsed(start) > TIME_LIMIT - 5:
            break
            
        improved = False
        for i in range(N):
            cx, cy = x[i], y[i]
            ri = r[i]
            
            curr_min_slack = min(cx-ri, 1-cx-ri, cy-ri, 1-cy-ri)
            for j in range(N):
                if i == j: continue
                d = np.hypot(cx-x[j], cy-y[j])
                curr_min_slack = min(curr_min_slack, d - ri - r[j])
            best_slack = curr_min_slack
            best_dx, best_dy = 0, 0
            
            scale = 0.01
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = cx + dx, cy + dy
                    if nx < ri or nx > 1-ri or ny < ri or ny > 1-ri:
                        continue
                    min_slack = min(nx-ri, 1-nx-ri, ny-ri, 1-ny-ri)
                    for j in range(N):
                        if i == j: continue
                        d = np.hypot(nx-x[j], ny-y[j])
                        min_slack = min(min_slack, d - ri - r[j])
                    if min_slack > best_slack:
                        best_slack = min_slack
                        best_dx, best_dy = dx, dy
            
            if best_dx != 0 or best_dy != 0:
                x[i] += best_dx
                y[i] += best_dy
                improved = True
        
        if not improved:
            break
            
        if _is_valid(x, y, r):
            for i in range(N):
                slack_b = min(x[i]-r[i], 1-x[i]-r[i], y[i]-r[i], 1-y[i]-r[i])
                slack_p = 1e9
                for j in range(N):
                    if i == j: continue
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    slack_p = min(slack_p, d - r[i] - r[j])
                max_dr = min(slack_b, slack_p)
                if max_dr > 1e-8:
                    r[i] += max_dr * 0.5
    
    return x, y, r

def _optimize(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # L-BFGS-B with standard objective
    if _elapsed(start) < TIME_LIMIT - 20:
        try:
            vx, vy, vr = _lbfgsb_optimize(x, y, r, start, max_iter=2000, beta=30.0)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # L-BFGS-B with equalization
    if _elapsed(start) < TIME_LIMIT - 15:
        try:
            vx, vy, vr = _lbfgsb_optimize(best[0], best[1], best[2], start, max_iter=1500, beta=30.0, lambda_eq=30.0)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Coordinate Descent
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            vx, vy, vr = _coordinate_descent(best[0], best[1], best[2], start, max_rounds=50)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    return best

def _tight_polish(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # L-BFGS-B with very tight log-sum-exp
    if _elapsed(start) < TIME_LIMIT - 8:
        try:
            vx, vy, vr = _lbfgsb_optimize(best[0], best[1], best[2], start, max_iter=5000, beta=50.0)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # L-BFGS-B with equalization for tighter uniform packing
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            vx, vy, vr = _lbfgsb_optimize(best[0], best[1], best[2], start, max_iter=3000, beta=50.0, lambda_eq=50.0)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Coordinate Descent Polish
    if _elapsed(start) < TIME_LIMIT - 1:
        try:
            vx, vy, vr = _coordinate_descent(best[0], best[1], best[2], start, max_rounds=100)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
            
    return best

def _grow_radii(x, y, r, start, max_rounds=50):
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_rounds):
        if _elapsed(start) > TIME_LIMIT - 2:
            break
        
        improved = False
        for i in range(N):
            slack_boundary = min(x[i] - r[i], 1.0 - x[i] - r[i],
                                  y[i] - r[i], 1.0 - y[i] - r[i])
            
            max_dr_pair = 1e9
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                slack = d - r[i] - r[j]
                if slack < max_dr_pair:
                    max_dr_pair = slack
            
            max_dr = min(slack_boundary, max_dr_pair)
            if max_dr > 1e-10:
                r[i] += max_dr * 0.995
                improved = True
        
        if not improved:
            break
        
        x, y = _push_apart(x, y, r, iters=150)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        if not _is_valid(x, y, r):
            r = np.minimum(r, 0.995 * r)
            x, y = _push_apart(x, y, r, iters=150)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
    
    return x, y, r

def _final_polish(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # L-BFGS-B with high beta log-sum-exp
    if _elapsed(start) < TIME_LIMIT - 12:
        try:
            vx, vy, vr = _lbfgsb_optimize(best[0], best[1], best[2], start, max_iter=5000, beta=100.0)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # L-BFGS-B with equalization for final uniform refinement
    if _elapsed(start) < TIME_LIMIT - 8:
        try:
            vx, vy, vr = _lbfgsb_optimize(best[0], best[1], best[2], start, max_iter=5000, beta=80.0, lambda_eq=80.0)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Tight polish
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = _tight_polish(*best, start)
            if _is_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best = res
                    best_score = s
        except Exception:
            pass
    
    # Grow radii
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            vx, vy, vr = _grow_radii(*best, start, max_rounds=40)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Final Coordinate Descent
    if _elapsed(start) < TIME_LIMIT - 1:
        try:
            vx, vy, vr = _coordinate_descent(*best, start, max_rounds=200)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
            
    return best

def construct():
    start = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_phyllotaxis_init())
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_voronoi_init())
    
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])
    
    # Optimize each initialization with time budget
    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 30:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res
    
    # Final high-precision polish
    if _elapsed(start) < TIME_LIMIT - 15:
        res = _final_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res
                best_score = s
    
    # One more round of radius growth
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            vx, vy, vr = _grow_radii(*best, start, max_rounds=30)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
            
    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
