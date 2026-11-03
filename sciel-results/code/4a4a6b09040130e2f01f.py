"""Circle packing in unit square for N=26 circles.
Combined approach: primal-dual interior-point with Delaunay contact graph,
plus physics-based growth and high-precision polishing."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi

N = 26
DEADLINE = 143.0  # safety margin before 150s kill


def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _structured_initial(n=N, seed=0):
    """Create a structured initial configuration based on hexagonal packing."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            cx += 0.01 * rng.randn()
            cy += 0.01 * rng.randn()
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    
    return circles


def _voronoi_initial(seed):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    try:
        vor = Voronoi(points)
    except Exception:
        return _structured_initial(N, seed=seed)
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
        radii[i] = max(radii[i], 0.01)
    
    circles = np.column_stack([points, radii])
    # Small jitter
    circles[:, :2] += rng.uniform(-1e-5, 1e-5, size=(N, 2))
    return circles


def _hex_initial(seed):
    """Hexagonal grid initialization with selection of best N points."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = (i + 0.5) / 7.0
            y = (j + 0.5) / 7.0
            if i % 2 == 1:
                y += 0.25 / 7.0
            points.append((x, y))
    
    points = np.array(points)
    # Select N closest to center
    dists_center = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists_center)[:N]
    p = points[indices]
    
    # Compute initial radii based on nearest neighbor
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(p - p[i], axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        wall_dist = np.min([p[i][0], 1-p[i][0], p[i][1], 1-p[i][1]])
        radii[i] = 0.5 * min(min_dist, wall_dist) * 0.92
    
    circles = np.column_stack([p, radii])
    circles[:, :2] += rng.uniform(-1e-4, 1e-4, size=(N, 2))
    return circles


def _resolve_overlaps(circles, max_iter=500):
    """Iteratively resolve overlaps by pushing circles apart."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iter):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i_idx] + r[j_idx] - dist
        
        if not np.any(overlap > 1e-9):
            break
        
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-12:
                    dx_k = 1e-6
                    dy_k = 0
                else:
                    dx_k = dx[k] / dist[k]
                    dy_k = dy[k] / dist[k]
                
                push = overlap[k] * 0.5
                x[i] -= push * dx_k
                y[i] -= push * dy_k
                x[j] += push * dx_k
                y[j] += push * dy_k
                
                shrink = overlap[k] * 0.1
                r[i] = max(r[i] - shrink, 0.001)
                r[j] = max(r[j] - shrink, 0.001)
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
    
    return np.column_stack([x, y, r])


def _get_delaunay_pairs(cx, cy, i_idx, j_idx):
    """Get Delaunay triangulation edges."""
    points = np.column_stack([cx, cy])
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        if not edges:
            return i_idx, j_idx
        edge_list = sorted(edges)
        return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
    except Exception:
        return i_idx, j_idx


def _interior_point_solve(circles, time_limit=25, seed=0):
    """Primal-dual interior-point method with dynamic Delaunay contact graph.
    
    Formulates the problem as:
    maximize sum(r_i)
    subject to:
      x_i - r_i >= 0, 1 - x_i - r_i >= 0  (boundary)
      y_i - r_i >= 0, 1 - y_i - r_i >= 0  (boundary)
      ||p_i - p_j|| >= r_i + r_j           (non-overlap, for contact pairs)
      r_i > 0
    
    Uses log-barrier for inequality constraints and Newton iterations.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Interior-point parameters
    mu = 0.1  # barrier parameter
    mu_decay = 0.95
    max_mu_iter = 50
    
    # Slack variables for constraints
    # Boundary slacks: s_xi, s_xj, s_yi, s_yj for each circle (4 per circle)
    # Pair slacks: s_k for each contact pair k
    
    def _compute_slacks(cx, cy, cr, d_i, d_j):
        """Compute slack values for all constraints."""
        # Boundary slacks (4 per circle)
        s_xi = cx - cr          # x_i - r_i >= 0
        s_xj = 1 - cx - cr      # 1 - x_i - r_i >= 0
        s_yi = cy - cr          # y_i - r_i >= 0
        s_yj = 1 - cy - cr      # 1 - y_i - r_i >= 0
        
        # Pair slacks (for contact pairs)
        dx = cx[d_i] - cx[d_j]
        dy = cy[d_i] - cy[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        s_pair = dist - cr[d_i] - cr[d_j]
        
        return s_xi, s_xj, s_yi, s_yj, s_pair, dist, dx, dy
    
    def _newton_step(cx, cy, cr, d_i, d_j, mu):
        """Compute one Newton step for the KKT system."""
        n_pairs = len(d_i)
        
        # Compute slacks
        s_xi, s_xj, s_yi, s_yj, s_pair, dist, dx, dy = _compute_slacks(cx, cy, cr, d_i, d_j)
        
        # Ensure positive slacks (project if needed)
        s_xi = np.maximum(s_xi, 1e-8)
        s_xj = np.maximum(s_xj, 1e-8)
        s_yi = np.maximum(s_yi, 1e-8)
        s_yj = np.maximum(s_yj, 1e-8)
        s_pair = np.maximum(s_pair, 1e-8)
        
        # Log barrier objective: -sum(r) - mu * sum(log(slacks))
        # Gradient and Hessian w.r.t. (x, y, r)
        
        # d(-sum r)/dr = -1
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.ones(n)  # -(-1) = +1 in minimization of -sum(r) + barrier
        
        # Barrier gradient for boundary slacks
        # d(-mu*log(s_xi))/dx_i = -mu/s_xi * 1 = -mu/s_xi
        # d(-mu*log(s_xi))/dr_i = -mu/s_xi * (-1) = mu/s_xi
        grad_x -= mu / s_xi
        grad_x += mu / s_xj
        grad_y -= mu / s_yi
        grad_y += mu / s_yj
        grad_r += mu / s_xi + mu / s_xj + mu / s_yi + mu / s_yj
        
        # Barrier gradient for pair slacks
        # s_k = dist_k - r_i - r_j
        # d(-mu*log(s_k))/dx_i = -mu/s_k * (dx_k/dist_k)
        for k in range(n_pairs):
            i = d_i[k]
            j = d_j[k]
            if dist[k] > 1e-12:
                ux = dx[k] / dist[k]
                uy = dy[k] / dist[k]
                grad_x[i] -= mu / s_pair[k] * ux
                grad_x[j] += mu / s_pair[k] * ux
                grad_y[i] -= mu / s_pair[k] * uy
                grad_y[j] += mu / s_pair[k] * uy
            grad_r[i] += mu / s_pair[k]
            grad_r[j] += mu / s_pair[k]
        
        # Hessian (approximate - diagonal dominance for stability)
        # H_xx = mu/s_xi^2 + mu/s_xj^2 + sum_k (mu/s_k^2)*(ux_k^2)
        # Simplified: use diagonal Hessian for speed
        H_xx = mu / s_xi**2 + mu / s_xj**2
        H_yy = mu / s_yi**2 + mu / s_yj**2
        H_rr = mu / s_xi**2 + mu / s_xj**2 + mu / s_yi**2 + mu / s_yj**2
        
        for k in range(n_pairs):
            i = d_i[k]
            j = d_j[k]
            if dist[k] > 1e-12:
                ux = dx[k] / dist[k]
                uy = dy[k] / dist[k]
                H_xx[i] += mu / s_pair[k]**2 * ux**2
                H_xx[j] += mu / s_pair[k]**2 * ux**2
                H_yy[i] += mu / s_pair[k]**2 * uy**2
                H_yy[j] += mu / s_pair[k]**2 * uy**2
            H_rr[i] += mu / s_pair[k]**2
            H_rr[j] += mu / s_pair[k]**2
        
        # Newton step: solve H * delta = -grad
        # Use diagonal approximation for speed
        step_x = -grad_x / np.maximum(H_xx, 1e-6)
        step_y = -grad_y / np.maximum(H_yy, 1e-6)
        step_r = -grad_r / np.maximum(H_rr, 1e-6)
        
        # Limit step size
        max_step = 0.05
        step_x = np.clip(step_x, -max_step, max_step)
        step_y = np.clip(step_y, -max_step, max_step)
        step_r = np.clip(step_r, -max_step, max_step)
        
        return step_x, step_y, step_r
    
    it = 0
    no_improve = 0
    max_no_improve = 300
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Update contact graph periodically
        if it % 10 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_idx, j_idx)
        
        # Newton iteration
        for newton_it in range(5):
            step_x, step_y, step_r = _newton_step(x, y, r, d_i, d_j, mu)
            
            # Line search
            alpha = 1.0
            for ls in range(10):
                trial_x = x + alpha * step_x
                trial_y = y + alpha * step_y
                trial_r = r + alpha * step_r
                
                # Check feasibility
                if np.any(trial_r < 1e-6) or np.any(trial_x < 0) or np.any(trial_x > 1) or \
                   np.any(trial_y < 0) or np.any(trial_y > 1):
                    alpha *= 0.5
                    continue
                
                # Check pair constraints
                tdx = trial_x[d_i] - trial_x[d_j]
                tdy = trial_y[d_i] - trial_y[d_j]
                tdist = np.sqrt(tdx**2 + tdy**2)
                if np.any(tdist < trial_r[d_i] + trial_r[d_j] - 1e-6):
                    alpha *= 0.5
                    continue
                
                break
            
            x += alpha * step_x
            y += alpha * step_y
            r += alpha * step_r
            
            # Clamp
            x = np.clip(x, 1e-4, 1 - 1e-4)
            y = np.clip(y, 1e-4, 1 - 1e-4)
            r = np.maximum(r, 1e-4)
            max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            r = np.minimum(r, np.maximum(max_r, 1e-4))
        
        # Decay barrier parameter
        mu *= mu_decay
        
        it += 1
        
        # Check improvement
        if it % 5 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _physics_grow(circles, time_limit=20, seed=0):
    """Physics-based force optimization with progressive radius growth."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    lr = 0.008
    decay = 0.99999
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    beta = 0.90
    
    base_growth = 0.0003
    growth_ramp = 0.0004
    
    it = 0
    no_improve = 0
    max_no_improve = 500
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_idx, j_idx)
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > 0
        if np.any(mask):
            rep = 200.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 150.0
            gr[d_j[mask]] += 150.0
        
        gx += np.where(x < r, 350.0, 0.0) - np.where(1 - x < r, 350.0, 0.0)
        gy += np.where(y < r, 350.0, 0.0) - np.where(1 - y < r, 350.0, 0.0)
        
        progress = it / max(1, int(time_limit * 50))
        growth_pressure = base_growth + growth_ramp * progress
        gr -= growth_pressure * 100.0
        
        momentum_x = beta * momentum_x + (1 - beta) * gx
        momentum_y = beta * momentum_y + (1 - beta) * gy
        momentum_r = beta * momentum_r + (1 - beta) * gr
        
        x += lr * momentum_x
        y += lr * momentum_y
        r += lr * momentum_r
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _high_precision_polish(circles, time_limit=20, seed=0):
    """High-precision polish using Delaunay triangulation for contact pairs."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    lr = 0.003
    decay = 0.99999
    no_improve_count = 0
    max_no_improve = 600
    
    for it in range(25000):
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_idx, j_idx)
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > 0
        if np.any(mask):
            rep = 150.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 120.0
            gr[d_j[mask]] += 120.0
        
        gx += np.where(x < r, 300.0, 0.0) - np.where(1 - x < r, 300.0, 0.0)
        gy += np.where(y < r, 300.0, 0.0) - np.where(1 - y < r, 300.0, 0.0)
        
        gr -= 2.0
        
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        
        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score + 1e-14:
                no_improve_count = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _refine_radii(circles, time_limit=10):
    """Try to grow radii while maintaining feasibility."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    for it in range(8000):
        if time.time() - start_time > time_limit:
            break
        
        max_allowed = np.copy(r)
        boundary_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        max_allowed = np.minimum(max_allowed, boundary_max)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            max_allowed[i] = min(max_allowed[i], max(dist - r[j], 0.001))
            max_allowed[j] = min(max_allowed[j], max(dist - r[i], 0.001))
        
        headroom = max_allowed - r
        if np.max(headroom) < 1e-7:
            break
        
        n_grow = min(4, n)
        top_indices = np.argsort(headroom)[-n_grow:]
        grow_amount = 0.0015
        for idx in top_indices:
            r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
        
        new_circles = np.column_stack([x, y, r])
        new_circles = _resolve_overlaps(new_circles, max_iter=150)
        
        score = _validate_and_score(new_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = new_circles.copy()
            x = new_circles[:, 0].copy()
            y = new_circles[:, 1].copy()
            r = new_circles[:, 2].copy()
    
    return best_circles, best_score


def _shake_and_refine(p, steps=5000, time_limit=15.0, seed=42):
    """Shake and refine: perturb then re-solve to escape local minima."""
    current_score = _validate_and_score(p)
    if current_score is None:
        current_score = 0.0
    best_p = p.copy()
    best_score = current_score
    rng = np.random.RandomState(seed)
    
    start = time.time()
    sigma = 0.015
    
    for _ in range(steps):
        if time.time() - start > time_limit:
            break
        
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] *= (1 + rng.normal(0, 0.03, size=N))
        p_new[:, 2] = np.clip(p_new[:, 2], 0.005, 0.5)
        
        p_new = _resolve_overlaps(p_new, max_iter=200)
        
        score = _validate_and_score(p_new)
        if score is not None and score > best_score:
            best_score = score
            best_p = p_new.copy()
        
        # Reduce perturbation over time
        sigma *= 0.999
    
    return best_p, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = DEADLINE
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        ('struct1', _structured_initial(N, seed=1)),
        ('struct2', _structured_initial(N, seed=7)),
        ('voronoi1', _voronoi_initial(seed=10)),
        ('hex1', _hex_initial(seed=3)),
        ('struct3', _structured_initial(N, seed=13)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 40:
            break
        
        # Resolve initial overlaps
        resolved = _resolve_overlaps(init_circles, max_iter=300)
        
        # Interior-point solve
        remaining = time_budget - (time.time() - start_time) - 30
        if remaining > 15:
            ip_circles, ip_score = _interior_point_solve(resolved, time_limit=min(remaining * 0.35, 20), seed=hash(name) % 10000)
            if ip_score > best_score:
                best_score = ip_score
                best_circles = ip_circles
        
        # Physics-based growth
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining > 12:
            loc_circles, loc_score = _physics_grow(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.30, 15), seed=hash(name) % 10000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
    
    # Final polish phase
    if best_circles is not None:
        # Shake and refine to escape local minima
        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 12:
            shake_circles, shake_score = _shake_and_refine(best_circles, time_limit=min(remaining * 0.40, 15), seed=7777)
            if shake_score > best_score:
                best_score = shake_score
                best_circles = shake_circles
        
        # High-precision polish
        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 10:
            hp_circles, hp_score = _high_precision_polish(best_circles, time_limit=min(remaining * 0.45, 15), seed=9999)
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
        
        # Radius refinement
        remaining = time_budget - (time.time() - start_time) - 10
        if remaining > 8:
            ref_circles, ref_score = _refine_radii(best_circles, time_limit=min(remaining * 0.35, 10))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles
        
        # Another interior-point pass on the refined result
        remaining = time_budget - (time.time() - start_time) - 8
        if remaining > 10:
            ip2_circles, ip2_score = _interior_point_solve(best_circles, time_limit=min(remaining * 0.50, 12), seed=12345)
            if ip2_score > best_score:
                best_score = ip2_score
                best_circles = ip2_circles
        
        # Final high-precision polish
        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 6:
            hp2_circles, hp2_score = _high_precision_polish(best_circles, time_limit=min(remaining * 0.70, 10), seed=11111)
            if hp2_score > best_score:
                best_score = hp2_score
                best_circles = hp2_circles
    
    # Final validation
    if best_circles is None:
        return _structured_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _structured_initial(N)
    
    return best_circles
