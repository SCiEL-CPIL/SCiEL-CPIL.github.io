"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

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


def _hex_initial(n=N, seed=0):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Use a hexagonal pattern
    r_est = 0.08
    dx = 2 * r_est * 1.05
    dy = dx * np.sqrt(3) / 2
    
    points = []
    for row in range(7):
        for col in range(5):
            cx = 0.5 + (col - 2) * dx + (dx / 2 if row % 2 else 0)
            cy = 0.5 + (row - 3) * dy
            cx = np.clip(cx, r_est, 1 - r_est)
            cy = np.clip(cy, r_est, 1 - r_est)
            points.append([cx, cy, r_est])
    
    points = np.array(points[:n])
    points[:, 0] += 0.003 * rng.randn(n)
    points[:, 1] += 0.003 * rng.randn(n)
    points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
    points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    return points


def _corner_weighted_initial(n=N, seed=0):
    """Place circles in corners and along edges for better boundary usage."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Start with corner positions
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    edge_mid = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    inner = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
             (0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
             (0.35, 0.65), (0.65, 0.35), (0.65, 0.65), (0.35, 0.35),
             (0.2, 0.4), (0.8, 0.6), (0.4, 0.8), (0.6, 0.2),
             (0.45, 0.55), (0.55, 0.45), (0.5, 0.35), (0.5, 0.65),
             (0.35, 0.5), (0.65, 0.5), (0.5, 0.45), (0.45, 0.5),
             (0.6, 0.5), (0.5, 0.6)]
    
    positions = corners + edge_mid + inner
    positions = positions[:n]
    
    for i in range(n):
        circles[i, 0] = positions[i][0] + 0.01 * rng.randn()
        circles[i, 1] = positions[i][1] + 0.01 * rng.randn()
        circles[i, 0] = np.clip(circles[i, 0], 0.05, 0.95)
        circles[i, 1] = np.clip(circles[i, 1], 0.05, 0.95)
        circles[i, 2] = 0.06
    
    return circles


def _random_initial(n=N, seed=42):
    """Create a random initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


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
    """Get Delaunay triangulation edges for contact pairs."""
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


def _force_optimize(circles, time_limit=20, seed=0):
    """Physics-based force optimization with adaptive step size and momentum."""
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
    
    lr = 0.006
    decay = 0.99998
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    beta = 0.80
    
    it = 0
    no_improve = 0
    max_no_improve = 400
    d_i, d_j = i_idx, j_idx
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        if it % 25 == 0:
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
        
        # Boundary forces
        gx += np.where(x < r, 400.0, 0.0) - np.where(1 - x < r, 400.0, 0.0)
        gy += np.where(y < r, 400.0, 0.0) - np.where(1 - y < r, 400.0, 0.0)
        
        # Radius growth pressure
        gr -= 3.0
        
        # Momentum update
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
        
        if it % 8 == 0:
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


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """L-BFGS-B with smooth softplus penalty."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    beta_sp = 300.0
    
    def _softplus(z):
        return np.where(z > 20, z, np.log1p(np.exp(np.clip(z, -50, 50))))
    
    def _sigmoid(z):
        return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        bv = (_softplus(beta_sp * (rr - xx)) + _softplus(beta_sp * (rr - (1 - xx))) +
              _softplus(beta_sp * (rr - yy)) + _softplus(beta_sp * (rr - (1 - yy)))) / beta_sp
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(beta_sp * (rr[i_idx] + rr[j_idx] - dist)) / beta_sp
        
        return -np.sum(rr) + 100.0 * (np.sum(bv) + np.sum(ov))
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        sx_left = _sigmoid(beta_sp * (rr - xx))
        sx_right = _sigmoid(beta_sp * (rr - (1 - xx)))
        sy_top = _sigmoid(beta_sp * (rr - yy))
        sy_bot = _sigmoid(beta_sp * (rr - (1 - yy)))
        
        gx -= 100.0 * sx_left - 100.0 * sx_right
        gy -= 100.0 * sy_top - 100.0 * sy_bot
        gr += 100.0 * (sx_left + sx_right + sy_top + sy_bot)
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        sig_ov = _sigmoid(beta_sp * overlap)
        
        gx[i_idx] -= 100.0 * sig_ov * dx / dist
        gy[i_idx] -= 100.0 * sig_ov * dy / dist
        gx[j_idx] += 100.0 * sig_ov * dx / dist
        gy[j_idx] += 100.0 * sig_ov * dy / dist
        gr[i_idx] += 100.0 * sig_ov
        gr[j_idx] += 100.0 * sig_ov
        
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 1000, 'maxfun': 5000,
                                                   'ftol': 1e-15, 'gtol': 1e-13})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0, 1)
    new_y = np.clip(new_params[n:2*n], 0, 1)
    new_r = np.maximum(new_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _simulated_annealing(circles, time_limit=30, seed=0):
    """Simulated annealing with penalty-based scoring and diverse moves."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty_score(cx, cy, cr, pw=100.0):
        if np.any(cr <= 0):
            return -1e6
        bv = np.maximum(0, cr - cx) + np.maximum(0, cr - (1 - cx)) + \
              np.maximum(0, cr - cy) + np.maximum(0, cr - (1 - cy))
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0, cr[i_idx] + cr[j_idx] - dist)
        return float(np.sum(cr) - pw * (np.sum(bv) + np.sum(ov)))
    
    def _strict_score(cx, cy, cr):
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or \
           np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))
    
    best_strict_score = _strict_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    if best_strict_score is None:
        best_strict_score = 0.0
    
    current_ps = _penalty_score(x, y, r)
    
    t_init = 0.08
    t_min = 0.0003
    max_iter = 80000
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    t = t_init
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.35:
            # Move position
            scale = 0.01 + 0.02 * t / t_init
            delta = scale * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.60:
            # Change radius
            new_r[i] *= (1 + 0.10 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        elif move_type < 0.75:
            # Swap two radii
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
        elif move_type < 0.88:
            # Nudge pair
            j = rng.randint(n)
            if j != i:
                delta = 0.006 * rng.randn()
                new_r[i] += delta
                new_r[j] -= delta
                new_r[i] = max(0.005, new_r[i])
                new_r[j] = max(0.005, new_r[j])
        else:
            # Move multiple circles
            for _ in range(3):
                k = rng.randint(n)
                new_x[k] += 0.008 * rng.randn()
                new_y[k] += 0.008 * rng.randn()
        
        new_ps = _penalty_score(new_x, new_y, new_r)
        delta = new_ps - current_ps
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_ps = new_ps
            
            if it % 30 == 0:
                s = _strict_score(x, y, r)
                if s is not None and s > best_strict_score:
                    best_strict_score = s
                    best_circles = np.column_stack([x, y, r]).copy()
        
        t *= cooling
    
    s = _strict_score(x, y, r)
    if s is not None and s > best_strict_score:
        best_strict_score = s
        best_circles = np.column_stack([x, y, r]).copy()
    
    return best_circles, best_strict_score


def _slsqp_refine(circles, time_limit=15, seed=0):
    """SLSQP constrained optimization for final refinement."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        return -np.sum(p[2*n:3*n])
    
    def boundary_constraints(p):
        if time.time() - start_time > time_limit:
            return np.ones(n) * 1e-6
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        return np.minimum(np.minimum(xx - rr, 1 - xx - rr), np.minimum(yy - rr, 1 - yy - rr))
    
    def overlap_constraints(p):
        if time.time() - start_time > time_limit:
            return np.ones(len(i_idx)) * 1e-6
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - rr[i_idx] - rr[j_idx]
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, params, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-16, 'disp': False})
    except Exception:
        pass
    
    if time.time() - start_time < time_limit:
        new_params = result.x
        new_x = np.clip(new_params[:n], 0, 1)
        new_y = np.clip(new_params[n:2*n], 0, 1)
        new_r = np.maximum(new_params[2*n:3*n], 0.001)
        max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
        new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
        
        new_circles = np.column_stack([new_x, new_y, new_r])
        new_score = _validate_and_score(new_circles)
        if new_score is not None and new_score > best_score:
            best_score = new_score
            best_circles = new_circles
    
    return best_circles, best_score


def _delaunay_polish(circles, time_limit=15, seed=0):
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
    
    lr = 0.004
    decay = 0.99998
    no_improve_count = 0
    max_no_improve = 400
    d_i, d_j = i_idx, j_idx
    
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
            rep = 200.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 150.0
            gr[d_j[mask]] += 150.0
        
        gx += np.where(x < r, 400.0, 0.0) - np.where(1 - x < r, 400.0, 0.0)
        gy += np.where(y < r, 400.0, 0.0) - np.where(1 - y < r, 400.0, 0.0)
        
        gr -= 3.0
        
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        
        if it % 12 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score + 1e-12:
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
    
    for it in range(4000):
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
        if np.max(headroom) < 1e-6:
            break
        
        n_grow = min(4, n)
        top_indices = np.argsort(headroom)[-n_grow:]
        grow_amount = 0.003
        for idx in top_indices:
            r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
        
        new_circles = np.column_stack([x, y, r])
        new_circles = _resolve_overlaps(new_circles, max_iter=80)
        
        score = _validate_and_score(new_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = new_circles.copy()
            x = new_circles[:, 0].copy()
            y = new_circles[:, 1].copy()
            r = new_circles[:, 2].copy()
    
    return best_circles, best_score


def _shake_and_refine(circles, time_limit=15, seed=0):
    """Perturb and re-optimize to escape local minima."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            break
        
        # Shake positions
        shake_x = x + 0.02 * rng.randn(n)
        shake_y = y + 0.02 * rng.randn(n)
        shake_r = np.maximum(r * (1 + 0.05 * rng.randn(n)), 0.005)
        
        shake_circles = np.column_stack([shake_x, shake_y, shake_r])
        shake_circles = _resolve_overlaps(shake_circles, max_iter=200)
        
        # Quick force optimization
        remaining2 = time_limit - (time.time() - start_time)
        if remaining2 > 3:
            loc_circles, loc_score = _force_optimize(shake_circles, time_limit=min(remaining2 * 0.5, 8), seed=seed + attempt)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 145  # Keep 5s safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        ('hex1', _hex_initial(N, seed=1)),
        ('hex2', _hex_initial(N, seed=3)),
        ('corner1', _corner_weighted_initial(N, seed=1)),
        ('corner2', _corner_weighted_initial(N, seed=5)),
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=10)),
        ('random2', _random_initial(N, seed=25)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 25:
            break
        
        resolved = _resolve_overlaps(init_circles, max_iter=300)
        
        # Force optimization
        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 10:
            loc_circles, loc_score = _force_optimize(resolved, time_limit=min(remaining * 0.25, 12), seed=hash(name) % 10000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
        
        # L-BFGS optimization
        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 10:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.25, 12), seed=hash(name) % 10000)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles
        
        # Simulated annealing
        remaining = time_budget - (time.time() - start_time) - 12
        if remaining > 10:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.35, 18), seed=hash(name) % 10000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    # Final polish phase
    if best_circles is not None:
        # Delaunay polish
        remaining = time_budget - (time.time() - start_time) - 18
        if remaining > 8:
            del_circles, del_score = _delaunay_polish(best_circles, time_limit=min(remaining * 0.40, 10), seed=9999)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles
        
        # Radius refinement
        remaining = time_budget - (time.time() - start_time) - 14
        if remaining > 6:
            ref_circles, ref_score = _refine_radii(best_circles, time_limit=min(remaining * 0.35, 7))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles
        
        # Shake and refine
        remaining = time_budget - (time.time() - start_time) - 12
        if remaining > 8:
            shake_circles, shake_score = _shake_and_refine(best_circles, time_limit=min(remaining * 0.50, 10), seed=55555)
            if shake_score > best_score:
                best_score = shake_score
                best_circles = shake_circles
        
        # Another force optimization
        remaining = time_budget - (time.time() - start_time) - 10
        if remaining > 7:
            loc2_circles, loc2_score = _force_optimize(best_circles, time_limit=min(remaining * 0.40, 8), seed=12345)
            if loc2_score > best_score:
                best_score = loc2_score
                best_circles = loc2_circles
        
        # L-BFGS polish
        remaining = time_budget - (time.time() - start_time) - 8
        if remaining > 6:
            lbfgs2_circles, lbfgs2_score = _lbfgs_optimize(best_circles, time_limit=min(remaining * 0.50, 8), seed=22222)
            if lbfgs2_score > best_score:
                best_score = lbfgs2_score
                best_circles = lbfgs2_circles
        
        # SLSQP high-precision polish
        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 4:
            hp_circles, hp_score = _slsqp_refine(best_circles, time_limit=min(remaining * 0.80, 6), seed=33333)
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
        
        # Final Delaunay polish
        remaining = time_budget - (time.time() - start_time) - 3
        if remaining > 3:
            del2_circles, del2_score = _delaunay_polish(best_circles, time_limit=min(remaining * 0.70, 5), seed=11111)
            if del2_score > best_score:
                best_score = del2_score
                best_circles = del2_circles
    
    # Final validation
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
