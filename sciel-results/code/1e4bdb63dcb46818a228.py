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


def _random_initial(n=N, seed=42):
    """Create a random initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _hex_dense_initial(n=N, seed=0):
    """Denser hexagonal initialization for better starting geometry."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Approximate radius for 26 circles in unit square
    r_start = 0.08
    points = []
    dx = 2 * r_start * 1.1
    dy = dx * np.sqrt(3) / 2
    
    x_offset = 0.5 - (4 * dx) / 2.0
    y_offset = 0.5 - (5 * dy) / 2.0
    
    for row in range(6):
        for col in range(5):
            cx = x_offset + col * dx + (dx / 2 if row % 2 else 0)
            cy = y_offset + row * dy
            cx = np.clip(cx, r_start, 1 - r_start)
            cy = np.clip(cy, r_start, 1 - r_start)
            points.append([cx, cy, r_start])
            
    points = np.array(points[:n])
    points[:, 0] += 0.005 * rng.randn(n)
    points[:, 1] += 0.005 * rng.randn(n)
    points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
    points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    return points


def _voronoi_initial(n=N, seed=0):
    """Voronoi-based initialization: place centers on a hex grid, then set radii
    to half the minimum distance to neighbors and walls. This gives a feasible
    starting point with radii close to local upper bounds."""
    rng = np.random.RandomState(seed)
    
    # Generate hex grid points, sorted by distance from center
    r0 = 0.13
    centers = []
    for row in range(12):
        for col in range(12):
            x = col * 2 * r0
            y = row * np.sqrt(3) * r0
            if row % 2 == 1:
                x += r0
            centers.append((x, y))
    
    centers.sort(key=lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2)
    centers = centers[:n]
    
    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    width = max_x - min_x
    height = max_y - min_y
    scale = min(0.85 / width, 0.85 / height)
    
    sx = (xs - min_x) * scale + 0.075
    sy = (ys - min_y) * scale + 0.075
    
    # Add small noise
    sx += 0.003 * rng.randn(n)
    sy += 0.003 * rng.randn(n)
    sx = np.clip(sx, 0.01, 0.99)
    sy = np.clip(sy, 0.01, 0.99)
    
    # Compute radii: half min distance to neighbors, also bounded by wall distance
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = sx[i_idx] - sx[j_idx]
    dy = sy[i_idx] - sy[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    
    radii = np.full(n, 0.5)
    for k in range(len(i_idx)):
        d = dist[k] / 2
        radii[i_idx[k]] = min(radii[i_idx[k]], d)
        radii[j_idx[k]] = min(radii[j_idx[k]], d)
    
    wall_dist = np.minimum(np.minimum(sx, 1 - sx), np.minimum(sy, 1 - sy))
    radii = np.minimum(radii, wall_dist)
    radii *= 0.95  # Small margin
    radii = np.maximum(radii, 0.005)
    
    return np.column_stack([sx, sy, radii])


def _resolve_overlaps(circles, max_iter=300):
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
                    dx_k, dy_k = 1e-6, 0
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


def _softplus(z, eps):
    """Smooth softplus: eps * log(1 + exp(z/eps)), clipped for stability."""
    zc = np.clip(z / eps, -50, 50)
    return eps * np.log1p(np.exp(zc))


def _softplus_grad(z, eps):
    """Derivative of softplus: sigmoid(z/eps)."""
    zc = np.clip(z / eps, -50, 50)
    return 1.0 / (1.0 + np.exp(-zc))


def _lbfgs_optimize(circles, time_limit=15, penalty=1000.0, eps=1e-4, seed=0):
    """L-BFGS-B with smooth softplus-squared penalty and analytic gradient.
    
    Uses the v = [x_1..x_N, y_1..y_N, r_1..r_N] parameterization with
    explicit analytic gradients to reduce numerical noise in line searches.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    v = np.concatenate([x, y, r])
    
    def objective(vv):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = vv[:n]
        yy = vv[n:2*n]
        rr = np.maximum(vv[2*n:3*n], 1e-6)
        
        score = -np.sum(rr)
        
        # Boundary violations (squared softplus)
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        wall_pen = (np.sum(_softplus(lx, eps)**2) + np.sum(_softplus(ux, eps)**2) +
                    np.sum(_softplus(ly, eps)**2) + np.sum(_softplus(uy, eps)**2))
        
        # Overlap violations
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (rr[i_idx] + rr[j_idx]) - dist
        pair_pen = np.sum(_softplus(overlap, eps)**2)
        
        return score + penalty * (wall_pen + pair_pen)
    
    def gradient(vv):
        if time.time() - start_time > time_limit:
            return np.zeros_like(vv)
        xx = vv[:n]
        yy = vv[n:2*n]
        rr = np.maximum(vv[2*n:3*n], 1e-6)
        
        grad = np.zeros_like(vv)
        grad[2*n:3*n] -= 1.0  # d/d r of -sum(r)
        
        def d_sp_sq(z):
            """d/dz [softplus(z,eps)^2] = 2 * softplus(z,eps) * sigmoid(z/eps)"""
            return 2.0 * _softplus(z, eps) * _softplus_grad(z, eps)
        
        # Boundary gradients
        for i in range(n):
            z = rr[i] - xx[i]
            dz = d_sp_sq(z)
            grad[i] += -penalty * dz
            grad[2*n + i] += penalty * dz
            
            z = xx[i] + rr[i] - 1.0
            dz = d_sp_sq(z)
            grad[i] += penalty * dz
            grad[2*n + i] += penalty * dz
            
            z = rr[i] - yy[i]
            dz = d_sp_sq(z)
            grad[n + i] += -penalty * dz
            grad[2*n + i] += penalty * dz
            
            z = yy[i] + rr[i] - 1.0
            dz = d_sp_sq(z)
            grad[n + i] += penalty * dz
            grad[2*n + i] += penalty * dz
        
        # Pairwise gradients
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (rr[i_idx] + rr[j_idx]) - dist
        dz = d_sp_sq(overlap)
        ux = dx / dist
        uy = dy / dist
        
        np.add.at(grad[:n], i_idx, -penalty * dz * ux)
        np.add.at(grad[:n], j_idx, penalty * dz * ux)
        np.add.at(grad[n:2*n], i_idx, -penalty * dz * uy)
        np.add.at(grad[n:2*n], j_idx, penalty * dz * uy)
        np.add.at(grad[2*n:3*n], i_idx, penalty * dz)
        np.add.at(grad[2*n:3*n], j_idx, penalty * dz)
        
        return grad
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, v, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 2000, 'maxfun': 8000,
                                                   'ftol': 1e-16, 'gtol': 1e-14,
                                                   'maxls': 100})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_v = result.x
    new_x = np.clip(new_v[:n], 0, 1)
    new_y = np.clip(new_v[n:2*n], 0, 1)
    new_r = np.maximum(new_v[2*n:3*n], 0.001)
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


def _force_optimize(circles, time_limit=20, seed=0):
    """Physics-based force optimization with adaptive step size."""
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
    beta = 0.85
    
    def _get_near_pairs(cx, cy):
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
    
    it = 0
    no_improve = 0
    max_no_improve = 800
    growth_pressure = 3.0
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        if it % 25 == 0:
            d_i, d_j = _get_near_pairs(x, y)
        
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
            rep = 200.0 * overlap[mask] / dist[mask] + 50.0 * overlap[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] -= 80.0 * np.minimum(overlap[mask], 0.05)
            gr[d_j[mask]] -= 80.0 * np.minimum(overlap[mask], 0.05)
        
        gx += np.where(x < r, 400.0, 0.0) - np.where(1 - x < r, 400.0, 0.0)
        gy += np.where(y < r, 400.0, 0.0) - np.where(1 - y < r, 400.0, 0.0)
        bx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
        by = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        gr -= 50.0 * (bx + by)
        
        current_score = _validate_and_score(np.column_stack([x, y, r]))
        if current_score is not None:
            growth_pressure = 2.0 + 1.0 * (current_score / max(best_score, 1e-6))
        gr -= growth_pressure
        
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
            if score is not None and score > best_score + 1e-10:
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


def _simulated_annealing(circles, time_limit=30, seed=0):
    """Simulated annealing with penalty-based scoring."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty_score(cx, cy, cr, pw=80.0):
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
    
    t_init = 0.06
    t_min = 0.0005
    max_iter = 60000
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
        
        if move_type < 0.40:
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.70:
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        elif move_type < 0.85:
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
        else:
            j = rng.randint(n)
            if j != i:
                delta = 0.005 * rng.randn()
                new_r[i] += delta
                new_r[j] -= delta
                new_r[i] = max(0.005, new_r[i])
                new_r[j] = max(0.005, new_r[j])
        
        new_ps = _penalty_score(new_x, new_y, new_r)
        delta = new_ps - current_ps
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_ps = new_ps
            
            if it % 40 == 0:
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
    
    def _get_delaunay_pairs(cx, cy):
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
    
    lr = 0.003
    decay = 0.99997
    no_improve_count = 0
    max_no_improve = 1000
    
    for it in range(30000):
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y)
        
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
            rep = 200.0 * overlap[mask] / dist[mask] + 100.0 * overlap[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] -= 100.0 * np.minimum(overlap[mask], 0.03)
            gr[d_j[mask]] -= 100.0 * np.minimum(overlap[mask], 0.03)
        
        gx += np.where(x < r, 500.0, 0.0) - np.where(1 - x < r, 500.0, 0.0)
        gy += np.where(y < r, 500.0, 0.0) - np.where(1 - y < r, 500.0, 0.0)
        bx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
        by = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        gr -= 80.0 * (bx + by)
        
        current_score = _validate_and_score(np.column_stack([x, y, r]))
        if current_score is not None:
            gr -= 2.5 + 0.5 * (current_score / max(best_score, 1e-6))
        else:
            gr -= 1.0
        
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
            if score is not None and score > best_score + 1e-10:
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


def _high_precision_polish(circles, time_limit=20):
    """Final high-precision polish with very tight tolerances and small eps."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    v = np.concatenate([x, y, r])
    
    eps = 5e-5  # Very small temperature for sharp penalty
    penalty = 2000.0
    
    def objective(vv):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = vv[:n]
        yy = vv[n:2*n]
        rr = np.maximum(vv[2*n:3*n], 1e-6)
        
        score = -np.sum(rr)
        
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        wall_pen = (np.sum(_softplus(lx, eps)**2) + np.sum(_softplus(ux, eps)**2) +
                    np.sum(_softplus(ly, eps)**2) + np.sum(_softplus(uy, eps)**2))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (rr[i_idx] + rr[j_idx]) - dist
        pair_pen = np.sum(_softplus(overlap, eps)**2)
        
        return score + penalty * (wall_pen + pair_pen)
    
    def gradient(vv):
        if time.time() - start_time > time_limit:
            return np.zeros_like(vv)
        xx = vv[:n]
        yy = vv[n:2*n]
        rr = np.maximum(vv[2*n:3*n], 1e-6)
        
        grad = np.zeros_like(vv)
        grad[2*n:3*n] -= 1.0
        
        def d_sp_sq(z):
            return 2.0 * _softplus(z, eps) * _softplus_grad(z, eps)
        
        for i in range(n):
            z = rr[i] - xx[i]
            dz = d_sp_sq(z)
            grad[i] += -penalty * dz
            grad[2*n + i] += penalty * dz
            
            z = xx[i] + rr[i] - 1.0
            dz = d_sp_sq(z)
            grad[i] += penalty * dz
            grad[2*n + i] += penalty * dz
            
            z = rr[i] - yy[i]
            dz = d_sp_sq(z)
            grad[n + i] += -penalty * dz
            grad[2*n + i] += penalty * dz
            
            z = yy[i] + rr[i] - 1.0
            dz = d_sp_sq(z)
            grad[n + i] += penalty * dz
            grad[2*n + i] += penalty * dz
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (rr[i_idx] + rr[j_idx]) - dist
        dz = d_sp_sq(overlap)
        ux = dx / dist
        uy = dy / dist
        
        np.add.at(grad[:n], i_idx, -penalty * dz * ux)
        np.add.at(grad[:n], j_idx, penalty * dz * ux)
        np.add.at(grad[n:2*n], i_idx, -penalty * dz * uy)
        np.add.at(grad[n:2*n], j_idx, penalty * dz * uy)
        np.add.at(grad[2*n:3*n], i_idx, penalty * dz)
        np.add.at(grad[2*n:3*n], j_idx, penalty * dz)
        
        return grad
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, v, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 3000, 'maxfun': 12000,
                                                   'ftol': 1e-18, 'gtol': 1e-16,
                                                   'maxls': 200})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_v = result.x
    new_x = np.clip(new_v[:n], 0, 1)
    new_y = np.clip(new_v[n:2*n], 0, 1)
    new_r = np.maximum(new_v[2*n:3*n], 0.001)
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
    
    for it in range(5000):
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
        
        n_grow = min(3, n)
        top_indices = np.argsort(headroom)[-n_grow:]
        grow_amount = 0.002
        for idx in top_indices:
            r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
        
        new_circles = np.column_stack([x, y, r])
        new_circles = _resolve_overlaps(new_circles, max_iter=100)
        
        score = _validate_and_score(new_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = new_circles.copy()
            x = new_circles[:, 0].copy()
            y = new_circles[:, 1].copy()
            r = new_circles[:, 2].copy()
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 115  # Safety margin before 120s limit
    
    best_circles = None
    best_score = 0.0
    
    # Multiple diverse initializations
    initializations = [
        ('voronoi1', _voronoi_initial(N, seed=1)),
        ('voronoi2', _voronoi_initial(N, seed=7)),
        ('hex_dense1', _hex_dense_initial(N, seed=1)),
        ('hex_dense2', _hex_dense_initial(N, seed=5)),
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=15)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 35:
            break
        
        # Resolve initial overlaps
        resolved = _resolve_overlaps(init_circles, max_iter=200)
        
        # Force optimization
        remaining = time_budget - (time.time() - start_time) - 30
        if remaining > 15:
            loc_circles, loc_score = _force_optimize(
                resolved, time_limit=min(remaining * 0.35, 18), seed=hash(name) % 10000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
        
        # L-BFGS with analytic gradient (primary optimizer)
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining > 12:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.30, 15), penalty=1000.0, eps=1e-4)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles
        
        # Simulated annealing
        remaining = time_budget - (time.time() - start_time) - 18
        if remaining > 15:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.40, 25), seed=hash(name) % 10000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    # Final polish phase
    if best_circles is not None:
        # Delaunay polish
        remaining = time_budget - (time.time() - start_time) - 22
        if remaining > 12:
            del_circles, del_score = _delaunay_polish(
                best_circles, time_limit=min(remaining * 0.5, 18), seed=9999)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles
        
        # Second L-BFGS pass with different penalty/eps
        remaining = time_budget - (time.time() - start_time) - 16
        if remaining > 10:
            lbfgs2_circles, lbfgs2_score = _lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.4, 12),
                penalty=500.0, eps=5e-4)
            if lbfgs2_score > best_score:
                best_score = lbfgs2_score
                best_circles = lbfgs2_circles
        
        # Radius refinement
        remaining = time_budget - (time.time() - start_time) - 12
        if remaining > 8:
            ref_circles, ref_score = _refine_radii(
                best_circles, time_limit=min(remaining * 0.4, 10))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles
        
        # Second force optimization
        remaining = time_budget - (time.time() - start_time) - 10
        if remaining > 8:
            loc2_circles, loc2_score = _force_optimize(
                best_circles, time_limit=min(remaining * 0.5, 12), seed=12345)
            if loc2_score > best_score:
                best_score = loc2_score
                best_circles = loc2_circles
        
        # Final Delaunay polish
        remaining = time_budget - (time.time() - start_time) - 6
        if remaining > 6:
            del2_circles, del2_score = _delaunay_polish(
                best_circles, time_limit=min(remaining * 0.5, 8), seed=11111)
            if del2_score > best_score:
                best_score = del2_score
                best_circles = del2_circles
        
        # High-precision final polish
        remaining = time_budget - (time.time() - start_time) - 3
        if remaining > 8:
            hp_circles, hp_score = _high_precision_polish(
                best_circles, time_limit=min(remaining * 0.8, 18))
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
    
    # Final validation
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
