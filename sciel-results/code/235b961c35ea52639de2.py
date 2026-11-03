"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


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


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    iu = np.triu_indices(N, k=1)
    for _ in range(300):
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _mma_optimize(xyz, time_budget=10.0):
    """
    Method of Moving Asymptotes (MMA) optimization.
    Uses convex subproblems to handle non-convex constraints robustly.
    """
    start_time = time.time()
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    n = len(r)
    
    # MMA parameters
    max_iter = 200
    tol = 1e-8
    penalty = 100.0
    eps = 1e-6
    
    # Bounds
    x_min, x_max = np.zeros(n), np.ones(n)
    y_min, y_max = np.zeros(n), np.ones(n)
    r_min = np.full(n, 1e-6)
    r_max = np.full(n, 0.5)
    
    # Initial move limits
    dx_move = 0.1 * np.ones(n)
    dy_move = 0.1 * np.ones(n)
    dr_move = 0.05 * np.ones(n)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_xyz = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute objective and gradients
        val = -np.sum(r)
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.full(n, -1.0)
        
        # Boundary constraints (inequality: g_k <= 0)
        # Left x: r - x <= 0
        gx += -penalty * np.maximum(r - x, 0)
        gr += penalty * np.maximum(r - x, 0)
        # Right x: x + r - 1 <= 0
        gx += penalty * np.maximum(x + r - 1, 0)
        gr += penalty * np.maximum(x + r - 1, 0)
        # Left y: r - y <= 0
        gy += -penalty * np.maximum(r - y, 0)
        gr += penalty * np.maximum(r - y, 0)
        # Right y: y + r - 1 <= 0
        gy += penalty * np.maximum(y + r - 1, 0)
        gr += penalty * np.maximum(y + r - 1, 0)
        
        # Pairwise overlap constraints: r_i + r_j - dist <= 0
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, eps)
        overlap = r[i_idx] + r[j_idx] - dist
        mask = overlap > 0
        if np.any(mask):
            rep = penalty * overlap[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += rep
            gr[j_idx[mask]] += rep
        
        # MMA: Solve convex subproblem
        # Minimize: sum_i (r_i) + penalty * sum_k g_k(x)
        # Subject to: x_min <= x <= x_max, etc.
        
        # Use L-BFGS-B for the subproblem with tight bounds
        params = np.column_stack([x, y, r]).flatten()
        
        def sub_obj(p):
            xx = p[0::3]
            yy = p[1::3]
            rr = p[2::3]
            obj = -np.sum(rr)
            # Add penalty for violations
            obj += penalty * (np.sum(np.maximum(rr - xx, 0)**2) + 
                            np.sum(np.maximum(xx + rr - 1, 0)**2) +
                            np.sum(np.maximum(rr - yy, 0)**2) +
                            np.sum(np.maximum(yy + rr - 1, 0)**2))
            ddx = xx[i_idx] - xx[j_idx]
            ddy = yy[i_idx] - yy[j_idx]
            ddist = np.sqrt(ddx * ddx + ddy * ddy)
            ddist = np.maximum(ddist, eps)
            dov = np.maximum(rr[i_idx] + rr[j_idx] - ddist, 0)
            obj += penalty * np.sum(dov**2)
            return obj
        
        def sub_grad(p):
            xx = p[0::3]
            yy = p[1::3]
            rr = p[2::3]
            sgx = np.zeros(n)
            sgy = np.zeros(n)
            sgr = np.full(n, -1.0)
            
            # Boundary gradients
            mask_lx = rr - xx > 0
            sgx[mask_lx] += -2 * penalty * (rr[mask_lx] - xx[mask_lx])
            sgr[mask_lx] += 2 * penalty * (rr[mask_lx] - xx[mask_lx])
            
            mask_ux = xx + rr - 1 > 0
            sgx[mask_ux] += 2 * penalty * (xx[mask_ux] + rr[mask_ux] - 1)
            sgr[mask_ux] += 2 * penalty * (xx[mask_ux] + rr[mask_ux] - 1)
            
            mask_ly = rr - yy > 0
            sgy[mask_ly] += -2 * penalty * (rr[mask_ly] - yy[mask_ly])
            sgr[mask_ly] += 2 * penalty * (rr[mask_ly] - yy[mask_ly])
            
            mask_uy = yy + rr - 1 > 0
            sgy[mask_uy] += 2 * penalty * (yy[mask_uy] + rr[mask_uy] - 1)
            sgr[mask_uy] += 2 * penalty * (yy[mask_uy] + rr[mask_uy] - 1)
            
            # Overlap gradients
            ddx = xx[i_idx] - xx[j_idx]
            ddy = yy[i_idx] - yy[j_idx]
            ddist = np.sqrt(ddx * ddx + ddy * ddy)
            ddist = np.maximum(ddist, eps)
            dov = rr[i_idx] + rr[j_idx] - ddist
            mask_ov = dov > 0
            if np.any(mask_ov):
                fac = 2 * penalty * dov[mask_ov]
                inv_d = 1.0 / ddist[mask_ov]
                np.add.at(sgx, i_idx[mask_ov], -fac * ddx[mask_ov] * inv_d)
                np.add.at(sgx, j_idx[mask_ov], fac * ddx[mask_ov] * inv_d)
                np.add.at(sgy, i_idx[mask_ov], -fac * ddy[mask_ov] * inv_d)
                np.add.at(sgy, j_idx[mask_ov], fac * ddy[mask_ov] * inv_d)
                np.add.at(sgr, i_idx[mask_ov], fac)
                np.add.at(sgr, j_idx[mask_ov], fac)
            
            return np.column_stack([sgx, sgy, sgr]).flatten()
        
        bounds = []
        for i in range(n):
            bounds.append((max(x[i] - dx_move[i], x_min[i]), min(x[i] + dx_move[i], x_max[i])))
            bounds.append((max(y[i] - dy_move[i], y_min[i]), min(y[i] + dy_move[i], y_max[i])))
            bounds.append((max(r[i] - dr_move[i], r_min[i]), min(r[i] + dr_move[i], r_max[i])))
        
        try:
            result = minimize(sub_obj, params, jac=sub_grad, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            new_params = result.x
            x_new = new_params[0::3]
            y_new = new_params[1::3]
            r_new = new_params[2::3]
            
            # Check if we moved significantly
            move_x = np.max(np.abs(x_new - x))
            move_y = np.max(np.abs(y_new - y))
            move_r = np.max(np.abs(r_new - r))
            
            if move_x < tol and move_y < tol and move_r < tol:
                break
            
            # Update move limits (shrink if we hit bounds, expand otherwise)
            dx_move = np.where(np.abs(x_new - x) > 0.9 * dx_move, 0.5 * dx_move, 1.5 * dx_move)
            dy_move = np.where(np.abs(y_new - y) > 0.9 * dy_move, 0.5 * dy_move, 1.5 * dy_move)
            dr_move = np.where(np.abs(r_new - r) > 0.9 * dr_move, 0.5 * dr_move, 1.5 * dr_move)
            
            x = x_new
            y = y_new
            r = r_new
            
            # Validate and track best
            current_score = _validate_and_score(np.column_stack([x, y, r]))
            if current_score is not None and current_score > best_score:
                best_score = current_score
                best_xyz = np.column_stack([x, y, r]).copy()
                
        except Exception:
            break
    
    return best_xyz, best_score


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B with escalating penalty.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14, 
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


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
    global _start_time
    _start_time = time.time()
    time_budget = _TIME_LIMIT
    
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
        elapsed = _elapsed()
        if elapsed > time_budget - 25:
            break
        
        resolved = _resolve_overlaps(init_circles, max_iter=300)
        
        # MMA optimization
        remaining = time_budget - _elapsed() - 20
        if remaining > 10:
            mma_circles, mma_score = _mma_optimize(resolved, time_budget=min(remaining * 0.25, 12))
            if mma_score > best_score:
                best_score = mma_score
                best_circles = mma_circles
        
        # Analytic L-BFGS polish with escalating penalty
        remaining = time_budget - _elapsed() - 15
        if remaining > 10:
            target = best_circles if best_circles is not None else resolved
            polished = _polish_analytic(target, time_budget=min(remaining * 0.30, 10))
            polished = _enforce_validity(polished)
            pol_score = _validate_and_score(polished)
            if pol_score is not None and pol_score > best_score:
                best_score = pol_score
                best_circles = polished
        
        # Simulated annealing
        remaining = time_budget - _elapsed() - 12
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
        remaining = time_budget - _elapsed() - 18
        if remaining > 8:
            del_circles, del_score = _delaunay_polish(best_circles, time_limit=min(remaining * 0.40, 10), seed=9999)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles
        
        # Radius refinement
        remaining = time_budget - _elapsed() - 14
        if remaining > 6:
            ref_circles, ref_score = _refine_radii(best_circles, time_limit=min(remaining * 0.35, 7))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles
        
        # Shake and refine
        remaining = time_budget - _elapsed() - 12
        if remaining > 8:
            shake_circles, shake_score = _shake_and_refine(best_circles, time_limit=min(remaining * 0.50, 10), seed=55555)
            if shake_score > best_score:
                best_score = shake_score
                best_circles = shake_circles
        
        # Another force optimization
        remaining = time_budget - _elapsed() - 10
        if remaining > 7:
            loc2_circles, loc2_score = _force_optimize(best_circles, time_limit=min(remaining * 0.40, 8), seed=12345)
            if loc2_score > best_score:
                best_score = loc2_score
                best_circles = loc2_circles
        
        # Final MMA polish
        remaining = time_budget - _elapsed() - 8
        if remaining > 6:
            mma2_circles, mma2_score = _mma_optimize(best_circles, time_budget=min(remaining * 0.50, 8))
            if mma2_score > best_score:
                best_score = mma2_score
                best_circles = mma2_circles
        
        # Final analytic L-BFGS polish
        remaining = time_budget - _elapsed() - 8
        if remaining > 6:
            polished2 = _polish_analytic(best_circles, time_budget=min(remaining * 0.50, 8))
            polished2 = _enforce_validity(polished2)
            pol2_score = _validate_and_score(polished2)
            if pol2_score is not None and pol2_score > best_score:
                best_score = pol2_score
                best_circles = polished2
        
        # Final Delaunay polish
        remaining = time_budget - _elapsed() - 3
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
