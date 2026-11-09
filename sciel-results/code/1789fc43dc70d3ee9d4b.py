"""Hexagonal-lattice init + vectorized soft-min polish."""
import numpy as np
import time

N = 200


def _score(pts):
    """Exact radius: min over pairs (dist/2) and walls (0.5 - |coord|)."""
    pts = np.asarray(pts, dtype=np.float64)
    # pairwise
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pairs = dist.min() / 2.0
    r_walls = (0.5 - np.abs(pts)).min()
    return min(r_pairs, r_walls)


def _soft_min_radius(pts, beta, t):
    """
    Smooth surrogate objective: we want to maximize t subject to
    soft constraints that pairwise dist >= 2t and wall >= t.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    viol_pairs = 2.0 * t - dist
    wall_margin = 0.5 - np.abs(pts)
    viol_walls = t - wall_margin
    viol = np.concatenate([viol_pairs.ravel(), viol_walls.ravel()])
    b = beta * viol
    bmax = b.max()
    lse = bmax + np.log(np.sum(np.exp(b - bmax)))
    return t - (1.0 / beta) * lse


def _soft_min_radius_grad(pts, beta, t):
    """Gradient of the soft objective w.r.t. pts and t. Vectorized."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    viol_pairs = 2.0 * t - dist
    wall_margin = 0.5 - np.abs(pts)
    viol_walls = t - wall_margin
    
    viol = np.concatenate([viol_pairs.ravel(), viol_walls.ravel()])
    b = beta * viol
    bmax = b.max()
    e = np.exp(b - bmax)
    s = e.sum()
    w = e / s  # softmax weights
    
    grad_pts = np.zeros_like(pts)
    
    # Pairs contribution
    w_pairs = w[:n * (n - 1)].reshape(n, n)
    
    # A[i,j] = w_ij / dist_ij
    with np.errstate(divide='ignore', invalid='ignore'):
        A = w_pairs / dist
    A[np.isnan(A)] = 0.0
    
    row_sums = np.sum(A, axis=1)
    Ap = A @ pts
    
    grad_pts += row_sums[:, None] * pts - Ap
    
    # Walls contribution
    w_walls = w[n * (n - 1):].reshape(n, 2)
    signs = np.sign(pts)
    grad_pts -= w_walls * signs
    
    # grad_t
    sum_w_pairs = np.sum(w[:n*(n-1)])
    sum_w_walls = np.sum(w[n*(n-1):])
    grad_t = 1.0 - 2.0 * sum_w_pairs - 1.0 * sum_w_walls
    
    return grad_pts, grad_t


def _generate_hexagonal_grid():
    """Hexagonal (triangular) lattice points inside [-0.5,0.5]^2, pick N closest to centre."""
    a = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    x0 = -0.5 - a
    x1 = 0.5 + a
    y0 = -0.5 - a
    y1 = 0.5 + a
    nx = int(np.ceil((x1 - x0) / a)) + 1
    dy = a * np.sqrt(3.0) / 2.0
    ny = int(np.ceil((y1 - y0) / dy)) + 1
    pts = []
    for j in range(ny):
        y = y0 + j * dy
        offset = (a / 2.0) if (j % 2) else 0.0
        for i in range(nx):
            x = x0 + i * a + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
    d2 = np.sum(pts * pts, axis=1)
    idx = np.argsort(d2)
    return pts[idx[:N]]


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    np.random.seed(12345)

    # 1) Hexagonal init
    pts = _generate_hexagonal_grid()
    best_pts = pts.copy()
    best_score = _score(pts)

    # 2) Optimization
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    if minimize is not None:
        beta = 500.0
        x0 = np.concatenate([best_pts.ravel(), [best_score * 0.95]])
        
        def neg_obj(x):
            p = x[:2 * N].reshape(N, 2)
            t = x[2 * N]
            t = np.clip(t, 0.0, 0.5)
            return -_soft_min_radius(p, beta, t)
            
        def neg_obj_grad(x):
            p = x[:2 * N].reshape(N, 2)
            t = x[2 * N]
            t = np.clip(t, 0.0, 0.5)
            gp, gt = _soft_min_radius_grad(p, beta, t)
            return np.concatenate([gp.ravel(), [-gt]])
            
        for attempt in range(5):
            if time.time() > deadline:
                break
            try:
                res = minimize(neg_obj, x0, jac=neg_obj_grad, method='L-BFGS-B',
                               options={'maxiter': 300, 'gtol': 1e-10, 'maxls': 50})
                p = res.x[:2 * N].reshape(N, 2)
                p = np.clip(p, -0.5 + 1e-9, 0.5 - 1e-9)
                s = _score(p)
                if s > best_score:
                    best_score, best_pts = s, p.copy()
                x0 = np.concatenate([p.ravel(), [np.clip(res.x[2 * N], 0.0, 0.5)]])
            except Exception:
                break

    return best_pts
