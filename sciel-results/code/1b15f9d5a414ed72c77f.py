"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIP = len(_TRIPLETS)  # 165


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _bary_to_cart(bary):
    """Convert barycentric (u,v) with u>=0,v>=0,u+v<=1 to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian to barycentric (u,v) where w=1-u-v."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])


def _project_bary(bary):
    """Project barycentric coords into valid triangle: u>=0, v>=0, u+v<=1."""
    u = np.clip(bary[:, 0], 0.0, 1.0)
    v = np.clip(bary[:, 1], 0.0, 1.0)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])


def _soft_min_area(areas, beta):
    """Soft minimum via log-sum-exp: -1/beta * log(sum(exp(-beta*a)))."""
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _objective_bary(bary_flat, beta):
    """Objective in barycentric coords: minimize -soft_min_area + boundary penalty."""
    bary = bary_flat.reshape(-1, 2)
    # Boundary penalty: penalize u<0, v<0, u+v>1
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    pen_u = np.sum(np.maximum(0, -u) ** 2)
    pen_v = np.sum(np.maximum(0, -v) ** 2)
    pen_w = np.sum(np.maximum(0, -w) ** 2)
    boundary_pen = 10.0 * (pen_u + pen_v + pen_w)
    
    pts = _bary_to_cart(_project_bary(bary))
    areas = _all_triangle_areas(pts)
    sm = _soft_min_area(areas, beta)
    return -sm + boundary_pen


def _push_apart(pts, n_iters=100, lr=0.005):
    """
    Gradient-based push-apart: directly maximize the minimum triangle area
    using subgradient of the soft-min objective.
    Works in barycentric coordinates to naturally respect constraints.
    """
    bary = _cart_to_bary(pts)
    bary = _project_bary(bary)
    bary = bary.copy()
    
    for it in range(n_iters):
        pts = _bary_to_cart(bary)
        areas = _all_triangle_areas(pts)
        min_a = np.min(areas)
        
        # Find the triplet(s) achieving (near) minimum
        tol = 1e-6 * max(1.0, min_a)
        active = areas <= min_a + tol
        active_idx = np.where(active)[0]
        
        if len(active_idx) == 0:
            active_idx = np.array([np.argmin(areas)])
        
        # For each active triplet, compute gradient of area w.r.t. barycentric coords
        # area = 0.5 * |cross|, cross = (pj-pi) x (pk-pi)
        # We need d(area)/d(bary_i), d(area)/d(bary_j), d(area)/d(bary_k)
        # bary -> cart: x = u + 0.5v, y = v*S3H
        # dx/du=1, dx/dv=0.5, dy/du=0, dy/dv=S3H
        
        grad_bary = np.zeros((N, 2))
        
        for tri_idx in active_idx:
            i, j, k = _TRIPLETS[tri_idx]
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            dx2 = pk[0] - pi[0]
            dy2 = pk[1] - pi[1]
            cross_val = dx1 * dy2 - dy1 * dx2
            
            sign = 1.0 if cross_val >= 0 else -1.0
            
            # Gradient of cross w.r.t. Cartesian coords (same as before)
            # d(cross)/d(px_i) = py_j - py_k
            # d(cross)/d(py_i) = px_k - px_j
            # d(cross)/d(px_j) = py_k - py_i
            # d(cross)/d(py_j) = px_i - px_k
            # d(cross)/d(px_k) = py_i - py_j
            # d(cross)/d(py_k) = px_j - px_i
            
            gc_cart = np.zeros((3, 2))
            gc_cart[0, 0] = pj[1] - pk[1]
            gc_cart[0, 1] = pk[0] - pj[0]
            gc_cart[1, 0] = pk[1] - pi[1]
            gc_cart[1, 1] = pi[0] - pk[0]
            gc_cart[2, 0] = pi[1] - pj[1]
            gc_cart[2, 1] = pj[0] - pi[0]
            
            # d(area)/d(cart) = 0.5 * sign * gc_cart
            # d(area)/d(bary) = d(area)/d(cart) * J where J = [[1, 0.5], [0, S3H]]
            # So d(area)/d(u) = gc[0]*1 + gc[1]*0 = gc[0]
            #    d(area)/d(v) = gc[0]*0.5 + gc[1]*S3H
            
            for pidx, (pi_idx, p_cart) in enumerate(zip([i, j, k], [pi, pj, pk])):
                d_area_du = 0.5 * sign * gc_cart[pidx, 0]
                d_area_dv = 0.5 * sign * (0.5 * gc_cart[pidx, 0] + S3H * gc_cart[pidx, 1])
                grad_bary[pi_idx, 0] += d_area_du
                grad_bary[pi_idx, 1] += d_area_dv
        
        # Normalize gradient (average over active constraints)
        grad_bary /= len(active_idx)
        
        # Update: move in direction of increasing min area
        bary = bary + lr * grad_bary
        bary = _project_bary(bary)
    
    return _bary_to_cart(bary)


def _bfgs_optimize_bary(bary_init, beta_start=20.0, beta_end=100.0, n_iters=500):
    """
    BFGS optimization in barycentric space with annealing beta.
    Returns optimized barycentric coordinates.
    """
    bary = bary_init.copy().flatten()
    best_val = _objective_bary(bary, beta_start)
    best_bary = bary.copy()
    
    # Anneal beta from beta_start to beta_end
    for phase in range(5):
        beta = beta_start * (beta_end / beta_start) ** (phase / 4.0)
        
        def obj(x):
            return _objective_bary(x, beta)
        
        def grad_obj(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xp[i] += eps
                xm = x.copy()
                xm[i] -= eps
                g[i] = (obj(xp) - obj(xm)) / (2 * eps)
            return g
        
        try:
            res = minimize(obj, bary, method='L-BFGS-B', jac=grad_obj,
                           options={'maxiter': 100, 'gtol': 1e-8})
            bary = res.x
            val = _objective_bary(bary, beta)
            if val < best_val:
                best_val = val
                best_bary = bary.copy()
        except Exception:
            pass
    
    return best_bary.reshape(-1, 2)


def _random_bary(rng):
    """Random point in barycentric triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return np.array([u, v])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Symmetric initial configurations ----
    # Configuration 1: corners + edge midpoints + interior
    init1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1.0/3.0, S3H / 3.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, S3H / 3.0],
        [0.5, 2.0 * S3H / 3.0],
    ], dtype=float)
    
    # Configuration 2: more spread out
    init2 = np.array([
        [0.05, 0.02],
        [0.95, 0.02],
        [0.5, S3H - 0.05],
        [0.3, 0.0],
        [0.7, 0.0],
        [0.5, 0.0],
        [0.2, S3H / 3],
        [0.8, S3H / 3],
        [0.5, 2 * S3H / 3],
        [0.3, S3H * 0.6],
        [0.7, S3H * 0.6],
    ], dtype=float)
    
    # Configuration 3: quasi-lattice
    init3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, 0.0],
        [0.25, S3H/2],
        [0.75, S3H/2],
        [0.5, S3H/2],
        [0.5, S3H*0.25],
        [0.5, S3H*0.75],
    ], dtype=float)
    
    for init in [init1, init2, init3]:
        _update_best(init)

    # ---- Phase A: Push-apart + BFGS from multiple starts ----
    starts = [init1, init2, init3]
    for _ in range(8):
        # Perturbed versions
        base = starts[rng.integers(len(starts))]
        perturbed = _project_to_triangle(base + rng.normal(0, 0.1, size=(N, 2)))
        starts.append(perturbed)
    
    for s in range(len(starts)):
        if not _elapsed_ok(30):
            break
        pts = starts[s].copy()
        bary = _cart_to_bary(pts)
        bary = _project_bary(bary)
        
        # Push apart
        if _elapsed_ok(25):
            pts_pushed = _push_apart(pts, n_iters=150, lr=0.003)
            _update_best(pts_pushed)
        
        # BFGS in barycentric space
        if _elapsed_ok(20):
            bary_opt = _bfgs_optimize_bary(bary, beta_start=15.0, beta_end=80.0)
            pts_opt = _bary_to_cart(bary_opt)
            _update_best(pts_opt)
    
    # ---- Phase B: Refine best with push-apart + BFGS ----
    if best_pts is not None:
        for refine_round in range(3):
            if not _elapsed_ok(15):
                break
            bary = _cart_to_bary(best_pts)
            bary = _project_bary(bary)
            
            # Push apart
            if _elapsed_ok(12):
                pts_pushed = _push_apart(best_pts, n_iters=200, lr=0.002)
                _update_best(pts_pushed)
            
            # BFGS with higher beta
            if _elapsed_ok(10):
                bary_opt = _bfgs_optimize_bary(bary, beta_start=30.0, beta_end=150.0)
                pts_opt = _bary_to_cart(bary_opt)
                _update_best(pts_opt)
    
    # ---- Phase C: Final local search with small perturbations ----
    if best_pts is not None:
        for _ in range(10):
            if not _elapsed_ok(5):
                break
            cand = _project_to_triangle(best_pts + rng.normal(0, 0.01, size=(N, 2)))
            bary = _cart_to_bary(cand)
            bary = _project_bary(bary)
            bary_opt = _bfgs_optimize_bary(bary, beta_start=50.0, beta_end=200.0)
            pts_opt = _bary_to_cart(bary_opt)
            _update_best(pts_opt)

    if best_pts is None:
        # fallback
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        best_pts = _bary_to_cart(np.column_stack([u, v]))

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
