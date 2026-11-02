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


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v >= 0."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.array([x, y])


def _slp_step(pts, max_iter=50, tol=1e-8):
    """
    Successive Linear Programming (SLP) step.
    Maximizes t subject to:
      area(i,j,k) >= t for all triplets
      points in triangle
    Uses linearization of the area constraint at current point.
    """
    n = len(pts)
    # Work in Cartesian coordinates
    p = pts.copy()
    
    for it in range(max_iter):
        areas = _all_triangle_areas(p)
        min_area = np.min(areas)
        
        # Find active constraints (triangles with area close to min)
        active_mask = areas <= min_area + 1e-6 * max(1.0, min_area)
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) == 0:
            active_idx = [np.argmin(areas)]
        
        # Build LP: maximize t
        # Variables: p[0..2n-1], t (index 2n)
        # Constraints: for each active triplet (i,j,k):
        #   area(i,j,k) >= t
        #   Linearize: cross product is bilinear, so linearize around current p
        
        # Area = 0.5 * |cross|
        # For active triplets, we know the sign of cross at current p
        # cross = (pj-pi) x (pk-pi) = (px_j - px_i)*(py_k - py_i) - (py_j - py_i)*(px_k - px_i)
        
        # Linearization of cross around p:
        # cross(p + dp) ≈ cross(p) + d(cross)/dp · dp
        # d(cross)/d(px_i) = -(py_k - py_i)  ... wait, let me be careful
        # cross = (px_j - px_i)*(py_k - py_i) - (py_j - py_i)*(px_k - px_i)
        # d/d(px_i): -(py_k - py_i) + (py_j - py_i)*(1) = -(py_k - py_i) + (py_j - py_i)
        #           = py_j - py_k
        # d/d(px_j): (py_k - py_i)
        # d/d(px_k): -(py_j - py_i)
        # d/d(py_i): -(px_j - px_i) + (px_k - px_i) = px_k - px_j
        # d/d(py_j): (px_k - px_i)
        # d/d(py_k): -(px_j - px_i)
        
        # We want: 0.5 * sign * (cross(p) + grad · dp) >= t
        # i.e., sign * (cross(p) + grad · dp) - 2t >= 0
        
        # Let's build the LP using scipy.optimize.linprog
        # Maximize t => minimize -t
        # Variables: x[0:2n] = point coords, x[2n] = t
        
        m = len(active_idx)
        # A_ub * x <= b_ub
        # For each active constraint: -sign*(grad_x · dx + cross_val) + 2*t <= 0
        # Rearranged: sign*(grad_x · dx + cross_val) - 2*t >= 0
        # For linprog (minimize c·x s.t. A_ub·x <= b_ub):
        # -sign*(grad_x · dx + cross_val) + 2*t <= 0
        
        A_ub = np.zeros((m, 2 * n + 1))
        b_ub = np.zeros(m)
        
        for idx, tri_idx in enumerate(active_idx):
            i, j, k = _TRIPLETS[tri_idx]
            pi = p[i]
            pj = p[j]
            pk = p[k]
            
            # Current cross value
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            dx2 = pk[0] - pi[0]
            dy2 = pk[1] - pi[1]
            cross_val = dx1 * dy2 - dy1 * dx2
            
            sign = 1.0 if cross_val >= 0 else -1.0
            
            # Gradient of cross w.r.t. point coordinates
            # d(cross)/d(px_i) = py_j - py_k
            # d(cross)/d(px_j) = py_k - py_i
            # d(cross)/d(px_k) = py_i - py_j
            # d(cross)/d(py_i) = px_k - px_j
            # d(cross)/d(py_j) = px_i - px_k
            # d(cross)/d(py_k) = px_j - px_i
            
            # Constraint: sign * (cross_val + grad·dp) - 2*t >= 0
            # => -sign * (grad·dp) + 2*t <= sign * cross_val
            # Note: dp = x[0:2n] - p_flat, so grad·dp = grad·x[0:2n] - grad·p_flat
            
            # A_ub row: -sign * grad for point coords, +2 for t
            # b_ub: sign * cross_val + sign * grad · p_flat = sign * (cross_val + grad·p_flat)
            
            grad = np.zeros(2 * n)
            grad[2 * i] = pj[1] - pk[1]      # d/d(px_i)
            grad[2 * i + 1] = pk[0] - pj[0]  # d/d(py_i)
            grad[2 * j] = pk[1] - pi[1]      # d/d(px_j)
            grad[2 * j + 1] = pi[0] - pk[0]  # d/d(py_j)
            grad[2 * k] = pi[1] - pj[1]      # d/d(px_k)
            grad[2 * k + 1] = pj[0] - pi[0]  # d/d(py_k)
            
            A_ub[idx, 0:2 * n] = -sign * grad
            A_ub[idx, 2 * n] = 2.0
            
            p_flat = p.flatten()
            b_ub[idx] = sign * (cross_val + np.dot(grad, p_flat))
        
        # Triangle boundary constraints (in Cartesian):
        # y >= 0
        # x >= y / S3  =>  x - y/S3 >= 0
        # x <= 1 - y/S3 => -x - y/S3 >= -1
        
        # Add these as additional rows in A_ub
        n_boundary = 3 * n
        A_full = np.vstack([A_ub, np.zeros((n_boundary, 2 * n + 1))])
        b_full = np.concatenate([b_ub, np.zeros(n_boundary)])
        
        for i in range(n):
            # y_i >= 0  =>  -y_i <= 0
            A_full[n_boundary // 3 * 3 + i if False else len(A_ub) + i, 2 * i + 1] = -1.0
            b_full[len(A_ub) + i] = 0.0
            
            # x_i - y_i/S3 >= 0  =>  -x_i + y_i/S3 <= 0
            A_full[len(A_ub) + n + i, 2 * i] = -1.0
            A_full[len(A_ub) + n + i, 2 * i + 1] = 1.0 / S3
            b_full[len(A_ub) + n + i] = 0.0
            
            # -x_i - y_i/S3 >= -1  =>  x_i + y_i/S3 <= 1
            A_full[len(A_ub) + 2 * n + i, 2 * i] = 1.0
            A_full[len(A_ub) + 2 * n + i, 2 * i + 1] = 1.0 / S3
            b_full[len(A_ub) + 2 * n + i] = 1.0
        
        # Objective: minimize -t => c[2n] = -1
        c = np.zeros(2 * n + 1)
        c[2 * n] = -1.0
        
        # Bounds: t >= 0 (lower bound), points unbounded in LP (handled by constraints)
        bounds = [(None, None)] * (2 * n) + [(0.0, None)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
            if not res.success:
                break
            new_p = res.x[0:2 * n].reshape(n, 2)
            new_t = res.x[2 * n]
        except Exception:
            break
        
        # Check improvement
        if abs(new_t - min_area) < tol:
            p = new_p
            break
        
        # Damping to ensure stability
        p = 0.5 * p + 0.5 * new_p
        p = _project_to_triangle(p)
    
    return p


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Symmetric initial configuration with high symmetry ----
    # Use a configuration that respects the 3-fold symmetry of the equilateral triangle
    # Corners + midpoints + inner symmetric points
    init = np.array([
        [0.0, 0.0],           # corner 0
        [1.0, 0.0],           # corner 1
        [0.5, S3H],           # corner 2
        [0.5, 0.0],           # mid bottom edge
        [0.75, S3H / 2],      # mid right edge
        [0.25, S3H / 2],      # mid left edge
        [1.0/3.0, S3H / 3.0], # centroid
        [0.25, 0.0],          # bottom edge, left quarter
        [0.75, 0.0],          # bottom edge, right quarter
        [0.5, S3H / 3.0],     # on median from top
        [0.5, 2.0 * S3H / 3.0], # on median from top, higher
    ], dtype=float)
    _update_best(init)

    # ---- Phase A: SLP optimization from multiple starts ----
    n_starts = 15
    bases = [init]
    for _ in range(10):
        bases.append(_project_to_triangle(init + rng.normal(0, 0.15, size=(N, 2))))
    for _ in range(5):
        rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        bases.append(rand_pts)

    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        pts = bases[s % len(bases)].copy()
        _update_best(pts)
        pts = _slp_step(pts, max_iter=30, tol=1e-7)
        _update_best(pts)

    # ---- Phase B: Local polish with Nelder-Mead on best configuration ----
    if best_pts is not None and _elapsed_ok(20):
        for scale in [0.03, 0.015, 0.008]:
            if not _elapsed_ok(10):
                break
            for _ in range(5):
                if not _elapsed_ok(8):
                    break
                cand = _project_to_triangle(best_pts + rng.normal(0, scale, size=(N, 2)))
                
                def nm_obj(x):
                    p = x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    areas = _all_triangle_areas(p)
                    # Penalize if any area is too small
                    min_a = np.min(areas)
                    # Soft penalty for areas below target
                    target = min_a
                    penalty = np.sum(np.maximum(0, target - areas) ** 2)
                    return -min_a + 0.1 * penalty
                try:
                    res = minimize(nm_obj, cand.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                    p = _project_to_triangle(res.x.reshape(-1, 2))
                    _update_best(p)
                except Exception:
                    pass

    # ---- Final SLP polish ----
    if best_pts is not None and _elapsed_ok(15):
        pts = _slp_step(best_pts, max_iter=50, tol=1e-9)
        _update_best(pts)

    if best_pts is None:
        # fallback
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        best_pts = _barycentric_to_cartesian(np.column_stack([u, v]))

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
