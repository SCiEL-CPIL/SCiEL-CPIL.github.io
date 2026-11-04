"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Barycentric to Cartesian: vertex0=(0,0), vertex1=(1,0), vertex2=(0.5, H)
# alpha + beta + gamma = 1, all >= 0
# x = beta + 0.5*gamma, y = H*gamma


def _bary_to_cart(bary):
    """Convert (n,3) barycentric to (n,2) Cartesian."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert (n,2) Cartesian to (n,3) barycentric (unnormalized, may have small negatives)."""
    x = points[:, 0]
    y = points[:, 1]
    gamma = y / H
    beta = x - 0.5 * gamma
    alpha = 1.0 - beta - gamma
    return np.column_stack([alpha, beta, gamma])


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    points[:, 0] = x
    points[:, 1] = y
    return points


def _is_inside_triangle(points, eps=1e-5):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    if np.any(S3 * x - y < -eps):
        return False
    if np.any(S3 * (1 - x) - y < -eps):
        return False
    return True


def _bary_objective(flat):
    """
    Objective in barycentric coords. We maximize min area, so minimize -min_area.
    We add a large penalty if any barycentric coord is negative (outside triangle).
    flat: 3*N array
    """
    bary = flat.reshape(N, 3)
    # Penalty for negative barycentric coordinates
    neg = np.minimum(bary, 0)
    penalty = 100.0 * np.sum(neg ** 2)
    
    pts = _bary_to_cart(bary)
    min_area = _min_triangle_area_fast(pts)
    return -min_area + penalty


def _bary_gradient(flat):
    """
    Approximate gradient of the barycentric objective using finite differences.
    This is expensive but N=33 is manageable.
    """
    eps = 1e-7
    grad = np.zeros_like(flat)
    f0 = _bary_objective(flat)
    for i in range(flat.shape[0]):
        orig = flat[i]
        flat[i] = orig + eps
        f_plus = _bary_objective(flat)
        flat[i] = orig - eps
        f_minus = _bary_objective(flat)
        flat[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _generate_fps_initialization(n_points=11, seed=42):
    """Farthest Point Sampling based initialization in Cartesian, converted to barycentric."""
    rng = np.random.RandomState(seed)
    
    points = [np.array([0.5, H / 3])]
    
    for _ in range(n_points - 1):
        n_candidates = 2000
        r1 = rng.rand(n_candidates)
        r2 = rng.rand(n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        cx = r1 + 0.5 * r2
        cy = H * r2
        candidates = np.column_stack([cx, cy])
        
        pts_arr = np.array(points)
        diff = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        best_idx = np.argmax(min_dist_sq)
        points.append(candidates[best_idx])
    
    pts = np.array(points)
    bary = _cart_to_bary(pts)
    # Ensure barycentric coords are non-negative (clip)
    bary = np.clip(bary, 0, None)
    # Renormalize
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary.flatten()


def _structured_inits():
    """Generate diverse structured initial configurations in barycentric coords."""
    inits = []
    
    # Symmetric arrangement
    base_bary = np.array([
        [0.10, 0.10, 0.80],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.40, 0.10, 0.50],
        [0.10, 0.40, 0.50],
        [0.50, 0.40, 0.10],
        [0.33, 0.33, 0.34],
        [0.20, 0.55, 0.25],
        [0.55, 0.20, 0.25],
        [0.25, 0.20, 0.55],
        [0.45, 0.25, 0.30],
    ])
    base_bary[:, 2] = 1 - base_bary[:, 0] - base_bary[:, 1]
    inits.append(base_bary.flatten())
    
    # Grid-like
    grid_bary = np.array([
        [0.15, 0.15, 0.70],
        [0.45, 0.15, 0.40],
        [0.75, 0.15, 0.10],
        [0.25, 0.45, 0.30],
        [0.55, 0.45, 0.00],
        [0.85, 0.45, -0.30],
        [0.35, 0.75, 0.00],
        [0.65, 0.75, -0.40],
        [0.40, 0.30, 0.30],
        [0.60, 0.30, 0.10],
        [0.50, 0.50, 0.00],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    grid_bary = np.clip(grid_bary, 0, None)
    grid_bary = grid_bary / grid_bary.sum(axis=1, keepdims=True)
    inits.append(grid_bary.flatten())
    
    # FPS inits
    for seed in [42, 123, 456, 789, 101]:
        inits.append(_generate_fps_initialization(N, seed))
    
    return inits


def _local_refine_bary(bary_flat, time_budget=10.0):
    """
    Local refinement using L-BFGS-B with barycentric representation.
    Uses a barrier-like penalty for boundary violations.
    """
    start = time.time()
    
    # Bounds: each barycentric coord in [0, 1]
    bounds = [(0, 1)] * (3 * N)
    
    def obj(flat):
        bary = flat.reshape(N, 3)
        # Strong penalty for negative coords (shouldn't happen with bounds, but for safety)
        neg = np.minimum(bary, 0)
        penalty = 50.0 * np.sum(neg ** 2)
        pts = _bary_to_cart(bary)
        min_area = _min_triangle_area_fast(pts)
        return -min_area + penalty
    
    def grad(flat):
        # Numerical gradient
        eps = 1e-6
        g = np.zeros_like(flat)
        f0 = obj(flat)
        for i in range(flat.shape[0]):
            orig = flat[i]
            flat[i] = orig + eps
            fp = obj(flat)
            flat[i] = orig - eps
            fm = obj(flat)
            flat[i] = orig
            g[i] = (fp - fm) / (2 * eps)
        return g
    
    try:
        res = minimize(
            obj,
            bary_flat,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return bary_flat


def _nelder_mead_refine(flat, time_budget=5.0):
    """Nelder-Mead refinement in Cartesian space."""
    start = time.time()
    
    def cart_obj(flat_c):
        pts = flat_c.reshape(N, 2)
        pts = _clip_to_triangle(pts)
        return -_min_triangle_area_fast(pts)
    
    try:
        res = minimize(
            cart_obj,
            flat,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return flat


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_points, best_score
        pts = _clip_to_triangle(pts)
        if _is_inside_triangle(pts, eps=1e-4):
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    
    inits = _structured_inits()
    
    # Phase 1: Barycentric L-BFGS-B refinement from multiple starts
    for i, init_flat in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        try:
            refined = _local_refine_bary(init_flat)
            bary = refined.reshape(N, 3)
            bary = np.clip(bary, 0, None)
            bary = bary / bary.sum(axis=1, keepdims=True)
            pts = _bary_to_cart(bary)
            update_best(pts)
        except Exception:
            continue
    
    # Phase 2: Cartesian Nelder-Mead from best found and inits
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    for init_flat in inits:
        bary = init_flat.reshape(N, 3)
        bary = np.clip(bary, 0, None)
        bary = bary / bary.sum(axis=1, keepdims=True)
        pts = _bary_to_cart(bary)
        candidates.append(pts.flatten())
    
    # Add perturbations of best
    if best_points is not None:
        rng = np.random.RandomState(42)
        for _ in range(3):
            perturbed = best_points + rng.randn(N, 2) * 0.03
            candidates.append(perturbed.flatten())
    
    for i, init_flat in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        try:
            refined = _nelder_mead_refine(init_flat)
            pts = refined.reshape(N, 2)
            update_best(pts)
        except Exception:
            continue
    
    # Phase 3: Final polish with small L-BFGS-B steps in barycentric
    if best_points is not None:
        bary = _cart_to_bary(best_points)
        bary = np.clip(bary, 1e-6, None)
        bary = bary / bary.sum(axis=1, keepdims=True)
        
        try:
            refined = _local_refine_bary(bary.flatten())
            bary2 = refined.reshape(N, 3)
            bary2 = np.clip(bary2, 0, None)
            bary2 = bary2 / bary2.sum(axis=1, keepdims=True)
            pts2 = _bary_to_cart(bary2)
            update_best(pts2)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        for init_flat in inits:
            bary = init_flat.reshape(N, 3)
            bary = np.clip(bary, 0, None)
            bary = bary / bary.sum(axis=1, keepdims=True)
            pts = _bary_to_cart(bary)
            update_best(pts)
    
    if best_points is None or not _is_inside_triangle(best_points, eps=1e-3):
        best_points = _clip_to_triangle(_bary_to_cart(_cart_to_bary(np.column_stack([
            np.linspace(0.1, 0.9, N),
            np.linspace(0.1, 0.5, N)
        ]))))
        # That fallback is messy; use a simple one
        rng = np.random.RandomState(0)
        r1 = rng.rand(N)
        r2 = rng.rand(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        cx = r1 + 0.5 * r2
        cy = H * r2
        best_points = _clip_to_triangle(np.column_stack([cx, cy]))
    
    return best_points
