"""Improved packing of 50 circles in a unit circle using soft-min optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # seconds, safety margin before 150s wall-clock kill


def _compute_radius(pts):
    """Compute the packing radius (minimum of min pair dist/2 and min distance to boundary)."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.min(np.hypot(pts[:, 0], pts[:, 1])), 1.0)
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity to exclude self-pairs
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    norms = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - norms)
    return min(min_pair, min_boundary)


def _soft_min_radius(pts, beta=100.0):
    """
    Soft-min surrogate for the packing radius using log-sum-exp.
    This gives a smooth approximation to min(min_pair/2, min_boundary).
    As beta -> infinity, this approaches the true min.
    """
    n = len(pts)
    if n < 2:
        return min(1.0 - np.min(np.hypot(pts[:, 0], pts[:, 1])), 1.0)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Only upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    
    # Boundary distances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - norms
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta*x)))/beta
    # For numerical stability, subtract min
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = min_val - log_sum / beta
    
    return soft_min


def _soft_min_radius_grad(pts, beta=100.0):
    """
    Gradient of the soft-min radius with respect to pts.
    Returns (value, grad) where grad has shape (n, 2).
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros((n, 2))
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    
    # Boundary distances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - norms
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, boundary_dists])
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    # Weights for each distance in the soft-min
    weights = exp_vals / sum_exp
    
    grad = np.zeros((n, 2))
    
    # Pairwise distance gradients
    # d(pair_dist)/d(pts[i]) = (pts[i] - pts[j]) / pair_dist
    # d(pair_dist)/d(pts[j]) = (pts[j] - pts[i]) / pair_dist
    n_pairs = len(iu)
    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        d = pair_dists[k]
        if d < 1e-12:
            continue
        w = weights[k]
        diff_ij = pts[i] - pts[j]
        grad[i] += w * diff_ij / d
        grad[j] -= w * diff_ij / d
    
    # Boundary distance gradients
    # d(1 - norm)/d(pts[i]) = -pts[i] / norm
    n_boundary = n
    for i in range(n):
        norm = norms[i]
        if norm < 1e-12:
            continue
        w = weights[n_pairs + i]
        grad[i] -= w * pts[i] / norm
    
    value = min_val - np.log(sum_exp) / beta
    return value, grad


def _hex_init(n):
    """Generate a hexagonal lattice initial packing, trimmed to n points."""
    best_pts = None
    best_score = 0.0
    for s in np.linspace(0.15, 0.35, 40):
        pts = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    pts.append((x, y))
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            candidate = pts[idx]
            score = _compute_radius(candidate)
            if score > best_score:
                best_score = score
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    return np.random.randn(n, 2) * 0.5


def _ring_init(n):
    """Generate a ring-based initial packing."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Rings
    remaining = n - 1
    ring_idx = 1
    while remaining > 0 and ring_idx <= 10:
        # Approximate radius for this ring
        r = 0.2 * ring_idx
        # Number of points on this ring
        circumference = 2 * np.pi * r
        count = min(remaining, max(1, int(circumference / 0.25)))
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
        remaining -= count
        ring_idx += 1
    
    pts = np.array(pts[:n])
    return pts


def _random_init(n, seed=42):
    """Generate a random initial packing."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 2)
    norms = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts / norms[:, np.newaxis] * 0.8
    return pts


def _optimize_from_init(pts_init, beta_schedule, max_time):
    """
    Optimize from a given initial point using L-BFGS-B with soft-min objective.
    Returns the best points found.
    """
    n = len(pts_init)
    best_pts = pts_init.copy()
    best_r = _compute_radius(best_pts)
    
    def objective_and_grad(x):
        pts = x.reshape(n, 2)
        val, grad = _soft_min_radius_grad(pts, beta=beta_schedule)
        # We want to maximize, so minimize negative
        return -val, -grad.flatten()
    
    x0 = pts_init.flatten()
    
    try:
        res = minimize(
            objective_and_grad, x0, method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
        )
        if res.x is not None:
            pts_res = res.x.reshape(n, 2)
            r_res = _compute_radius(pts_res)
            if r_res > best_r:
                best_pts = pts_res
                best_r = r_res
    except Exception:
        pass
    
    return best_pts, best_r


def _nelder_mead_polish(pts, max_iter=1000, tol=1e-10):
    """
    Final polish using Nelder-Mead on the actual (non-smooth) radius.
    """
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -_compute_radius(pts)
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'disp': False}
        )
        if res.x is not None:
            pts_res = res.x.reshape(n, 2)
            r_res = _compute_radius(pts_res)
            r_orig = _compute_radius(pts)
            if r_res > r_orig:
                return pts_res
    except Exception:
        pass
    
    return pts


def _final_polish(pts, remaining_time):
    """
    High-precision final polish using very high beta soft-min and then Nelder-Mead.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Try several high-beta values
    for beta in [500.0, 1000.0, 2000.0]:
        if time.time() > _DEADLINE - 5:
            break
        
        def objective_and_grad(x):
            pts = x.reshape(n, 2)
            val, grad = _soft_min_radius_grad(pts, beta=beta)
            return -val, -grad.flatten()
        
        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective_and_grad, x0, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'disp': False}
            )
            if res.x is not None:
                pts_res = res.x.reshape(n, 2)
                r_res = _compute_radius(pts_res)
                if r_res > best_r:
                    best_pts = pts_res
                    best_r = r_res
        except Exception:
            pass
    
    # Nelder-Mead polish
    if time.time() < _DEADLINE - 3:
        best_pts = _nelder_mead_polish(best_pts, max_iter=500, tol=1e-12)
        best_r = _compute_radius(best_pts)
    
    return best_pts


def construct():
    t0 = time.time()
    n = N
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_hex_init(n))
    inits.append(_ring_init(n))
    
    rng = np.random.RandomState(123)
    for seed in [42, 123, 456, 789, 101112]:
        inits.append(_random_init(n, seed=seed))
    
    # Also add perturbed versions of hex init
    hex_pts = inits[0]
    for scale in [0.01, 0.02, 0.05]:
        pert = hex_pts + rng.randn(n, 2) * scale
        inits.append(pert)
    
    best_pts = None
    best_r = 0.0
    
    # Optimize from each initial configuration
    beta_schedule = 100.0
    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 15:
            break
        
        # Check time budget
        elapsed = time.time() - t0
        remaining = _DEADLINE - elapsed - 15  # reserve 15s for final polish
        
        if remaining < 5:
            break
        
        # Scale maxiter based on remaining time
        max_iter = max(50, int(remaining * 10))
        
        pts_res, r_res = _optimize_from_init(init_pts, beta_schedule, max_iter)
        
        if r_res > best_r:
            best_pts = pts_res
            best_r = r_res
    
    if best_pts is None:
        best_pts = inits[0]
        best_r = _compute_radius(best_pts)
    
    # Final polish
    if time.time() < _DEADLINE - 10:
        best_pts = _final_polish(best_pts, _DEADLINE - time.time())
        best_r = _compute_radius(best_pts)
    
    # Verify final result is valid
    final_r = _compute_radius(best_pts)
    if final_r <= 0 or not np.all(np.isfinite(best_pts)):
        # Fallback to hex grid
        best_pts = _hex_init(n)
    
    return best_pts
