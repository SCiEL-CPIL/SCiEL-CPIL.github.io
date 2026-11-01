"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(pts):
    """Compute the packing score: min pairwise dist/2, min (1-|c_i|)."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Container constraint
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_cont = 1.0 - np.max(norms)
    return min(min_pair, min_cont)


def _obj_and_grad(pts, r_target, soft_beta):
    """
    Objective: maximize r (equivalently minimize -r) subject to constraints.
    We use a soft penalty formulation:
    f(pts) = -r_target + penalty for violations
    
    Actually, let's use a cleaner approach:
    We want to find pts such that all pairwise dists >= 2r and all |c_i| <= 1-r.
    
    For L-BFGS-B, we'll minimize a smooth objective that penalizes constraint violations.
    Let's define:
    g_i = max(0, 2*r - dist_i) for each pair  (violation if too close)
    h_i = max(0, |c_i| - (1-r)) for each point (violation if outside)
    
    We minimize sum of soft violations. But for better optimization, let's use:
    Objective = -min_dist/2 + lambda * (violation terms)
    
    Actually, the cleanest approach for L-BFGS-B:
    Fix r, and minimize the maximum constraint violation (or a smooth surrogate).
    Then binary search on r.
    
    For a fixed r, define:
    v(pts) = max( max_i (|c_i| - (1-r)),  max_{i<j} (2*r - dist_ij) )
    We want v(pts) <= 0.
    
    Smooth surrogate: use log-sum-exp or soft-max.
    Let beta be a parameter.
    soft_max(x) = (1/beta) * log(sum(exp(beta*x)))
    
    So our objective for fixed r is:
    f(pts) = soft_max( [ |c_i| - (1-r) for all i,  2*r - dist_ij for all i<j ] )
    
    We minimize f(pts). If we can get f(pts) <= 0, then r is feasible.
    """
    n = len(pts)
    # Container violations
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)
    
    # Pairwise violations
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2.0 * r_target - dists
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    pair_viol = pair_viol[iu]
    
    all_viol = np.concatenate([cont_viol, pair_viol])
    
    # Soft max
    max_v = np.max(all_viol)
    if max_v > 50.0 / soft_beta:
        # Overflow guard
        return max_v, np.zeros_like(pts)
    
    exp_v = np.exp(soft_beta * (all_viol - max_v))
    soft_max_val = max_v + np.log(np.sum(exp_v)) / soft_beta
    
    # Gradient
    # d(soft_max)/d(all_viol_k) = exp(soft_beta * (all_viol_k - max_v)) / sum(exp(...))
    weights = exp_v / np.sum(exp_v)
    
    grad = np.zeros_like(pts)
    
    # Container part: d|c_i|/d(c_i) = c_i / |c_i|
    for i in range(n):
        norm_i = norms[i]
        if norm_i > 1e-10:
            grad[i] += weights[i] * pts[i] / norm_i
    
    # Pairwise part: d(2r - dist_ij)/d(c_i) = -(c_i - c_j) / dist_ij
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[n + idx]
        d = dists[i, j]
        if d > 1e-10:
            diff_ij = pts[i] - pts[j]
            grad[i] -= w * diff_ij / d
            grad[j] += w * diff_ij / d
    
    return soft_max_val, grad


def _ring_init(n, r_guess):
    """Initialize points in concentric rings."""
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    remaining = n - 1
    ring_radius = r_guess * 2.0
    ring_idx = 1
    placed = 1
    while placed < n:
        # How many points can fit on this ring?
        # Circumference / (2*r_guess) approximately
        n_on_ring = max(1, int(2.0 * np.pi * ring_radius / (2.0 * r_guess)))
        n_on_ring = min(n_on_ring, remaining)
        angles = 2.0 * np.pi * np.arange(n_on_ring) / n_on_ring + np.pi / n_on_ring
        for k in range(n_on_ring):
            pts[placed] = [ring_radius * np.cos(angles[k]), ring_radius * np.sin(angles[k])]
            placed += 1
        remaining = n - placed
        ring_radius += 2.0 * r_guess
        ring_idx += 1
    return pts


def _optimize_for_radius(pts_init, r, soft_beta, max_iter=2000, tol=1e-10):
    """Minimize soft constraint violation for fixed r."""
    n = pts_init.shape[0]
    
    def objective(x):
        # x is 1D, reshape to 2D for calculations
        pts = x.reshape(n, 2)
        val, grad = _obj_and_grad(pts, r, soft_beta)
        # grad is 2D, flatten to 1D for L-BFGS-B
        return val, grad.flatten()
    
    # Flatten initial points for L-BFGS-B
    x0 = pts_init.flatten()
    
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    
    # Reshape result back to 2D
    pts_opt = result.x.reshape(n, 2)
    return pts_opt, result.fun


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    # Initial guess for radius
    r_best = 0.12
    best_pts = _ring_init(N, r_best)
    best_score = _score(best_pts)
    
    # Binary search on radius
    r_low = 0.05
    r_high = 0.15
    
    # First, find a feasible r_high
    while time.time() < deadline - 30.0:
        r_mid = (r_low + r_high) / 2.0
        pts_init = _ring_init(N, r_mid)
        
        # Try with increasing soft_beta for tighter convergence
        found_feasible = False
        for beta in [10.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 10.0:
                break
            pts_opt, val = _optimize_for_radius(pts_init, r_mid, beta, max_iter=1000)
            if val <= 1e-8:
                # Check actual score
                score = _score(pts_opt)
                if score >= r_mid - 1e-9:
                    found_feasible = True
                    if score > best_score:
                        best_score = score
                        best_pts = pts_opt.copy()
                        r_best = r_mid
                    break
        
        if found_feasible:
            r_low = r_mid
        else:
            r_high = r_mid
        
        # Converged?
        if r_high - r_low < 1e-6:
            break
    
    # Final polish with high precision
    if time.time() < deadline - 5.0:
        r_polish = r_low
        pts_init = best_pts.copy()
        for beta in [100.0, 500.0, 1000.0]:
            if time.time() > deadline - 2.0:
                break
            pts_opt, val = _optimize_for_radius(pts_init, r_polish, beta, max_iter=3000)
            score = _score(pts_opt)
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
            pts_init = pts_opt
    
    # Validate
    best_pts = np.array(best_pts, dtype=np.float64)
    if np.any(~np.isfinite(best_pts)) or len(best_pts) != N:
        # Fallback to hexagonal grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_pts

# EVOLVE-BLOCK-END
