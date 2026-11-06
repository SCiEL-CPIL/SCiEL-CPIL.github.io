"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _log_score_and_grad(points):
    """Compute log(Delta/n^n) and its gradient analytically.
    
    The score is Delta/n^n where Delta is the product of all ordered pair distances
    after scaling to diameter 2. We work in log space for numerical stability.
    
    Returns (log_score, gradient) where gradient is d(log_score)/d(points).
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(sq_dists, 1e-30))
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e30, np.zeros_like(pts)
    
    # Scale to diameter 2: scaled_dists = dists * 2 / diam
    scale = 2.0 / diam
    scaled_dists = dists * scale
    
    # Mask out diagonal (self-distances)
    mask = ~np.eye(n, dtype=bool)
    
    # log_delta = sum over ordered pairs of log(scaled_dist)
    # = sum_{i != j} log(scaled_dists[i,j])
    log_delta = np.sum(np.log(scaled_dists[mask]))
    
    # log_score = log_delta - n * log(n)
    log_score = log_delta - n * np.log(n)
    
    # Now compute gradient of log_score w.r.t. pts
    # log_score = sum_{i!=j} log(|p_i - p_j| * 2/diam) - n*log(n)
    #           = sum_{i!=j} [log|p_i - p_j| + log(2) - log(diam)] - n*log(n)
    #           = sum_{i!=j} log|p_i - p_j| - n*(n-1)*log(diam) + n*(n-1)*log(2) - n*log(n)
    
    # Let S = sum_{i!=j} log|p_i - p_j|, D = diam = max_{i!=j} |p_i - p_j|
    # log_score = S - n*(n-1)*log(D) + const
    
    # dS/dp_k = sum_{j!=k} (p_k - p_j)/|p_k - p_j|^2  [for each coordinate]
    # d(log D)/dp_k: D is max of |p_i - p_j|, so only the pair(s) achieving the max contribute
    # If unique max pair (a,b): d(log D)/dp_k = delta_{k,a}*(p_a-p_b)/D^2 + delta_{k,b}*(p_b-p_a)/D^2
    
    # Compute dS/dp
    dS = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-15:
                continue
            # d/dp_i of log|p_i - p_j| = (p_i - p_j) / |p_i - p_j|^2
            dS[i] += diff[i, j] / (d * d)
    
    # Find which pair(s) achieve the diameter
    max_val = diam
    # Find all pairs with distance close to max
    eps = 1e-10
    max_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(dists[i, j] - max_val) < eps:
                max_pairs.append((i, j))
    
    # d(log D)/dp_k for the max pair(s)
    dlogD = np.zeros_like(pts)
    for (a, b) in max_pairs:
        if len(max_pairs) > 1:
            # Multiple max pairs - use subgradient (average)
            weight = 1.0 / len(max_pairs)
        else:
            weight = 1.0
        d_ab = diff[a, b]  # p_a - p_b
        D_sq = max_val * max_val
        dlogD[a] += weight * d_ab / D_sq
        dlogD[b] += weight * (-d_ab) / D_sq
    
    # Gradient of log_score = dS/dp - n*(n-1) * dlogD/dp
    n_pairs = n * (n - 1)
    grad = dS - n_pairs * dlogD
    
    return log_score, grad

def _score(points):
    """Compute Delta / n^n for 8 points."""
    log_s, _ = _log_score_and_grad(points)
    return np.exp(log_s)

def _neg_log_score_and_grad(points):
    """Negative log-score and its gradient for minimization."""
    pts = np.array(points, dtype=float).flatten()
    pts_2d = pts.reshape(8, 2)
    log_s, grad = _log_score_and_grad(pts_2d)
    return -log_s, grad.flatten()

def _neg_log_score(points):
    """Negative log-score."""
    pts = np.array(points, dtype=float)
    log_s, _ = _log_score_and_grad(pts)
    return -log_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with regular octagon
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_points = np.array(octagon, dtype=float)
    best_log_score = _log_score_and_grad(best_points)[0]
    
    rng = np.random.RandomState(42)
    
    def _is_valid(candidate):
        """Check if candidate points are distinct."""
        diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists) > 1e-8
    
    def generate_starts():
        starts = []
        # Regular octagon
        starts.append(np.array(octagon, dtype=float))
        
        # Perturbed octagons with various patterns
        for trial in range(30):
            angles = 2 * pi * np.arange(8) / 8
            strategy = trial % 6
            
            if strategy == 0:
                radii = 1.0 + 0.25 * rng.randn(8)
                radii = np.clip(radii, 0.4, 1.6)
                angles_pert = angles + 0.15 * rng.randn(8)
            elif strategy == 1:
                base_r = 1.0 + 0.15 * rng.randn()
                alt_r = 1.0 - 0.15 * rng.randn()
                radii = np.where(np.arange(8) % 2 == 0, base_r, alt_r)
                angles_pert = angles + 0.08 * rng.randn(8)
            elif strategy == 2:
                radii = 1.0 + 0.2 * rng.randn(4)
                radii = np.tile(radii, 2)
                angles_pert = angles + 0.1 * rng.randn(8)
            elif strategy == 3:
                phase = rng.randn()
                radii = 1.0 + 0.2 * np.sin(2 * angles + phase)
                angles_pert = angles + 0.08 * rng.randn(8)
            elif strategy == 4:
                radii = 1.0 + 0.35 * rng.randn(8)
                radii = np.clip(radii, 0.2, 1.8)
                angles_pert = angles + 0.2 * rng.randn(8)
            else:
                # Two circles with different radii
                r1 = 0.8 + 0.3 * rng.randn()
                r2 = 1.0 + 0.3 * rng.randn()
                radii = np.where(np.arange(8) % 2 == 0, r1, r2)
                angles_pert = angles + 0.1 * rng.randn(8)
            
            pts = np.column_stack([radii * np.cos(angles_pert), radii * np.sin(angles_pert)])
            starts.append(pts)
        
        # Square with vertices at corners and midpoints (4-fold symmetry)
        for r_inner in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            angles_outer = np.array([0, pi/2, pi, 3*pi/2])
            angles_inner = angles_outer + pi/4
            pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            pts_inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
            pts = np.vstack([pts_outer, pts_inner])
            starts.append(pts)
        
        # Try configurations with 2-fold symmetry (elongated)
        for dx in [0.1, 0.2, 0.3, 0.4]:
            base = np.array(octagon, dtype=float)
            for i in range(8):
                if i % 2 == 0:
                    base[i, 0] += dx
                else:
                    base[i, 0] -= dx
            starts.append(base.copy())
        
        # Random configurations
        for _ in range(10):
            pts = rng.randn(8, 2) * 1.0
            starts.append(pts)
        
        return starts
    
    starts = generate_starts()
    
    # Optimize each starting point using L-BFGS-B with analytical gradient
    for idx, x0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
        
        x0_flat = x0.flatten()
        
        try:
            if time.time() > deadline - 3.0:
                break
            
            # Use L-BFGS-B with analytical gradient
            result = minimize(_neg_log_score, x0_flat, jac=_neg_log_score_and_grad,
                            method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxfun': 10000})
            
            if time.time() > deadline - 3.0:
                break
            
            candidate = result.x.reshape(8, 2)
            
            if not _is_valid(candidate):
                continue
            
            cand_log_score = _log_score_and_grad(candidate)[0]
            if cand_log_score > best_log_score:
                best_log_score = cand_log_score
                best_points = candidate
                
                # Refine with Nelder-Mead if time permits
                if time.time() < deadline - 8.0:
                    try:
                        result2 = minimize(_neg_log_score, best_points.flatten(),
                                          method='Nelder-Mead',
                                          options={'maxiter': 5000, 'xatol': 1e-12,
                                                  'fatol': 1e-14})
                        candidate2 = result2.x.reshape(8, 2)
                        if _is_valid(candidate2):
                            cand_log_score2 = _log_score_and_grad(candidate2)[0]
                            if cand_log_score2 > best_log_score:
                                best_log_score = cand_log_score2
                                best_points = candidate2
                    except Exception:
                        pass
                        
        except Exception:
            pass
    
    # Final local refinement with multiple small perturbations
    if time.time() < deadline - 5.0:
        for pert_trial in range(15):
            if time.time() > deadline - 3.0:
                break
            x0 = best_points.flatten() + 0.01 * rng.randn(16)
            try:
                result = minimize(_neg_log_score, x0, jac=_neg_log_score_and_grad,
                                method='L-BFGS-B',
                                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
                candidate = result.x.reshape(8, 2)
                if _is_valid(candidate):
                    cand_log_score = _log_score_and_grad(candidate)[0]
                    if cand_log_score > best_log_score:
                        best_log_score = cand_log_score
                        best_points = candidate
            except Exception:
                pass
    
    # Return the best points found
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
