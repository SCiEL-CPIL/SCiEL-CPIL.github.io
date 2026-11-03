"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _radius(centers):
    """Compute the max common radius for given centers (exact, O(N^2))."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)


def _smooth_obj(x, alpha=40.0):
    """
    Smooth surrogate: negative soft-min of squared distances.
    r_soft = -logsumexp(-d^2 / alpha) / (2*alpha)  where d^2 includes
    pairwise squared distances and 4 wall distances per point.
    Minimizing -r_soft ≈ maximizing r.
    """
    c = x.reshape(N, 2)
    # Pairwise squared distances (upper triangle only to save time)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    # Keep only upper triangle values
    iu = np.triu_indices(N, k=1)
    pair_d2 = d2[iu]
    
    # Wall squared distances: (0.5 - |c_x|)^2 and (0.5 - |c_y|)^2 for each point
    wall_x = (0.5 - np.abs(c[:, 0])) ** 2
    wall_y = (0.5 - np.abs(c[:, 1])) ** 2
    wall_d2 = np.concatenate([wall_x, wall_y])
    
    # Combine all squared distances
    all_d2 = np.concatenate([pair_d2, wall_d2])
    
    # Soft-min: r_soft^2 = -logsumexp(-d2/alpha) / alpha  (this is ~ min d2 for large alpha)
    # Actually: softmin(d2) ≈ -alpha * log(mean(exp(-d2/alpha)))
    # Let's use: softmin_val = -alpha * logsumexp(-d2/alpha) + alpha*log(len)
    # Simpler: use the log-sum-exp trick
    m = np.min(all_d2)
    shifted = -all_d2 / alpha + m / alpha  # = -(d2 - m)/alpha
    lse = m / alpha + np.log(np.sum(np.exp(shifted)))
    # softmin of d2 ≈ m - alpha * log(sum(exp(-(d2-m)/alpha))) ... 
    # Let me just use: soft_r2 = -alpha * log(mean(exp(-d2/alpha)))
    # = -alpha * (logsumexp(-d2/alpha) - log(n))
    nvals = len(all_d2)
    logsumexp_val = np.log(np.sum(np.exp(-all_d2 / alpha)))
    soft_r2 = -alpha * (logsumexp_val - np.log(nvals))
    
    # r_soft = sqrt(soft_r2) but we want to maximize r, so minimize -r_soft
    # Actually for the objective, let's just minimize -sqrt(soft_r2)
    # But sqrt is non-smooth at 0. Since soft_r2 should be positive, use:
    r_soft = np.sqrt(max(soft_r2, 1e-15))
    return -r_soft


def _make_7x7_plus1():
    """Create 7x7 grid (49 points) + best 50th point."""
    ticks = (np.arange(7) + 0.5) / 7.0 - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Find best position for 50th point on a fine grid
    best_50th = np.array([0.0, 0.0])
    best_min_dist = 0.0
    for y in np.linspace(-0.49, 0.49, 30):
        for x in np.linspace(-0.49, 0.49, 30):
            candidate = np.array([x, y])
            dists = np.linalg.norm(grid - candidate, axis=1)
            min_d = np.min(dists)
            # Also check wall distance
            wall_d = np.min(0.5 - np.abs(candidate))
            effective_d = min(min_d, 2.0 * wall_d)  # radius-limited
            if effective_d > best_min_dist:
                best_min_dist = effective_d
                best_50th = candidate
    
    return np.vstack([grid, best_50th])


def _nelder_mead_smooth(x0, alpha=40.0, maxiter=500, time_limit=30.0):
    """Run Nelder-Mead on the smooth surrogate."""
    t0 = time.time()
    
    def _obj(x):
        return _smooth_obj(x, alpha)
    
    try:
        result = minimize(_obj, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
        if time.time() - t0 > time_limit:
            return x0
        return result.x
    except Exception:
        return x0


def _lbfgsb_smooth(x0, alpha=40.0, time_limit=30.0):
    """Run L-BFGS-B on the smooth surrogate with bounds."""
    t0 = time.time()
    
    def _obj(x):
        return _smooth_obj(x, alpha)
    
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        result = minimize(_obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                                   'maxfun': 1000})
        if time.time() - t0 > time_limit:
            return x0
        return result.x
    except Exception:
        return x0


def _coordinate_polish(centers, time_limit):
    """
    Coordinate-wise polish: for each point, optimize its 2 coordinates
    using Nelder-Mead on the smooth objective with other points fixed.
    This is a structural alternative to the greedy direction moves.
    """
    t0 = time.time()
    centers = centers.copy()
    alpha = 60.0
    best_r = _radius(centers)
    
    for sweep in range(3):
        if time.time() - t0 > time_limit:
            break
        for i in range(N):
            if time.time() - t0 > time_limit:
                break
            # Fix all other points, optimize point i
            x0 = centers.ravel()
            
            def _obj_fixed(x):
                c = x.copy().reshape(N, 2)
                # Only point i is variable; others fixed
                c[i] = x[2*i:2*i+2]
                # Compute smooth objective but only varying point i
                # Pairwise: dist from i to all j != i
                d2_pairs = np.sum((c[i] - c[:i])**2, axis=1)
                if i < N-1:
                    d2_pairs = np.concatenate([d2_pairs, 
                                               np.sum((c[i] - c[i+1:])**2, axis=1)])
                # Wall
                wall_d2 = np.array([(0.5 - abs(c[i][0]))**2, 
                                    (0.5 - abs(c[i][1]))**2])
                all_d2 = np.concatenate([d2_pairs, wall_d2])
                
                nvals = len(all_d2)
                logsumexp_val = np.log(np.sum(np.exp(-all_d2 / alpha)))
                soft_r2 = -alpha * (logsumexp_val - np.log(nvals))
                r_soft = np.sqrt(max(soft_r2, 1e-15))
                return -r_soft
            
            try:
                res = minimize(_obj_fixed, x0[2*i:2*i+2], method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8,
                                       'adaptive': True})
                if time.time() - t0 > time_limit:
                    break
                new_pos = res.x
                new_pos = np.clip(new_pos, -0.499, 0.499)
                centers[i] = new_pos
            except Exception:
                pass
    
    return centers, _radius(centers)


def construct():
    start_time = time.time()
    
    # Primary initialization: 7x7 grid + best 50th point
    centers_init = _make_7x7_plus1()
    best_centers = centers_init.copy()
    best_r = _radius(centers_init)
    
    # Phase 1: Nelder-Mead on smooth surrogate (full 100-dim)
    time_left = DEADLINE - 10 - (time.time() - start_time)
    if time_left > 5:
        t0 = time.time()
        refined = _nelder_mead_smooth(centers_init.ravel(), alpha=40.0, 
                                       maxiter=300, time_limit=min(time_left, 40.0))
        refined = np.clip(refined.reshape(N, 2), -0.499, 0.499)
        r = _radius(refined)
        if r > best_r:
            best_r = r
            best_centers = refined
        time_left = DEADLINE - 10 - (time.time() - start_time)
    
    # Phase 2: L-BFGS-B on smooth surrogate (with bounds)
    if time_left > 5:
        t0 = time.time()
        refined = _lbfgsb_smooth(best_centers.ravel(), alpha=50.0,
                                  time_limit=min(time_left, 30.0))
        refined = np.clip(refined.reshape(N, 2), -0.499, 0.499)
        r = _radius(refined)
        if r > best_r:
            best_r = r
            best_centers = refined
        time_left = DEADLINE - 10 - (time.time() - start_time)
    
    # Phase 3: Coordinate-wise polish (Nelder-Mead per point)
    if time_left > 3:
        refined, r = _coordinate_polish(best_centers, time_limit=min(time_left, 40.0))
        if r > best_r:
            best_r = r
            best_centers = refined
        time_left = DEADLINE - 10 - (time.time() - start_time)
    
    # Phase 4: Another round of Nelder-Mead with tighter alpha (finer soft-min)
    if time_left > 5:
        t0 = time.time()
        refined = _nelder_mead_smooth(best_centers.ravel(), alpha=80.0,
                                       maxiter=200, time_limit=min(time_left, 30.0))
        refined = np.clip(refined.reshape(N, 2), -0.499, 0.499)
        r = _radius(refined)
        if r > best_r:
            best_r = r
            best_centers = refined
        time_left = DEADLINE - 10 - (time.time() - start_time)
    
    # Phase 5: Final coordinate polish
    if time_left > 3:
        refined, r = _coordinate_polish(best_centers, time_limit=min(time_left, 20.0))
        if r > best_r:
            best_r = r
            best_centers = refined
    
    # Safety: ensure valid (N, 2) finite array inside the square
    best_centers = np.asarray(best_centers, dtype=float)
    if best_centers.shape != (N, 2) or not np.all(np.isfinite(best_centers)):
        best_centers = _make_7x7_plus1()
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
