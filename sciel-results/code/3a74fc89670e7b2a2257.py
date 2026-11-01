"""Packing 200 equal circles in a unit square via multi-temperature soft-min + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
DEADLINE = 135.0  # seconds, safety margin before 150s kill


def _score(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0
    # Wall distances
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    wall_min = np.min(walls)
    
    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_min = np.min(dists[:, 1]) / 2.0
    
    return min(wall_min, pair_min)


def _hex_init(n):
    """Generate initial positions using a hexagonal lattice pattern."""
    best = None
    best_r = -1
    
    for orientation in [0, np.pi/6, np.pi/3]:
        m = int(np.ceil(np.sqrt(n)))
        
        for scale in np.linspace(0.85, 1.15, 20):
            s = scale / m
            dx = s
            dy = s * np.sqrt(3) / 2
            
            pts = []
            for i in range(m):
                for j in range(m):
                    x = (j - (m-1)/2) * dx
                    y = (i - (m-1)/2) * dy
                    if i % 2 == 1:
                        x += dx / 2
                    pts.append([x, y])
            
            pts = np.array(pts[:n])
            
            if orientation != 0:
                c, s_rot = np.cos(orientation), np.sin(orientation)
                R = np.array([[c, -s_rot], [s_rot, c]])
                pts = pts @ R.T
            
            max_abs = np.max(np.abs(pts))
            if max_abs > 0.45:
                pts = pts * (0.45 / max_abs)
            
            r = _score(pts)
            if r > best_r:
                best_r = r
                best = pts.copy()
    
    return best if best is not None else _grid_init(n)


def _grid_init(n):
    """Fallback square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _random_init(n, seed=42):
    """Random initialization with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    pts = []
    min_dist = 0.02
    
    for _ in range(n * 50):
        p = rng.uniform(-0.48, 0.48, 2)
        if len(pts) > 0:
            pts_arr = np.array(pts)
            dists = np.linalg.norm(pts_arr - p, axis=1)
            if np.min(dists) < min_dist:
                continue
        pts.append(p)
        if len(pts) == n:
            break
    
    if len(pts) < n:
        # Fill remaining randomly
        while len(pts) < n:
            p = rng.uniform(-0.48, 0.48, 2)
            pts.append(p)
    
    return np.array(pts)


def _soft_min_neg(x, k, cutoff=0.15):
    """
    Negative of soft-min surrogate for the max-min radius problem.
    Uses only nearby pairs for efficiency.
    """
    pts = x.reshape(-1, 2)
    
    # Wall distances
    walls = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    
    # Pair distances - use only nearby pairs for efficiency
    tree = cKDTree(pts)
    pairs = tree.query_pairs(cutoff)
    
    if len(pairs) == 0:
        pair_dists = np.array([1.0])
    else:
        pair_dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, walls])
    
    # Soft min approximation using log-sum-exp
    min_val = np.min(all_dists)
    if min_val < 1e-10:
        return 1000.0
    
    # Scaled values for numerical stability
    scaled = all_dists / min_val
    # Use a numerically stable log-sum-exp
    max_scaled = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(-(scaled - max_scaled)))) + max_scaled
    soft_min = min_val * (1 - log_sum / k)
    
    # We want to maximize soft_min, so minimize negative
    return -soft_min


def _optimize_lbfgs_multi_temp(centers, time_limit=30.0):
    """
    Multi-temperature L-BFGS-B optimization on the soft-min surrogate.
    Coarse to fine temperature schedule.
    """
    start_time = time.time()
    pts = centers.copy()
    current_r = _score(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    # Temperature levels: coarse to fine
    k_levels = [15.0, 40.0, 100.0, 300.0, 800.0]
    
    for k in k_levels:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Multiple restarts at coarser temperatures
        n_restarts = 3 if k < 50 else 1
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # For restarts, add small perturbation
            if restart > 0:
                x0 = pts.flatten() + np.random.normal(0, 0.005, 2*N)
            else:
                x0 = pts.flatten()
            
            def obj(x):
                return _soft_min_neg(x, k)
            
            try:
                result = minimize(
                    obj, x0,
                    method='L-BFGS-B',
                    bounds=[(-0.499, 0.499)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                new_pts = result.x.reshape(-1, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_r = _score(new_pts)
                
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    
            except Exception:
                pass
    
    return best_pts, best_r


def _force_directed(centers, r_target, time_limit=20.0):
    """
    Force-directed relaxation: move overlapping circles apart.
    Iteratively apply repulsive forces until no overlaps or time limit.
    """
    start_time = time.time()
    pts = centers.copy()
    
    for iteration in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Find all overlapping pairs
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2 * r_target + 0.01)
        
        if len(pairs) == 0:
            # Check walls
            walls = np.minimum(
                np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
                np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
            )
            if np.min(walls) >= r_target - 1e-9:
                break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small push
                angle = np.random.uniform(0, 2 * np.pi)
                diff = np.array([np.cos(angle), np.sin(angle)])
                dist = 1e-10
            
            # Repulsive force: proportional to overlap
            overlap = 2 * r_target - dist
            if overlap > 0:
                force = diff / dist * overlap * 0.5
                forces[i] += force
                forces[j] -= force
        
        # Wall forces
        for i in range(N):
            for axis in range(2):
                if pts[i, axis] + r_target > 0.5:
                    forces[i, axis] -= (pts[i, axis] + r_target - 0.5) * 0.5
                if pts[i, axis] - r_target < -0.5:
                    forces[i, axis] += (r_target - pts[i, axis] - 0.5) * 0.5
        
        # Apply forces with damping
        step_size = 0.1
        pts += forces * step_size
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts


def _active_set_polish(centers, r_target, time_limit=15.0):
    """
    Active set method: iteratively tighten constraints for the most critical pairs.
    """
    start_time = time.time()
    pts = centers.copy()
    
    for outer_iter in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Find critical pairs (those with distance close to 2*r_target)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2 * r_target + 0.005)
        
        if len(pairs) == 0:
            # Only wall constraints
            walls = np.minimum(
                np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
                np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
            )
            if np.min(walls) >= r_target - 1e-9:
                break
            # Optimize with just wall constraints
            def obj_wall(x):
                x = x.reshape(-1, 2)
                w = np.minimum(
                    np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
                    np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
                )
                return -np.min(w)
            
            try:
                result = minimize(
                    obj_wall, pts.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.499, 0.499)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                pts = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
            except:
                pass
            continue
        
        # Select the most critical pairs (closest to 2*r_target)
        pair_dists = []
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            pair_dists.append((d, i, j))
        pair_dists.sort()
        
        # Use the most critical pairs (smallest distances)
        max_pairs = min(100, len(pair_dists))
        critical_pairs = [(i, j) for d, i, j in pair_dists[:max_pairs]]
        
        # Build constraint function
        def constraints_active(x):
            x = x.reshape(-1, 2)
            vals = []
            for i, j in critical_pairs:
                diff = x[i] - x[j]
                vals.append(np.dot(diff, diff) - (2 * r_target) ** 2)
            return np.array(vals)
        
        # Optimize: minimize sum of squared violations
        def obj_violation(x):
            x = x.reshape(-1, 2)
            vals = constraints_active(x)
            return np.sum(np.maximum(0, vals) ** 2)
        
        try:
            result = minimize(
                obj_violation, pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        except:
            break
    
    return pts


def _high_precision_polish(centers, r_target, time_limit=10.0):
    """
    Very tight final polish using multiple optimization techniques.
    """
    start_time = time.time()
    pts = centers.copy()
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Multiple passes with decreasing target radius tolerance
    for tol in [1e-8, 1e-10, 1e-12]:
        if time.time() - start_time > time_limit - 2:
            break
        
        # Active set polish
        polished = _active_set_polish(pts, r_target - tol, time_limit=3.0)
        r_new = _score(polished)
        if r_new > best_r:
            best_r = r_new
            best_pts = polished.copy()
            pts = polished.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    
    # Initial position: hexagonal lattice
    centers = _hex_init(N)
    current_r = _score(centers)
    
    best_centers = centers.copy()
    best_r = current_r
    
    # Phase 1: Multi-temperature L-BFGS-B optimization
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 30:
        optimized, r_opt = _optimize_lbfgs_multi_temp(centers, time_limit=min(40, remaining - 10))
        if r_opt > best_r:
            best_r = r_opt
            best_centers = optimized.copy()
    
    # Phase 2: Force-directed relaxation
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        r_target = best_r * 1.02
        relaxed = _force_directed(best_centers, r_target, time_limit=min(15, remaining - 5))
        r_relaxed = _score(relaxed)
        if r_relaxed > best_r:
            best_r = r_relaxed
            best_centers = relaxed.copy()
    
    # Phase 3: Active set polish
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        polished, r_polished = _active_set_polish(best_centers, best_r, time_limit=min(15, remaining - 5))
        if _score(polished) > best_r:
            best_r = _score(polished)
            best_centers = polished.copy()
    
    # Phase 4: High precision final polish
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 8:
        final_pts, r_final = _high_precision_polish(best_centers, best_r, time_limit=min(10, remaining - 3))
        if r_final > best_r:
            best_r = r_final
            best_centers = final_pts.copy()
    
    # Final validation and clipping
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
    
    # Final score check
    final_r = _score(best_centers)
    if final_r < best_r:
        # Clip may have reduced radius; use clipped version anyway
        pass
    
    return best_centers
