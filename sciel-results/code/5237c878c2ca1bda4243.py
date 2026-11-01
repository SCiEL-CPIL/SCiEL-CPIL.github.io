"""Packing 200 equal circles in a unit square: Force-Directed + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill


def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances using pdist
    dists = pdist(pts)
    min_dist = np.min(dists)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _hex_init():
    """Generate a hexagonal lattice initialization."""
    best_pts = None
    best_r = 0
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 15):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s / 2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + offset
            while x <= 0.5 - s / 2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts


def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _force_directed(pts, max_iter=500, dt=0.001, temp=0.01):
    """
    Force-directed relaxation: push circles apart and keep them inside the square.
    Uses repulsive forces between all pairs and attractive forces toward center.
    """
    pts = pts.copy()
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 2:
            break
        
        # Compute pairwise distances
        dists = pdist(pts)
        # Also compute distance to boundary for each point
        ax = np.abs(pts[:, 0])
        ay = np.abs(pts[:, 1])
        boundary_dists = np.minimum(0.5 - ax, 0.5 - ay)
        
        # Target radius estimate
        r_target = min(np.min(boundary_dists), np.min(dists) / 2.0)
        r_target = max(r_target, 0.01)
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsive forces (only for close pairs)
        n_pairs = len(dists)
        # Vectorize: for each pair, if dist < 2*r_target, push apart
        # We need to map pdist indices back to pairs
        # For efficiency, only consider pairs with dist < 3*r_target
        close_mask = dists < 3.0 * r_target
        if np.any(close_mask):
            # Get pair indices
            i_idx, j_idx = np.triu_indices(N, k=1)
            close_i = i_idx[close_mask]
            close_j = j_idx[close_mask]
            close_dists = dists[close_mask]
            
            # Avoid division by zero
            close_dists_safe = np.maximum(close_dists, 1e-10)
            # Force magnitude: repulsive, stronger when closer
            force_mag = np.maximum(0, (2.0 * r_target - close_dists) / close_dists_safe)
            force_mag = force_mag * 10.0  # Scale factor
            
            # Direction: from j to i
            diff = pts[close_i] - pts[close_j]
            norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
            norm_diff = np.maximum(norm_diff, 1e-10)
            unit_diff = diff / norm_diff
            
            # Apply forces
            for k in range(len(close_i)):
                forces[close_i[k]] += force_mag[k] * unit_diff[k]
                forces[close_j[k]] -= force_mag[k] * unit_diff[k]
        
        # Boundary forces: push inward if too close to boundary
        for i in range(N):
            # x-boundary
            if pts[i, 0] < -0.5 + r_target:
                forces[i, 0] += (r_target - (0.5 - abs(pts[i, 0]))) * 50.0
            elif pts[i, 0] > 0.5 - r_target:
                forces[i, 0] -= (r_target - (0.5 - abs(pts[i, 0]))) * 50.0
            # y-boundary
            if pts[i, 1] < -0.5 + r_target:
                forces[i, 1] += (r_target - (0.5 - abs(pts[i, 1]))) * 50.0
            elif pts[i, 1] > 0.5 - r_target:
                forces[i, 1] -= (r_target - (0.5 - abs(pts[i, 1]))) * 50.0
        
        # Add small random perturbation to escape local minima
        if temp > 0.001:
            forces += rng.normal(0, temp, size=pts.shape)
            temp *= 0.95
        
        # Update positions
        pts += dt * forces
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts


def _soft_min_neg(x, alpha):
    """
    Soft-min objective: -soft_min(all distances).
    x: flat array of size 2N.
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    pairwise_dists = pdist(pts)
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min using log-sum-exp trick
    m = np.min(all_dists)
    log_sum = m + np.log(np.sum(np.exp(-alpha * (all_dists - m))))
    soft_min_val = -(1.0 / alpha) * log_sum
    
    return -soft_min_val


def _optimize_lbfgs(pts, time_budget, alphas=[20.0, 50.0, 100.0, 200.0]):
    """
    Run L-BFGS-B optimization with increasing alpha (sharper soft-min).
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    for alpha in alphas:
        if time.time() - t0 > time_budget:
            break
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            x0 = res.x
        except Exception:
            pass
    
    return x0.reshape(N, 2)


def _active_set_polish(pts, max_iter=100):
    """
    Active set polish: identify the critical pairs (those achieving min distance)
    and the critical boundary constraints, then solve a QP to push them apart.
    """
    pts = pts.copy()
    r_current = _score(pts)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 2:
            break
        
        # Compute all pairwise distances and boundary distances
        dists = pdist(pts)
        ax = np.abs(pts[:, 0])
        ay = np.abs(pts[:, 1])
        boundary_dists = np.minimum(0.5 - ax, 0.5 - ay)
        
        # Find the minimum distance
        min_pair_dist = np.min(dists)
        min_boundary = np.min(boundary_dists)
        r = min(min_boundary, min_pair_dist / 2.0)
        
        if r <= 0:
            break
        
        # Identify active constraints
        # Pairwise: pairs with dist <= r * (1 + eps)
        eps = 1e-6
        active_pair_mask = dists <= min_pair_dist + eps
        active_boundary_mask = boundary_dists <= min_boundary + eps
        
        if not np.any(active_pair_mask) and not np.any(active_boundary_mask):
            break
        
        # Get active pair indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        active_i = i_idx[active_pair_mask]
        active_j = j_idx[active_pair_mask]
        
        # Simple gradient ascent on the min distance
        # For each active pair, push apart
        forces = np.zeros_like(pts)
        
        for i, j in zip(active_i, active_j):
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                forces[i] += direction
                forces[j] -= direction
        
        # For active boundary constraints, push inward
        for i in np.where(active_boundary_mask)[0]:
            # Determine which boundary is active
            if pts[i, 0] < 0 and (0.5 + pts[i, 0]) <= min_boundary + eps:
                forces[i, 0] += 1.0  # push right
            elif pts[i, 0] > 0 and (0.5 - pts[i, 0]) <= min_boundary + eps:
                forces[i, 0] -= 1.0  # push left
            if pts[i, 1] < 0 and (0.5 + pts[i, 1]) <= min_boundary + eps:
                forces[i, 1] += 1.0  # push up
            elif pts[i, 1] > 0 and (0.5 - pts[i, 1]) <= min_boundary + eps:
                forces[i, 1] -= 1.0  # push down
        
        # Normalize and apply small step
        force_norm = np.linalg.norm(forces)
        if force_norm > 1e-10:
            step = 0.001
            pts += step * forces / force_norm
            pts = np.clip(pts, -0.5, 0.5)
        
        # Check if we improved
        new_r = _score(pts)
        if new_r <= r_current:
            # Revert
            pts -= step * forces / force_norm
            pts = np.clip(pts, -0.5, 0.5)
            break
        
        r_current = new_r
    
    return pts


def _high_precision_polish(pts, time_budget=10.0):
    """
    Final high-precision polish using very sharp soft-min and tight tolerances.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Use very high alpha for sharpness
    for alpha in [200.0, 500.0, 1000.0]:
        if time.time() - t0 > time_budget:
            break
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            x0 = res.x
        except Exception:
            pass
    
    return x0.reshape(N, 2)


def construct():
    t_start = time.time()

    # Phase 1: Generate diverse initializations
    initializers = [
        ("hex", _hex_init),
        ("rand0", lambda: _random_init(0)),
        ("rand1", lambda: _random_init(1)),
    ]

    best_pts = None
    best_r = 0.0

    for name, init_fn in initializers:
        if time.time() > DEADLINE - 15:
            break

        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

        # Phase 2: Force-directed relaxation
        remaining = DEADLINE - time.time()
        if remaining < 10:
            break

        fd_time = min(remaining * 0.2, 15.0)
        fd_pts = _force_directed(pts, max_iter=300, dt=0.001, temp=0.01)
        fd_r = _score(fd_pts)
        if fd_r > best_r:
            best_r = fd_r
            best_pts = fd_pts.copy()

        # Phase 3: L-BFGS-B optimization with soft-min
        remaining = DEADLINE - time.time()
        if remaining < 10:
            break

        lbfgs_time = min(remaining * 0.3, 25.0)
        opt_pts = _optimize_lbfgs(fd_pts, lbfgs_time, alphas=[20.0, 50.0, 100.0])
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # Phase 4: Active set polish on the best result
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        as_pts = _active_set_polish(best_pts, max_iter=50)
        as_r = _score(as_pts)
        if as_r > best_r:
            best_r = as_r
            best_pts = as_pts.copy()

    # Phase 5: High-precision polish
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        hp_time = min(remaining * 0.5, 15.0)
        hp_pts = _high_precision_polish(best_pts, hp_time)
        hp_pts = np.clip(hp_pts, -0.5, 0.5)
        hp_r = _score(hp_pts)
        if hp_r > best_r:
            best_r = hp_r
            best_pts = hp_pts.copy()

    if best_pts is None:
        best_pts = _hex_init()

    return best_pts
