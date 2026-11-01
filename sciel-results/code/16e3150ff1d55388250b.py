"""Packing 200 equal circles in a unit square: Force-Directed + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 140.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization with multiple aspect ratios."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)
    
    # Try a wider range of spacings and aspect ratios
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 25):
        for aspect in [1.0, 0.95, 1.05, 0.9, 1.1]:
            pts_list = []
            row = 0
            while True:
                y = -0.5 + s/2 + row * s * np.sqrt(3) / 2 * aspect
                if y > 0.5 - s/2:
                    break
                offset = (row % 2) * s / 2
                x = -0.5 + s/2 + offset
                while x <= 0.5 - s/2:
                    pts_list.append((x, y))
                    x += s
                row += 1
            if len(pts_list) < N:
                continue
            pts = np.array(pts_list[:N])
            if jitter_scale > 0:
                pts += rng.normal(0, jitter_scale, size=pts.shape)
                pts = np.clip(pts, -0.49, 0.49)
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

def _random_init(seed=0, n_clusters=10):
    """Generate a random clustered initialization for diversity."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    # Add some structure by clustering
    cluster_centers = rng.uniform(-0.4, 0.4, size=(n_clusters, 2))
    for i in range(N):
        c = i % n_clusters
        pts[i] = pts[i] * 0.5 + cluster_centers[c] * 0.5
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _force_directed(pts, time_budget):
    """
    Soft-contact dynamics: symplectic Euler with polynomial overlap penalty.
    Uses high-order potential to penalize circle overlaps, allowing the system
    to settle into a stable equilibrium through physical dynamics.
    """
    t0 = time.time()
    pos = pts.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    # Parameters
    dt = 0.002
    damping = 0.995
    overlap_power = 6  # High-order polynomial for soft contact
    boundary_power = 4
    k_overlap = 50.0
    k_boundary = 100.0
    neighbor_k = 20
    
    current_r = _score(pos)
    best_pos = pos.copy()
    best_r = current_r
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - t0 < time_budget:
        # Compute forces using KDTree for neighbor lookup
        tree = cKDTree(pos)
        dists, indices = tree.query(pos, k=neighbor_k + 1)
        
        forces = np.zeros_like(pos)
        
        # Pairwise soft-contact forces: F = -dU/dx where U ~ (max(0, r_target - d))^power
        r_target = max(current_r * 2.0, 0.05)  # Target separation
        for i in range(N):
            for j_idx in range(1, neighbor_k + 1):
                j = int(indices[i, j_idx])
                d = dists[i, j_idx]
                if d < 1e-10:
                    # Coincident: push randomly
                    forces[i] += np.random.randn(2) * 0.01
                    continue
                # Overlap amount (positive if too close)
                overlap = max(0.0, r_target - d)
                if overlap > 0:
                    # Force magnitude: polynomial in overlap
                    mag = k_overlap * overlap ** (overlap_power - 1)
                    diff = pos[i] - pos[j]
                    forces[i] += diff / d * mag
        
        # Boundary forces: push away from edges when too close
        d_x = 0.5 - np.abs(pos[:, 0])
        d_y = 0.5 - np.abs(pos[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        
        r_boundary_target = max(current_r, 0.03)
        close_to_boundary = boundary_dists < r_boundary_target
        if np.any(close_to_boundary):
            for i in np.where(close_to_boundary)[0]:
                # Determine which boundary is closest
                if d_x[i] < d_y[i]:
                    # Close to left/right boundary
                    sign_x = 1.0 if pos[i, 0] > 0 else -1.0
                    overlap_b = max(0.0, r_boundary_target - d_x[i])
                    if overlap_b > 0:
                        forces[i, 0] += sign_x * k_boundary * overlap_b ** (boundary_power - 1)
                else:
                    # Close to top/bottom boundary
                    sign_y = 1.0 if pos[i, 1] > 0 else -1.0
                    overlap_b = max(0.0, r_boundary_target - d_y[i])
                    if overlap_b > 0:
                        forces[i, 1] += sign_y * k_boundary * overlap_b ** (boundary_power - 1)
        
        # Symplectic Euler integration
        vel = vel * damping + forces * dt
        pos = pos + vel * dt
        pos = np.clip(pos, -0.499, 0.499)
        
        # Evaluate and track best
        new_r = _score(pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = pos.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Adaptive timestep: reduce if oscillating
        if no_improve_count > max_no_improve:
            dt *= 0.8
            damping *= 0.99
            if dt < 1e-5:
                break
        
        # Update target based on current best
        current_r = best_r
    
    return best_pos

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B using numerically stable log-sum-exp.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Numerically stable log-sum-exp for soft min
    # soft_min = -1/alpha * log(sum(exp(-alpha * d)))
    # Use max trick for numerical stability
    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha
    
    # We want to maximize min distance, so minimize -min distance.
    return -soft_min_val

def _optimize_lbfgs(pts, time_budget):
    """
    Run L-BFGS-B optimization with multi-temperature annealing schedule.
    Uses tighter tolerances in later phases for higher precision.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Multi-temperature schedule: coarse to fine, with more iterations at high alpha
    phases = [
        {"alpha": 8.0, "max_iter": 200, "gtol": 1e-3},
        {"alpha": 25.0, "max_iter": 250, "gtol": 1e-5},
        {"alpha": 80.0, "max_iter": 300, "gtol": 1e-6},
        {"alpha": 250.0, "max_iter": 300, "gtol": 1e-7},
        {"alpha": 800.0, "max_iter": 250, "gtol": 1e-8},
        {"alpha": 2000.0, "max_iter": 200, "gtol": 1e-9},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.4999, 0.4999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12, 'maxls': 50})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in the minimum distance constraints.
    Uses directed perturbations for more effective search.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(54321)
    step_size = 0.002
    min_step = 1e-7
    stagnation_count = 0
    
    while time.time() - t0 < time_budget:
        # Compute constraints
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)
        
        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        current_r = min(min_boundary, min_pair / 2.0)
        
        # Identify active set: points with distance to boundary < current_r * 1.05
        # and pairs with distance < 2 * current_r * 1.05
        active_idx = set()
        
        tight_boundary = np.where(boundary_dists < current_r * 1.05)[0]
        active_idx.update(tight_boundary.tolist())
        
        # Find tight pairs using KDTree
        tree = cKDTree(best_pts)
        pairs, dists = tree.query(best_pts, k=5)
        for i in range(N):
            for j in range(1, 5):
                if dists[i, j] < 2.0 * current_r * 1.05:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))
        
        if not active_idx:
            break
            
        active_list = list(active_idx)
        n_active = len(active_list)
        
        # Try perturbing active points with multiple strategies
        improved = False
        
        # Strategy 1: Random perturbations
        for _ in range(15):
            perturb = np.zeros((N, 2))
            perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
            trial = best_pts + perturb
            trial = np.clip(trial, -0.4999, 0.4999)
            
            trial_r = _score(trial)
            if trial_r > best_r:
                best_pts = trial
                best_r = trial_r
                improved = True
                break
        
        # Strategy 2: Directed perturbations away from nearest neighbors
        if not improved:
            for i in active_list:
                nearest_dist, nearest_idx = tree.query(best_pts[i], k=2)
                for j_idx in range(1, 2):
                    j = int(nearest_idx[j_idx])
                    if j == i:
                        continue
                    # Push i away from j
                    diff = best_pts[i] - best_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        perturb = np.zeros((N, 2))
                        perturb[i] = diff / norm * step_size
                        trial = best_pts + perturb
                        trial = np.clip(trial, -0.4999, 0.4999)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            step_size *= 0.95
            stagnation_count = 0
        else:
            step_size *= 0.85
            stagnation_count += 1
            if step_size < min_step or stagnation_count > 50:
                break
                
    return best_pts

def construct():
    t_start = time.time()
    
    # 1. Initialization - try diverse starting points
    best_pts = None
    best_r = 0.0
    
    inits = [
        _hex_init(jitter_scale=0.0, seed=0),
        _hex_init(jitter_scale=0.003, seed=1),
        _hex_init(jitter_scale=0.008, seed=2),
        _hex_init(jitter_scale=0.015, seed=3),
        _random_init(seed=42),
        _random_init(seed=123),
    ]
    
    for pts in inits:
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    # 2. Soft-Contact Dynamics - symplectic integrator with polynomial overlap
    remaining = DEADLINE - time.time()
    if remaining > 15:
        fd_budget = min(remaining * 0.25, 20.0)
        fd_pts = _force_directed(best_pts, fd_budget)
        fd_r = _score(fd_pts)
        if fd_r > best_r:
            best_r = fd_r
            best_pts = fd_pts.copy()
            
    # 3. L-BFGS-B Polish with multi-temperature annealing
    remaining = DEADLINE - time.time()
    if remaining > 25:
        lbfgs_budget = min(remaining * 0.45, 55.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # 4. Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 10:
        polish_budget = min(remaining * 0.35, 25.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # 5. Final high-precision L-BFGS-B pass
    remaining = DEADLINE - time.time()
    if remaining > 20:
        lbfgs_budget = min(remaining * 0.4, 35.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # 6. Final Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.6, 20.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # 7. Final safety check and tiny polish
    remaining = DEADLINE - time.time()
    if remaining > 2:
        # Very short final polish
        polished_pts = _active_set_polish(best_pts, min(remaining * 0.5, 5.0))
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
