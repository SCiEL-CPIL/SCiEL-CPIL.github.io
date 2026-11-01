"""Combined 50-sphere packing: repulsion annealing + smooth surrogate + exact polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 135.0  # Safety margin before 150s limit

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _neg_radius(x):
    """Objective: negative radius (to be minimized)."""
    pts = x.reshape(N, 3)
    return -_compute_radius(pts)

def _soft_min_radius(x, beta=50.0):
    """
    Smooth surrogate for negative radius using log-sum-exp soft-min.
    Returns a differentiable approximation of -min(constraints).
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Upper triangle indices for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow (shifted <= 0)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def _repulsion_anneal(pts, n_steps=80, lr_start=0.008, lr_end=0.0005):
    """
    Repulsion annealing: iteratively push spheres apart using gradient of repulsive potential.
    Vectorized for speed. Helps escape shallow local minima before gradient-based optimization.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 35.0:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        # Compute forces (vectorized)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
        np.fill_diagonal(dists, 1e6)  # Avoid division by zero on diagonal
        
        # Force magnitude: 1/dist^2, applied along direction
        # Force on i from j: (1/dist^2) * (pts[i]-pts[j])/dist = diff[i,j] / dist^3
        inv_dist3 = 1.0 / (dists ** 3 + 1e-8)
        # Zero out diagonal
        np.fill_diagonal(inv_dist3, 0)
        
        # forces[i] = sum_j inv_dist3[i,j] * diff[i,j]
        forces += np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion: push away from walls
        for k in range(3):
            pos = pts[:, k]
            wall_dist_pos = 0.5 - pos
            wall_dist_neg = 0.5 + pos
            
            # Force from positive wall (push left if close to right wall)
            force_pos = np.where(wall_dist_pos < 0.15, -1.0 / (wall_dist_pos + 0.01), 0.0)
            # Force from negative wall (push right if close to left wall)
            force_neg = np.where(wall_dist_neg < 0.15, 1.0 / (wall_dist_neg + 0.01), 0.0)
            
            forces[:, k] += force_pos + force_neg
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
        
        # Track best
        cur_r = _compute_radius(pts)
        if cur_r > best_r:
            best_r = cur_r
            best_pts = pts.copy()
    
    return best_pts

def _metropolis_mc(pts, n_steps=1500, temp_start=0.04, temp_end=0.003):
    """
    Metropolis Monte Carlo to escape local minima.
    Tries random perturbations and accepts if radius improves or with probability exp(-delta/temp).
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        
        # Temperature annealing
        t = step / max(1, n_steps - 1)
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Generate perturbation: move a random subset of points
        n_move = np.random.randint(1, 6)  # Move 1-5 points
        idx = np.random.choice(N, size=n_move, replace=False)
        delta = np.random.normal(0, 0.015, size=(n_move, 3))
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        # Accept if better, or with Metropolis probability
        if new_r >= best_r:
            pts = new_pts
            best_pts = new_pts.copy()
            best_r = new_r
        else:
            delta_r = best_r - new_r
            if np.random.random() < np.exp(-delta_r / max(temp, 1e-10)):
                pts = new_pts
    
    return best_pts

def _init_grid():
    """5x5x2 grid with small jitter."""
    xs = (np.arange(5) + 0.5) / 5.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(2) + 0.5) / 2.0 - 0.5
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers += np.random.normal(0, 0.001, size=(N, 3))
    return np.clip(centers, -0.49, 0.49)

def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _local_search(pts, max_iter=150):
    """Coordinate-wise steepest ascent on exact radius."""
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        
        improved = False
        for i in range(N):
            for dim in range(3):
                for sign in [1, -1]:
                    for step_size in [0.004, 0.0015, 0.0005]:
                        new_pts = pts.copy()
                        new_pts[i, dim] += sign * step_size
                        new_pts[i, dim] = np.clip(new_pts[i, dim], -0.499, 0.499)
                        new_r = _compute_radius(new_pts)
                        if new_r > best_r + 1e-12:
                            pts = new_pts
                            best_r = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                continue
        
        if not improved:
            break
    
    return pts

def _refine_full(pts):
    """Full refinement pipeline: repulsion anneal -> smooth surrogate -> MC -> local search -> NM polish."""
    x_2d = pts.copy().astype(np.float64)
    
    # Phase 1: Repulsion annealing
    if time.time() - START_TIME < TIME_LIMIT - 50.0:
        x_2d = _repulsion_anneal(x_2d, n_steps=80)
    
    # Phase 2: Smooth surrogate with increasing beta
    x = x_2d.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    betas = [15.0, 40.0, 100.0, 250.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 25.0:
            break
        try:
            def obj(x):
                return _soft_min_radius(x, beta)
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    
    # Phase 3: Metropolis MC
    if time.time() - START_TIME < TIME_LIMIT - 30.0:
        x_2d = _metropolis_mc(x_2d, n_steps=1000)
    
    # Phase 4: Local search
    if time.time() - START_TIME < TIME_LIMIT - 15.0:
        x_2d = _local_search(x_2d, max_iter=100)
    
    # Phase 5: Nelder-Mead polish
    if time.time() - START_TIME < TIME_LIMIT - 10.0:
        try:
            res = minimize(_neg_radius, x_2d, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                x_2d = res.x
        except Exception:
            pass
    
    return np.clip(x_2d, -0.499, 0.499)

def _final_polish(pts):
    """High-precision final polish with tighter tolerances."""
    x_2d = pts.copy().astype(np.float64)
    best_r = _compute_radius(x_2d)
    
    # Multiple rounds of Nelder-Mead with decreasing tolerances
    for atol in [1e-10, 1e-12, 1e-14]:
        if time.time() - START_TIME > TIME_LIMIT - 5.0:
            break
        try:
            res = minimize(_neg_radius, x_2d, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': atol, 'fatol': atol, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                new_r = _compute_radius(res.x)
                if new_r > best_r:
                    x_2d = res.x
                    best_r = new_r
        except Exception:
            pass
    
    # Final local search with very small steps
    if time.time() - START_TIME < TIME_LIMIT - 3.0:
        for it in range(40):
            if time.time() - START_TIME > TIME_LIMIT - 3.0:
                break
            improved = False
            for i in range(N):
                for dim in range(3):
                    for sign in [1, -1]:
                        for step_size in [0.0004, 0.00015, 0.00005]:
                            new_pts = x_2d.copy()
                            new_pts[i, dim] += sign * step_size
                            new_pts[i, dim] = np.clip(new_pts[i, dim], -0.499, 0.499)
                            new_r = _compute_radius(new_pts)
                            if new_r > best_r + 1e-14:
                                x_2d = new_pts
                                best_r = new_r
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    continue
            if not improved:
                break
    
    return np.clip(x_2d, -0.499, 0.499)

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Diverse initializations
    inits = [
        _init_grid(),
        _init_fcc(),
        _init_bcc(),
        _init_random(),
    ]
    
    # Add more random inits if time permits
    while time.time() - START_TIME < TIME_LIMIT - 55.0:
        inits.append(_init_random())
    
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 35.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        refined = _refine_full(init_pts)
        refined = np.clip(refined, -0.499, 0.499)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
    
    # Final polish on best solution if time remains
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 10.0:
        # Local search first
        if time.time() - START_TIME < TIME_LIMIT - 15.0:
            polished = _local_search(best_pts, max_iter=50)
            r_ls = _compute_radius(polished)
            if r_ls > best_r:
                best_r = r_ls
                best_pts = polished
        
        # Then high-precision final polish
        if time.time() - START_TIME < TIME_LIMIT - 5.0:
            polished = _final_polish(best_pts)
            r2 = _compute_radius(polished)
            if r2 > best_r:
                best_r = r2
                best_pts = polished
    
    if best_pts is None:
        best_pts = _init_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
