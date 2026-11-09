"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(pts):
    """Compute the actual minimum radius (exact, not soft)."""
    n = len(pts)
    # Wall distance: min over all points and all 6 walls
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    r_wall = np.min(wall)
    if r_wall < 0:
        return -1.0
    # Pair distances
    # Use a chunked approach to save memory
    min_pair = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        if len(d) > 0:
            min_pair = min(min_pair, np.min(d))
    r_pair = min_pair / 2.0
    return min(r_wall, r_pair)

def _soft_min_radius(pts, beta=200.0):
    """Soft-min surrogate for the radius using log-sum-exp.
    We want to maximize min(r_wall, r_pair).
    Use a soft-min over all constraints (wall and pair).
    """
    n = len(pts)
    # Wall constraints: for each point, distance to nearest wall
    # dist to wall = 0.5 - max(|x|,|y|,|z|)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    
    # Pair distances / 2
    pair_vals = []
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1) / 2.0
        if len(d) > 0:
            pair_vals.append(d)
    if pair_vals:
        pair_arr = np.concatenate(pair_vals)
    else:
        pair_arr = np.array([1.0])
    
    # Combine all constraints into one array
    all_vals = np.concatenate([wall_dists, pair_arr])
    
    # Soft min via log-sum-exp: -log-sum-exp(-beta * vals) / beta
    # soft_min = -log(sum(exp(-beta * v))) / beta
    neg_beta_vals = -beta * all_vals
    # For numerical stability
    max_val = np.max(neg_beta_vals)
    if max_val > 700:
        max_val = 700
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
    soft_min = -log_sum / beta
    return soft_min

def _soft_min_radius_grad(pts, beta=200.0):
    """Gradient of the soft-min radius surrogate."""
    n = len(pts)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    
    pair_vals = []
    pair_indices = []  # (i, j) for each pair
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1) / 2.0
        if len(d) > 0:
            pair_vals.append(d)
            for j in range(i+1, n):
                pair_indices.append((i, j))
    if pair_vals:
        pair_arr = np.concatenate(pair_vals)
    else:
        pair_arr = np.array([1.0])
    
    all_vals = np.concatenate([wall_dists, pair_arr])
    
    neg_beta_vals = -beta * all_vals
    max_val = np.max(neg_beta_vals)
    if max_val > 700:
        max_val = 700
    exp_vals = np.exp(neg_beta_vals - max_val)
    sum_exp = np.sum(exp_vals)
    
    # d(soft_min)/d(v_k) = -exp(-beta*v_k) / sum(exp(-beta*v_j))
    # = -exp_vals[k] / sum_exp
    weights = -exp_vals / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Wall gradient contribution
    # wall_dists[i] = 0.5 - max(|pts[i,0]|, |pts[i,1]|, |pts[i,2]|)
    # d(wall_dists[i])/d(pts[i,k]) = -sign(pts[i,k]) if pts[i,k] is the max
    for i in range(n):
        w = weights[i]
        if abs(w) < 1e-15:
            continue
        abs_pts = np.abs(pts[i])
        max_idx = np.argmax(abs_pts)
        # derivative of -max(|x|) w.r.t. x[k] = -sign(x[k]) * (1 if k==max_idx else 0)
        # So d(wall_dists[i])/d(pts[i,k]) = -sign(pts[i,k]) for k==max_idx
        s = np.sign(pts[i, max_idx])
        if s == 0:
            s = 1.0
        grad[i, max_idx] += w * (-s)
    
    # Pair gradient contribution
    # pair_arr[m] = |pts[i]-pts[j]|/2 for pair (i,j)
    # d(pair_arr[m])/d(pts[i]) = (pts[i]-pts[j]) / |pts[i]-pts[j]|
    # d(pair_arr[m])/d(pts[j]) = (pts[j]-pts[i]) / |pts[i]-pts[j]|
    # Note: pair_arr[m] = dist/2, so derivative is (pts[i]-pts[j])/dist (not /2)
    # Actually: d(|p_i - p_j|/2)/dp_i = (p_i - p_j)/|p_i - p_j|
    for m, (i, j) in enumerate(pair_indices):
        w = weights[n + m]
        if abs(w) < 1e-15:
            continue
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            continue
        unit = diff / dist
        grad[i] += w * unit
        grad[j] += w * (-unit)
    
    return grad

def _fcc_lattice_init():
    """Initialize with FCC-like lattice structure for better starting point."""
    # FCC lattice: points at (0,0,0) and face centers
    # For a unit cube [-0.5, 0.5]^3, we can use a grid with FCC offsets
    # Simple approach: use a 5x5x4 grid with some FCC-like perturbations
    # Actually, let's use a body-centered cubic inspired init
    # 
    # Better: use a 4x5x5 grid = 100 points
    # Or 5x5x4 = 100
    # Let's use a structured grid with small random perturbations
    m = 5
    # 5*5*4 = 100
    xs = np.linspace(-0.4, 0.4, m)
    ys = np.linspace(-0.4, 0.4, m)
    zs = np.linspace(-0.45, 0.45, 4)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small random perturbations to break symmetry
    rng = np.random.default_rng(42)
    grid = grid + rng.normal(0, 0.01, grid.shape)
    # Clip to stay inside
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial point: FCC-inspired lattice
    pts = _fcc_lattice_init()
    
    # Compute initial radius
    r0 = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = r0
    
    beta_values = [100.0, 200.0, 500.0]
    
    for beta in beta_values:
        if time.time() > deadline - 20.0:
            break
        
        def objective(x):
            pts_flat = x.reshape(-1, 3)
            return -_soft_min_radius(pts_flat, beta=beta)
        
        def gradient(x):
            pts_flat = x.reshape(-1, 3)
            return _soft_min_radius_grad(pts_flat, beta=beta).reshape(-1)
        
        x0 = pts.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(
                objective, x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < objective(x0):
                pts = result.x.reshape(-1, 3)
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 15.0:
            break
    
    # Final polish: use exact radius with a few more iterations
    if time.time() < deadline - 5.0:
        beta = 1000.0
        def objective2(x):
            pts_flat = x.reshape(-1, 3)
            return -_soft_min_radius(pts_flat, beta=beta)
        
        def gradient2(x):
            pts_flat = x.reshape(-1, 3)
            return _soft_min_radius_grad(pts_flat, beta=beta).reshape(-1)
        
        x0 = best_pts.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(
                objective2, x0,
                jac=gradient2,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < objective2(x0):
                pts2 = result.x.reshape(-1, 3)
                r2 = _compute_radius(pts2)
                if r2 > best_r:
                    best_r = r2
                    best_pts = pts2.copy()
        except Exception:
            pass
    
    # Final validity check
    if not np.all(np.isfinite(best_pts)):
        return _fcc_lattice_init()
    if np.any(np.abs(best_pts) > 0.5):
        best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
