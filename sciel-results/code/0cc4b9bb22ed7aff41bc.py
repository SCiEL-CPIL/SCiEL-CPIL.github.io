"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _hex_lattice_init(n, rotation=0.0, seed=42, spacing_factor=0.95):
    """Generate hexagonal lattice points and select n closest to center."""
    rng = np.random.RandomState(seed)
    # Estimate spacing for ~n points in unit disk
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    s *= spacing_factor
    
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        # Rotate x,y columns
        pts_xy = points[:, :2]
        points[:, :2] = pts_xy @ R.T
        points[:, 2] = np.linalg.norm(points[:, :2], axis=1)
        
    # Sort by distance from center, take closest n
    idx = np.argsort(points[:, 2])
    selected = points[idx[:n]][:, :2]
    
    if len(selected) < n:
        # Fallback: fill remaining randomly
        remaining = n - len(selected)
        extra = rng.randn(remaining, 2) * 0.3
        selected = np.vstack([selected, extra])
    
    # Add small random noise to break symmetry
    noise = rng.randn(n, 2) * s * 0.01
    selected = selected + noise
    
    return selected


def _radial_ring_init(n, seed=42):
    """Generate points on concentric rings."""
    rng = np.random.RandomState(seed)
    points = []
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    n_rings = int(1.0 / s) + 1
    
    for r_idx in range(n_rings):
        r = (r_idx + 0.5) * s
        if r >= 1.0:
            break
        n_on_ring = max(1, int(2 * np.pi * r / s))
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        while len(points) < n:
            r = np.sqrt(rng.random())
            a = rng.random() * 2 * np.pi
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
    elif len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:n]]
    
    points += rng.normal(0, s * 0.01, size=points.shape)
    return points


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)


def _soft_min_objective_and_grad(pts, beta):
    """
    Compute negative soft-min objective and its gradient.
    Returns (value, grad).
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    radii_sq = np.sum(pts**2, axis=1)
    radii = np.sqrt(radii_sq)
    boundary_dists = 1.0 - radii
    
    # Constraint values
    pair_vals = dists / 2.0
    bound_vals = boundary_dists
    
    # For pairs, we take the upper triangle
    ii, jj = np.triu_indices(n, k=1)
    pair_vals_upper = pair_vals[ii, jj]
    
    # Combine all values
    all_vals = np.concatenate([pair_vals_upper, bound_vals])
    
    # Clip very small values to avoid log(0) or overflow in exp
    all_vals = np.maximum(all_vals, 1e-12)
    
    # Compute log-sum-exp of -beta * all_vals
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_val + np.log(sum_exp)
    
    # Objective: -softmin = - ( -1/beta * log_sum_exp ) = log_sum_exp / beta
    obj_val = log_sum_exp / beta
    
    # Compute weights p_k = exp(-beta * v_k) / sum(exp(-beta * v_j))
    weights = exp_vals / sum_exp
    
    # Gradient computation
    grad = np.zeros_like(pts)
    
    # 1. Pairwise contributions
    dists_upper = dists[ii, jj]
    safe_dists = np.maximum(dists_upper, 1e-10)
    
    w_pairs = weights[:len(ii)]
    
    A = np.zeros((n, n))
    A[ii, jj] = w_pairs / (2.0 * safe_dists)
    A[jj, ii] = A[ii, jj]
    
    row_sums = np.sum(A, axis=1)
    
    A_P = A @ pts
    grad_pairs = - pts * row_sums[:, np.newaxis] + A_P
    
    # 2. Boundary contributions
    w_bounds = weights[len(ii):]
    safe_radii = np.maximum(radii, 1e-10)
    grad_bounds = w_bounds[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    
    return obj_val, grad


def _local_polish(pts, beta=10000, max_iter=500, time_limit=10.0):
    """
    Perform local optimization on the current points using L-BFGS-B.
    This is a refined polish step.
    """
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            raise ValueError("timeout")
        x = x.reshape(N, 2)
        val, _ = _soft_min_objective_and_grad(x, beta)
        return val
        
    def grad(x):
        if time.time() - start_time > time_limit:
            raise ValueError("timeout")
        x = x.reshape(N, 2)
        _, g = _soft_min_objective_and_grad(x, beta)
        return g.ravel()
        
    try:
        result = minimize(
            obj,
            pts.ravel(),
            jac=grad,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': max_iter * 2
            }
        )
        new_pts = result.x.reshape(N, 2)
        
        # Check validity
        radii = np.sqrt(np.sum(new_pts**2, axis=1))
        if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
            return new_pts
        else:
            return pts
            
    except Exception:
        return pts


def _anneal_beta(points, beta_start, beta_end, n_steps, time_limit, start_time):
    """Anneal beta from beta_start to beta_end over n_steps."""
    current = points.copy()
    for i in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        frac = i / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        # Estimate time per step
        remaining_time = time_limit - (time.time() - start_time)
        time_per_step = remaining_time / (n_steps - i)
        
        # Use local polish with time limit
        current = _local_polish(current, beta=beta, max_iter=200, time_limit=min(8.0, time_per_step))
    return current


def construct():
    """Construct 300 circle centers in unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Multi-start strategy
    inits = []
    
    # 1. Hex lattice with different rotations
    for rot, seed in [(0.0, 42), (np.pi / 6, 43), (np.pi / 3, 44), (np.pi / 4, 45), (np.pi/12, 46)]:
        inits.append(_hex_lattice_init(N, rotation=rot, seed=seed))
        
    # 2. Radial ring init
    inits.append(_radial_ring_init(N, seed=47))
    
    # 3. Random init
    rng = np.random.RandomState(48)
    r = np.sqrt(rng.random(N))
    a = rng.random(N) * 2 * np.pi
    inits.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    
    best_points = None
    best_score = -np.inf
    
    # First pass: optimize each init with annealed beta
    for idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > 130.0:  # Leave some time for final polish
            break
        
        current = init.copy()
        
        # Progressive beta annealing
        current = _anneal_beta(current, 50.0, 500.0, 4, 
                               min(20.0, 130.0 - elapsed), start_time)
        
        # Final polish with high beta
        elapsed = time.time() - start_time
        if elapsed > 135.0:
            break
        current = _local_polish(current, beta=1000.0, max_iter=300, time_limit=min(10.0, 135.0 - elapsed))
        
        score = _compute_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Second pass: refine the best solution further
    if best_points is not None and time.time() - start_time < 140.0:
        current = best_points.copy()
        for beta in [2000.0, 5000.0, 10000.0]:
            elapsed = time.time() - start_time
            if elapsed > 142.0:
                break
            current = _local_polish(current, beta=beta, max_iter=500, time_limit=min(10.0, 142.0 - elapsed))
            score = _compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - start_time < 143.0:
        current = best_points.copy()
        current = _local_polish(current, beta=20000.0, max_iter=1000, time_limit=min(5.0, 143.0 - (time.time() - start_time)))
        score = _compute_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
            
    return best_points
# EVOLVE-BLOCK-END
