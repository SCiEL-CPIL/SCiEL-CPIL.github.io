import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > HALF) or np.any(points < -HALF):
        return 0.0
        
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = HALF - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _soft_min(values, beta):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    v_min = v.min()
    exp_vals = np.exp(-beta * (v - v_min))
    return v_min - (1.0 / beta) * np.log(np.sum(exp_vals))

def _objective_and_grad(x, beta):
    """
    Objective function and its gradient for L-BFGS-B.
    Returns (f, g) where f = -soft_min(radius_constraints) and g = gradient of f.
    """
    points = x.reshape(N, 3)
    
    # Wall distances: w_ik = 0.5 - |x_ik|
    # d(w_ik)/dx_ik = -sign(x_ik)
    wall_dists = HALF - np.abs(points)
    wall_grad = -np.sign(points)  # shape (N, 3)
    
    # Pair distances: d_ij = ||c_i - c_j||
    # d(d_ij)/dx_i = (c_i - c_j) / d_ij
    # d(d_ij)/dx_j = (c_j - c_i) / d_ij
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    # Unit vectors: u_ij = (c_i - c_j) / d_ij
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    np.fill_diagonal(unit_vecs[:, :, 0], 0.0)  # Not quite right, let's do it properly
    
    # Actually, let's compute gradients more carefully
    # For pair (i,j): contribution to pair_dists is d_ij / 2
    # d(pair_dists_ij)/dx_i = 0.5 * (c_i - c_j) / d_ij
    # d(pair_dists_ij)/dx_j = 0.5 * (c_j - c_i) / d_ij
    
    # We need soft_min over all constraints:
    # All constraints: {d_ij/2 for i<j} union {w_ik for all i,k}
    # soft_min = min_val - (1/beta) * log(sum(exp(-beta*(v - min_val))))
    # d(soft_min)/dx = -sum_j [exp(-beta*(v_j - min_val)) / sum_k exp(-beta*(v_k - min_val))] * beta * d(v_j)/dx
    #               = -sum_j [w_j] * d(v_j)/dx  where w_j = exp(-beta*(v_j-min_val))/sum(...)
    
    # Compute all constraint values
    triu_i, triu_j = np.triu_indices(N, k=1)
    pair_vals = dists[triu_i, triu_j] / 2.0  # shape (1225,)
    wall_vals = wall_dists.flatten()  # shape (150,)
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    n_constraints = len(all_vals)
    
    # Soft min
    v_min = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - v_min))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (n_constraints,)
    
    soft_min_val = v_min - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of soft_min w.r.t. x
    # d(soft_min)/dx = -sum_j weights[j] * d(v_j)/dx
    grad = np.zeros((N, 3))
    
    # Pair contributions: for pair (i,j), v = d_ij/2
    # d(v)/dx_i = 0.5 * (c_i - c_j) / d_ij = 0.5 * unit_vec_ij
    # d(v)/dx_j = -0.5 * unit_vec_ij
    pair_weights = weights[:len(pair_vals)]
    
    for idx in range(len(pair_vals)):
        i = triu_i[idx]
        j = triu_j[idx]
        w = pair_weights[idx]
        # unit vector from j to i
        uv = (points[i] - points[j]) / dists[i, j]
        grad[i] += 0.5 * w * uv
        grad[j] -= 0.5 * w * uv
    
    # Wall contributions: for wall constraint w_ik = 0.5 - |x_ik|
    # d(w_ik)/dx_ik = -sign(x_ik)
    wall_weights = weights[len(pair_vals):]
    wall_w = wall_weights.reshape(N, 3)
    grad += wall_w * wall_grad  # wall_grad = -sign(points)
    
    # f = -soft_min, so grad_f = -grad_soft_min
    f = -soft_min_val
    g = -grad.flatten()
    
    return f, g

def _objective_flat(x, beta=50.0):
    """Objective function only (for compatibility)."""
    f, _ = _objective_and_grad(x, beta)
    return f

def _init_bcc(n):
    """
    Generate a BCC (body-centered cubic) lattice initialization.
    BCC has 2 points per unit cell: corners + center.
    We create a grid of BCC cells and select n points.
    """
    # Try different BCC densities
    # A 3x3x3 BCC lattice has 27*2 = 54 points, close to 50
    points = []
    
    # 3x3x3 BCC: corners at integer coords, centers at half-integer coords
    # Scale to fit in [-0.5, 0.5]^3
    # Unit cell size s, so points range from 0 to 3*s in each dim
    # We want 3*s = 1, so s = 1/3
    s = 1.0 / 3.0
    
    for i in range(4):  # 0, 1, 2, 3
        for j in range(4):
            for k in range(4):
                # Corner points
                if i < 3 and j < 3 and k < 3:
                    # This is getting complex. Let's use a simpler approach.
                    pass
    
    # Simpler: generate BCC points in a cube and scale
    # BCC: for each cell (i,j,k), add corner (i,j,k) and center (i+0.5, j+0.5, k+0.5)
    # Use 3x3x3 cells: 27 cells * 2 points = 54 points
    bcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                bcc_points.append([i, j, k])
                # Center
                bcc_points.append([i + 0.5, j + 0.5, k + 0.5])
    
    bcc_points = np.array(bcc_points, dtype=float)
    
    # Scale to fit in [-0.5, 0.5]^3
    # Current range: [0, 2.5] in each dim (from 0 to 2.5)
    # Center: (1.25, 1.25, 1.25)
    # Scale: want range 1.0, current range 2.5, so scale = 1.0/2.5 = 0.4
    center = np.mean(bcc_points, axis=0)
    bcc_points = (bcc_points - center) * 0.4
    
    # Add jitter and clip
    bcc_points += np.random.randn(len(bcc_points), 3) * 0.01
    bcc_points = np.clip(bcc_points, -0.49, 0.49)
    
    # Select n points
    if len(bcc_points) >= n:
        idx = np.random.choice(len(bcc_points), n, replace=False)
        bcc_points = bcc_points[idx]
    else:
        # Pad with random points
        extra = np.random.uniform(-0.45, 0.45, (n - len(bcc_points), 3))
        bcc_points = np.vstack([bcc_points, extra])
    
    return bcc_points

def _init_fcc(n):
    """
    Generate an FCC (face-centered cubic) lattice initialization.
    FCC has 4 points per unit cell: corners + face centers.
    """
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                fcc_points.append([i, j, k])
                # Face centers
                fcc_points.append([i + 0.5, j + 0.5, k])
                fcc_points.append([i + 0.5, j, k + 0.5])
                fcc_points.append([i, j + 0.5, k + 0.5])
    
    fcc_points = np.array(fcc_points, dtype=float)
    
    # Scale to fit
    center = np.mean(fcc_points, axis=0)
    fcc_points = (fcc_points - center) * 0.35
    
    fcc_points += np.random.randn(len(fcc_points), 3) * 0.01
    fcc_points = np.clip(fcc_points, -0.49, 0.49)
    
    if len(fcc_points) >= n:
        idx = np.random.choice(len(fcc_points), n, replace=False)
        fcc_points = fcc_points[idx]
    else:
        extra = np.random.uniform(-0.45, 0.45, (n - len(fcc_points), 3))
        fcc_points = np.vstack([fcc_points, extra])
    
    return fcc_points

def _init_random(n):
    """Generate a random initial configuration."""
    points = np.random.uniform(-0.45, 0.45, (n, 3))
    return points

def _optimize(points, max_time, start_beta=30.0, end_beta=500.0):
    """
    Optimize the configuration using L-BFGS-B with adaptive beta.
    """
    start_time = time.time()
    x0 = points.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-HALF, HALF)] * (N * 3)
    
    best_x = x0.copy()
    best_f = _objective_flat(x0, start_beta)
    
    # Phase 1: Coarse optimization with lower beta
    try:
        def obj_coarse(x):
            return _objective_flat(x, start_beta)
        
        def grad_coarse(x):
            return _objective_and_grad(x, start_beta)[1]
        
        res = minimize(
            obj_coarse,
            x0,
            jac=grad_coarse,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time * 0.5:
        return best_x.reshape(N, 3), -best_f
    
    # Phase 2: Fine optimization with higher beta
    mid_beta = (start_beta + end_beta) / 2.0
    try:
        def obj_mid(x):
            return _objective_flat(x, mid_beta)
        
        def grad_mid(x):
            return _objective_and_grad(x, mid_beta)[1]
        
        res2 = minimize(
            obj_mid,
            best_x,
            jac=grad_mid,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x.copy()
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time * 0.8:
        return best_x.reshape(N, 3), -best_f
    
    # Phase 3: Final polish with very high beta
    try:
        def obj_fine(x):
            return _objective_flat(x, end_beta)
        
        def grad_fine(x):
            return _objective_and_grad(x, end_beta)[1]
        
        res3 = minimize(
            obj_fine,
            best_x,
            jac=grad_fine,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if res3.fun < best_f:
            best_f = res3.fun
            best_x = res3.x.copy()
            
    except Exception:
        pass
    
    return best_x.reshape(N, 3), -best_f

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Run multiple restarts with different initializations
    n_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        n_restarts += 1
        
        # Cycle through different initializations
        if n_restarts % 4 == 1:
            points = _init_bcc(N)
        elif n_restarts % 4 == 2:
            points = _init_fcc(N)
        elif n_restarts % 4 == 3:
            points = _init_random(N)
        else:
            # Perturb the best solution
            if best_points is not None:
                points = best_points + np.random.randn(N, 3) * 0.02
                points = np.clip(points, -0.49, 0.49)
            else:
                points = _init_bcc(N)
        
        # Optimize with adaptive time budget
        elapsed = time.time() - start_time
        remaining = DEADLINE_SECONDS - elapsed
        if remaining < 5.0:
            break
        
        # Use more time for early restarts, less for later ones
        opt_time = min(15.0, max(5.0, remaining * 0.4))
        
        opt_points, opt_score = _optimize(points, opt_time)
        
        # Compute actual score
        actual_score = _score(opt_points)
        
        if actual_score > best_score:
            best_score = actual_score
            best_points = opt_points.copy()
            
        # If we have a good score, try a few more refinements from the best point
        if best_score > 0.13 and time.time() - start_time < DEADLINE_SECONDS - 10:
            for _ in range(2):
                if time.time() - start_time > DEADLINE_SECONDS - 5:
                    break
                refined_points, refined_score = _optimize(best_points, 8.0)
                actual_refined_score = _score(refined_points)
                if actual_refined_score > best_score:
                    best_score = actual_refined_score
                    best_points = refined_points.copy()
    
    if best_points is None:
        best_points = _init_bcc(N)
        
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
