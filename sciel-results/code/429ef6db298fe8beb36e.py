"""
Sphere packing optimization using Multi-Start L-BFGS-B.
Strategy:
1. Use a Repulsive Energy surrogate (sum of inverse squared distances) for the main optimization.
   This function is smooth and provides strong gradients to push close pairs apart.
2. Use a Log-Sum-Exp (LSE) surrogate for a final polishing phase to precisely maximize
   the minimum radius (soft-min).
3. Multi-start with diverse initial configurations (Jittered Grid, FCC-like, Random).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE = 115.0 # Safety margin for 120s limit

def _repulsive_energy(pts, eps=1e-4):
    """
    Compute the repulsive energy E = sum_{i<j} 1/d_ij^2 + sum_{i,k} 1/w_ik^2.
    Minimizing E maximizes the spread of points.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Soften to avoid singularity
    dists_sq_soft = dists_sq + eps**2
    np.fill_diagonal(dists_sq_soft, np.inf)
    
    inv_d_sq = 1.0 / dists_sq_soft
    inv_d_fourth = inv_d_sq * inv_d_sq
    np.fill_diagonal(inv_d_fourth, 0.0)
    
    E_pairs = 0.5 * np.sum(inv_d_sq)
    
    # Gradient for pairs: -2 * sum_j (1/d^4) * (x_i - x_j)
    row_sums = np.sum(inv_d_fourth, axis=1)
    weighted_pts = inv_d_fourth @ pts
    grad_pairs = -2.0 * (pts * row_sums[:, np.newaxis] - weighted_pts)
    
    # Wall terms
    wall_dists = CUBE_HALF - np.abs(pts)
    wall_dists_clipped = np.clip(wall_dists, eps, None)
    
    inv_w_sq = 1.0 / (wall_dists_clipped ** 2)
    E_walls = np.sum(inv_w_sq)
    
    # Gradient for walls: 2 * (1/w^3) * sign(x)
    inv_w_cubed = 1.0 / (wall_dists_clipped ** 3)
    grad_walls = 2.0 * inv_w_cubed * np.sign(pts)
    
    total_E = E_pairs + E_walls
    total_grad = grad_pairs + grad_walls
    
    return total_E, total_grad

def _repulsive_obj_grad(pts_flat, eps=1e-4):
    """Wrapper for L-BFGS-B."""
    pts = pts_flat.reshape(N, 3)
    val, grad = _repulsive_energy(pts, eps)
    # Add a small regularization to prevent numerical issues if points coincide
    # though eps in energy handles this.
    return val, grad

def _soft_min_radius(pts, beta):
    """
    Compute the smooth surrogate for the minimum radius.
    Returns the estimated radius r and the gradient of -r (for minimization).
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Pairwise terms
    # We exclude i=j by setting dist to inf (exp(-inf)=0)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Wall terms
    wall_dists = CUBE_HALF - np.abs(pts)
    
    # Construct vector of all constraint values
    # Pairs: N*(N-1)/2 values
    # Walls: 3*N values
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    wall_vals = wall_dists.flatten()
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    # Calculate soft min
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    r_val = min_val - np.log(sum_exp) / beta
    
    # Gradient of -R_beta
    p = exp_terms / sum_exp
    grad = np.zeros_like(pts)
    
    # Pair contributions
    p_pairs = p[:len(pair_dists)]
    diff_vecs = pts[i_idx] - pts[j_idx]
    dists_vec = np.sqrt(np.sum(diff_vecs * diff_vecs, axis=1))
    dists_vec[dists_vec == 0] = 1e-10
    unit_vecs = diff_vecs / dists_vec[:, np.newaxis]
    weighted_units = unit_vecs * (0.5 * p_pairs[:, np.newaxis])
    np.add.at(grad, i_idx, weighted_units)
    np.add.at(grad, j_idx, -weighted_units)
    
    # Wall contributions
    p_walls = p[len(pair_dists):]
    p_walls_reshaped = p_walls.reshape(N, 3)
    grad += -p_walls_reshaped * np.sign(pts)
    
    return -r_val, -grad

def _objective_and_grad(pts_flat, beta):
    """Wrapper for L-BFGS-B."""
    pts = pts_flat.reshape(N, 3)
    val, grad = _soft_min_radius(pts, beta)
    return val, grad

def _optimize_lbgfgs(start_pts, time_limit, seed=0):
    """
    Run L-BFGS-B optimization:
    1. Minimize Repulsive Energy to find a good basin.
    2. Minimize LSE Soft-Min to polish the minimum radius.
    """
    start_time = time.time()
    x0 = start_pts.flatten()
    
    # Bounds for each coordinate
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    current_x = x0
    best_x = x0
    
    # Phase 1: Repulsive Energy Minimization
    # This is robust and fast.
    if time.time() - start_time < time_limit * 0.5:
        try:
            res_rep = minimize(
                _repulsive_obj_grad,
                current_x,
                args=(1e-4,), # eps
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_rep.success or res_rep.fun < np.inf:
                current_x = res_rep.x
                best_x = res_rep.x
        except Exception:
            pass
            
    # Phase 2: LSE Soft-Min Polishing
    # Use increasing beta to sharpen the minimum.
    betas = [50, 100, 200, 500]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        try:
            res_lse = minimize(
                _objective_and_grad,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True, # We provide gradient
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Accept if it converged
            if res_lse.success:
                current_x = res_lse.x
                best_x = res_lse.x
        except Exception:
            pass
            
    # Convert back to points
    best_pts = best_x.reshape(N, 3)
    return best_pts

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    points = np.array(points)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    
    # Add random starts
    for i in range(5):
        starts.append(_generate_random_start(100 + i))
        
    best_pts = None
    best_score = -np.inf
    
    # We need a fast scoring function to compare candidates during the run
    # We will use the true radius calculation (min dist / 2, min wall)
    def _score(pts):
        # Vectorized min distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        min_wall = np.min(0.5 - np.abs(pts))
        return min(min_dist / 2.0, min_wall)
        
    # Score initial points to pick the best one to start with?
    # L-BFGS is local. We should run it on all, or the best few.
    # Given time constraints, let's run on the top 3 initial scores.
    
    scored_starts = [(s, _score(s)) for s in starts]
    scored_starts.sort(key=lambda x: x[1], reverse=True)
    
    # Allocate time budget
    # 115s total.
    # Each L-BFGS run might take 5-20s depending on convergence.
    # Let's try to run on the best 4 starts, giving each ~25s.
    
    num_runs = 4
    time_per_run = (DEADLINE - (time.time() - start_time) - 5) / num_runs
    
    for i in range(num_runs):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        init_pts, init_score = scored_starts[i]
        
        # Run L-BFGS
        optimized_pts = _optimize_lbgfgs(init_pts, time_per_run, seed=i)
        
        # Evaluate true score
        current_score = _score(optimized_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = optimized_pts
            
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
            
    # If we have time left, do a final polish on the best result with higher beta
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        remaining_time = DEADLINE - 15 - (time.time() - start_time)
        # Run a single high-beta L-BFGS on the best result found
        # We modify _optimize_lbgfgs to accept a specific beta list or just use high beta
        # For simplicity, we re-run with a custom loop for high beta
        x0 = best_pts.flatten()
        bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
        
        # High beta for polishing
        betas = [500, 1000]
        current_x = x0
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                res = minimize(
                    _objective_and_grad,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.success or res.fun < _score(res.x.reshape(N,3)): # Approx check
                     current_x = res.x
            except:
                pass
        
        polished_pts = current_x.reshape(N, 3)
        polished_score = _score(polished_pts)
        
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
            
    # Fallback
    if best_pts is None:
        best_pts = scored_starts[0][0]
        
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
