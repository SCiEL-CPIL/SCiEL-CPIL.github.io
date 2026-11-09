"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0  # safety margin before 120 s evaluator kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys
from scipy.optimize import minimize

def _grid_score(points):
    """Compute the largest radius that fits for the given points."""
    if len(points) < 2:
        return 0.5
    # min pairwise distance
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    return min(min_pair, min_boundary) / 2.0

def _exp_repulsion_objective(x, beta, eps):
    """
    Objective: Sum of exp(-beta * dist) for pairs + exp(-beta * margin) for boundaries.
    Minimizing this pushes points apart and away from walls.
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    
    # Pairwise
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # To avoid overflow in exp, we can just sum exp(-beta * dists)
    # Since dists >= 0, exp(-beta * dists) <= 1.
    pair_term = np.sum(np.exp(-beta * dists) * mask)
    
    # Boundary
    # Use smooth absolute value: sqrt(x^2 + eps^2)
    margins = CUBE_HALF - np.sqrt(pts**2 + eps**2)
    bound_term = np.sum(np.exp(-beta * margins))
    
    return pair_term + bound_term

def _exp_repulsion_grad(x, beta, eps):
    """
    Gradient of the exp repulsion objective.
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Pairwise
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # d/dx_i sum_{j!=i} exp(-beta * d_ij)
    # = sum_{j!=i} exp(-beta * d_ij) * (-beta) * (x_i - x_j) / d_ij
    weights = np.exp(-beta * dists) * mask
    # w[i, j] is weight for pair (i, j)
    # grad[i] += sum_j w[i, j] * (-beta) * d[i, j] / d[i, j]
    # d[i, j] is vector x_i - x_j
    # So grad[i] = -beta * sum_j w[i, j] * (x_i - x_j) / d_ij
    
    # Efficient computation:
    # G[i] = -beta * sum_j w[i, j] * (x_i - x_j) / d_ij
    # Note: w is symmetric, d_ij = -d_ji, w_ij = w_ji
    # So we can compute for i and add to both.
    
    # Vectorized approach:
    # For each i, grad_i = -beta * sum_j (exp(-beta d_ij) / d_ij) * (x_i - x_j)
    # Let A[i, j] = exp(-beta d_ij) / d_ij for i != j, 0 otherwise
    # Then grad = -beta * (A * d) where * is elementwise and sum over j
    
    # To avoid division by zero, we set A to 0 where d is small or diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(mask & (dists > 1e-8), weights / dists, 0.0)
    
    # grad[i] = -beta * sum_j A[i, j] * d[i, j]
    # d[i, j] is (x_i - x_j)
    # sum_j A[i, j] * (x_i - x_j) = x_i * sum_j A[i, j] - sum_j A[i, j] * x_j
    
    sum_A = np.sum(A, axis=1, keepdims=True) # shape (n, 1)
    Ax = np.sum(A[:, :, None] * pts[None, :, :], axis=1) # shape (n, 3)
    
    grad_pair = -beta * (sum_A * pts - Ax)
    
    # Boundary
    # f = sum_i sum_k exp(-beta * (0.5 - sqrt(x_ik^2 + eps^2)))
    # d/dx_ik = exp(...) * (-beta) * d/dx_ik (0.5 - sqrt(...))
    # d/dx_ik (sqrt(x_ik^2 + eps^2)) = x_ik / sqrt(x_ik^2 + eps^2)
    # So d/dx_ik = exp(...) * (-beta) * (-x_ik / sqrt(...)) = beta * exp(...) * x_ik / sqrt(...)
    
    smooth_abs = np.sqrt(pts**2 + eps**2)
    margins = CUBE_HALF - smooth_abs
    exp_m = np.exp(-beta * margins)
    grad_bound = beta * exp_m * (pts / smooth_abs)
    
    return (grad_pair + grad_bound).flatten()

def _refine_lbfgs(points, time_limit=10.0, beta_start=2.0, beta_end=40.0, n_steps=5):
    """
    Refine using L-BFGS-B with exponential repulsion potential.
    Uses a schedule of increasing beta to focus on local minima.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points)
    
    # Beta controls the "strength" or "range" of the repulsion.
    # Small beta: long-range, smooth.
    # Large beta: short-range, sharp (like hard spheres).
    # We start with small beta to escape local minima, then increase.
    betas = np.linspace(beta_start, beta_end, n_steps)
    eps = 1e-4
    
    for beta in betas:
        if time.time() - start_t > time_limit:
            break
        
        x0 = best_pts.flatten()
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        
        try:
            res = minimize(
                _exp_repulsion_objective,
                x0,
                args=(beta, eps),
                method='L-BFGS-B',
                jac=_exp_repulsion_grad,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            # Check if the result is valid and better
            new_pts = res.x.reshape(-1, 3)
            new_pts = np.clip(new_pts, -CUBE_HALF, CUBE_HALF)
            new_score = _grid_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass
        
    return best_pts, best_score

def _refine_jitter(points, rng, time_limit=10.0, max_iter=50000):
    """
    Hill-climbing refinement with small random jitter.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points)
    step_size = 0.002
    it = 0
    while it < max_iter:
        if time.time() - start_t > time_limit:
            break
        idx = rng.integers(0, len(points))
        delta = rng.uniform(-step_size, step_size, size=3)
        new_pt = best_pts[idx].copy() + delta
        new_pt = np.clip(new_pt, -CUBE_HALF, CUBE_HALF)
        candidate = best_pts.copy()
        candidate[idx] = new_pt
        cand_score = _grid_score(candidate)
        if cand_score > best_score + 1e-14:
            best_pts = candidate
            best_score = cand_score
            step_size *= 0.98
        else:
            step_size = max(step_size * 0.99, 1e-5)
            if it % 500 == 0:
                step_size = max(step_size * 0.95, 1e-5)
        it += 1
    return best_pts, best_score

def _fcc_lattice():
    """
    Generate FCC lattice points in [-0.5, 0.5]^3 and select 100 with largest min distance to boundary.
    """
    points = []
    # FCC primitive vectors: v1=(0, 0.5, 0.5), v2=(0.5, 0, 0.5), v3=(0.5, 0.5, 0)
    # Lattice point = i*v1 + j*v2 + k*v3
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                x = 0.5 * (j + k)
                y = 0.5 * (i + k)
                z = 0.5 * (i + j)
                # Center: shift so centroid is at origin
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    points.append((x, y, z))
    
    points = np.array(points)
    
    if len(points) < N:
        # Fallback to cubic grid if not enough points
        nx, ny, nz = 5, 4, 5
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return points
    
    # Select N points with largest minimum distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_margins = np.min(margins, axis=1)
    idx = np.argsort(-min_margins)[:N]
    return points[idx]

def _cubic_grid():
    """Simple cubic grid 5x4x5 = 100 points."""
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    return np.array([(x, y, z) for z in zs for y in ys for x in xs])

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = []
    
    try:
        fcc = _fcc_lattice()
        if len(fcc) == N:
            configs.append(("fcc", fcc))
    except:
        pass
    
    cubic = _cubic_grid()
    if len(cubic) == N:
        configs.append(("cubic", cubic))
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if time.time() - start_time > DEADLINE_S - 60:
            break
        if len(pts) != N:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _cubic_grid()
        best_score = _grid_score(best_pts)
    
    # Refinement phase 1: Multi-restart L-BFGS-B
    # We perform several short L-BFGS runs from perturbed versions of the best initial config
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 10 - elapsed
    if remaining > 20:
        n_restarts = 5
        time_per_restart = remaining / (n_restarts + 2) # Reserve some time for jitter
        
        for i in range(n_restarts):
            if time.time() - start_time > DEADLINE_S - 10:
                break
            
            # Perturb the current best for diversity in restarts
            if i == 0:
                start_pts = best_pts.copy()
            else:
                start_pts = best_pts + rng.normal(0, 0.005, size=(N, 3))
                start_pts = np.clip(start_pts, -CUBE_HALF, CUBE_HALF)
            
            lbfgs_pts, lbfgs_score = _refine_lbfgs(start_pts, time_limit=time_per_restart)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_pts = lbfgs_pts.copy()
    
    # Refinement phase 2: Jitter polish
    elapsed2 = time.time() - start_time
    remaining2 = DEADLINE_S - 5 - elapsed2
    if remaining2 > 5:
        jitter_pts, jitter_score = _refine_jitter(best_pts, rng, time_limit=remaining2)
        if jitter_score > best_score:
            best_score = jitter_score
            best_pts = jitter_pts.copy()
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
