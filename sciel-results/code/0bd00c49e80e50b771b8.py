"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 140.0  # safety margin before 150 s evaluator kill

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

def _objective_and_grad(x, beta):
    """
    Returns the negative soft-min score and its gradient.
    We minimize -score, so we maximize score.
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dists = np.sqrt(np.where(mask, dists_sq, np.inf))
    
    # Soft-min of distances
    # f(v) = min(v) approx by -1/beta * log(sum(exp(-v/beta)))
    # To avoid overflow, shift by min
    min_d = np.min(dists)
    exp_d = np.exp(-(dists - min_d) / beta)
    exp_d = np.where(mask, exp_d, 0.0)
    sum_exp_d = np.sum(exp_d)
    soft_min_d = min_d - (1.0/beta) * np.log(sum_exp_d)
    
    # Gradient of soft_min_d w.r.t pts
    # d(soft_min_d)/d(dists) = -1/sum_exp_d * exp(-(dists - min_d)/beta)
    # d(dists)/d(pts) = (pts_i - pts_j) / dists_ij
    # Note: dists is symmetric.
    # Let's compute gradient contribution from distances
    grad_dists = -exp_d / sum_exp_d
    # Zero out diagonal
    grad_dists = np.where(mask, grad_dists, 0.0)
    
    # d(dists_ij)/d(pts_i) = (pts_i - pts_j) / dists_ij
    # d(dists_ij)/d(pts_j) = (pts_j - pts_i) / dists_ij = -d(dists_ij)/d(pts_i)
    
    # We need to accumulate gradients for each point
    grad_pts = np.zeros_like(pts)
    
    # Term: sum_{i<j} grad_dists_ij * (pts_i - pts_j) / dists_ij
    # This is equivalent to:
    # For each i: sum_{j != i} grad_dists_ij * (pts_i - pts_j) / dists_ij
    
    # Efficient computation:
    # Let W_ij = grad_dists_ij / dists_ij
    # grad_pts_i = sum_{j != i} W_ij * (pts_i - pts_j)
    #             = pts_i * sum_{j != i} W_ij - sum_{j != i} W_ij * pts_j
    
    # W is symmetric, W_ii = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        W = grad_dists / dists
    W = np.where(mask, W, 0.0)
    
    sum_W = np.sum(W, axis=1)
    sum_W_pts = W @ pts
    
    grad_pts += sum_W[:, None] * pts - sum_W_pts
    
    # Boundary margins
    margins = CUBE_HALF - np.abs(pts)
    # Soft-min of margins
    min_m = np.min(margins)
    exp_m = np.exp(-(margins - min_m) / beta)
    sum_exp_m = np.sum(exp_m)
    soft_min_m = min_m - (1.0/beta) * np.log(sum_exp_m)
    
    # Gradient of soft_min_m
    # d(soft_min_m)/d(margins) = -exp_m / sum_exp_m
    grad_margins = -exp_m / sum_exp_m
    
    # d(margins)/d(pts)
    # margin_k = 0.5 - |pts_k|
    # d(margin_k)/d(pts_k) = -sign(pts_k)
    # So grad_pts_k += grad_margins_k * (-sign(pts_k))
    
    signs = np.sign(pts)
    # Handle zeros (sign(0)=0, derivative of |x| at 0 is undefined but subgradient is [-1,1], 0 is a safe choice for gradient ascent/descent in practice or random)
    # Actually, for 0, we can use 0 to avoid bias, or random. Let's use 0.
    grad_pts += grad_margins[:, None] * (-signs)
    
    # Combine soft_min_d and soft_min_m
    # We want soft-min of (soft_min_d, soft_min_m)
    v1 = soft_min_d
    v2 = soft_min_m
    m_val = min(v1, v2)
    
    # Soft min of two scalars
    exp_1 = np.exp(-(v1 - m_val) / beta)
    exp_2 = np.exp(-(v2 - m_val) / beta)
    sum_exp = exp_1 + exp_2
    
    soft_min_final = m_val - (1.0/beta) * np.log(sum_exp)
    
    # Gradient of soft_min_final w.r.t pts
    # d(soft_min_final)/d(v1) = -exp_1 / sum_exp
    # d(soft_min_final)/d(v2) = -exp_2 / sum_exp
    
    w_d = exp_1 / sum_exp
    w_m = exp_2 / sum_exp
    
    # Total gradient is w_d * grad(soft_min_d) + w_m * grad(soft_min_m)
    # Note: grad_pts currently holds grad(soft_min_d) + grad(soft_min_m) mixed?
    # No, I calculated grad_pts for soft_min_d and added grad for soft_min_m separately.
    # Let's be precise.
    
    # Re-calculate cleanly to avoid confusion
    # grad_soft_min_d is already in grad_pts? No, I added boundary gradient to it.
    # Let's separate them.
    
    # Actually, the previous calculation `grad_pts += ...` for distances and `grad_pts += ...` for margins
    # resulted in `grad_pts = grad(soft_min_d) + grad(soft_min_m)`.
    # But we need `w_d * grad(soft_min_d) + w_m * grad(soft_min_m)`.
    
    # So I need to recompute or store them separately.
    # Given the cost, let's just recompute the two parts separately to be safe and clear.
    
    # Part 1: Gradient of soft_min_d
    grad_d = np.zeros_like(pts)
    grad_dists_d = -exp_d / sum_exp_d
    grad_dists_d = np.where(mask, grad_dists_d, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        W_d = grad_dists_d / dists
    W_d = np.where(mask, W_d, 0.0)
    sum_W_d = np.sum(W_d, axis=1)
    sum_W_d_pts = W_d @ pts
    grad_d = sum_W_d[:, None] * pts - sum_W_d_pts
    
    # Part 2: Gradient of soft_min_m
    grad_m = np.zeros_like(pts)
    grad_margins_m = -exp_m / sum_exp_m
    grad_m = grad_margins_m[:, None] * (-signs)
    
    # Combine
    total_grad = w_d * grad_d + w_m * grad_m
    
    # Objective is -soft_min_final / 2.0
    # So gradient is -total_grad / 2.0
    objective = -soft_min_final / 2.0
    gradient = -total_grad / 2.0
    
    return objective, gradient.flatten()

def _lbfgs_refine(points, beta, max_iter=500, time_limit=10.0):
    """
    Refine points using L-BFGS-B on the soft-min objective.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    def fun_and_grad(x):
        if time.time() - start_t > time_limit:
            raise TimeoutError()
        return _objective_and_grad(x, beta)
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, 
                       bounds=[(-0.5, 0.5)] * (len(points) * 3),
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            new_pts = res.x.reshape(-1, 3)
            new_score = _grid_score(new_pts)
            return new_pts, new_score
    except TimeoutError:
        pass
    except Exception:
        pass
    return points, _grid_score(points)

def _refine_jitter(points, rng, max_iter=2000, step=0.001, time_limit=5.0):
    """
    Hill-climbing refinement with small random jitter.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(best_pts)
    step_size = step
    it = 0
    while it < max_iter:
        if time.time() - start_t > time_limit:
            break
        idx = rng.integers(0, len(points))
        delta = rng.uniform(-step_size, step_size, size=3)
        new_pt = best_pts[idx].copy() + delta
        new_pt = np.clip(new_pt, -0.5, 0.5)
        candidate = best_pts.copy()
        candidate[idx] = new_pt
        cand_score = _grid_score(candidate)
        if cand_score > best_score + 1e-14:
            best_pts = candidate
            best_score = cand_score
            step_size *= 0.99
        else:
            step_size = max(step_size * 0.98, 1e-6)
        it += 1
    return best_pts, best_score

def _fcc_lattice():
    points = []
    for i in range(-2, 5):
        for j in range(-2, 5):
            for k in range(-2, 5):
                x = 0.5 * (j + k)
                y = 0.5 * (i + k)
                z = 0.5 * (i + j)
                x -= 1.5
                y -= 1.5
                z -= 1.5
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    points.append((x, y, z))
    points = np.array(points)
    if len(points) < 100:
        nx, ny, nz = 5, 4, 5
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return points
    margins = CUBE_HALF - np.abs(points)
    min_margins = np.min(margins, axis=1)
    idx = np.argsort(-min_margins)[:100]
    selected = points[idx]
    return selected

def _body_centered_lattice():
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    configs = []
    try:
        fcc = _fcc_lattice()
        if len(fcc) == 100:
            configs.append(("fcc", fcc))
    except:
        pass
    
    bcc = _body_centered_lattice()
    if len(bcc) == 100:
        configs.append(("bcc", bcc))
    
    m5 = 5
    ticks5 = (np.arange(m5) + 0.5) / m5 - 0.5
    grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
    if len(grid5) >= 100:
        configs.append(("cubic", grid5[:100]))
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if time.time() - start_time > DEADLINE_S - 20:
            break
        if len(pts) != 100:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _body_centered_lattice()
        best_score = _grid_score(best_pts)
    
    # Annealing schedule for beta
    # Start with high beta (smooth) to escape local minima, then lower it (sharp) to converge
    betas = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE_S - 10:
            break
        
        # Run L-BFGS with current beta
        # Allocate more time for earlier, smoother steps? Or uniform?
        # Let's give 10s to each step if possible, but cap total.
        elapsed = time.time() - start_time
        remaining = DEADLINE_S - 5 - elapsed
        if remaining < 5:
            break
            
        # Use remaining time / number of remaining steps as a guide, but min 5s
        time_per_step = min(15.0, remaining / len(betas))
        
        refined_pts, refined_score = _lbfgs_refine(best_pts, beta, max_iter=100, time_limit=time_per_step)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
        else:
            # If no improvement, maybe try a small jitter restart?
            # For now, just move to next beta
            pass
    
    # Final jitter polish
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 2 - elapsed
    if remaining > 1:
        final_pts, final_score = _refine_jitter(best_pts, rng, max_iter=1000, time_limit=remaining)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
# EVOLVE-BLOCK-END
