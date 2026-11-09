"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0  # safety margin before 120 s evaluator kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def _grid_score(points, r):
    """Compute the largest radius that fits for the given points."""
    if len(points) < 2:
        return 0.5
    # min pairwise distance
    # Use broadcasting for N=100
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    return min(min_pair, min_boundary) / 2.0

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for the min-score using log-sum-exp (soft-min).
    Returns a differentiable approximation to the true min radius.
    """
    pts = points
    n = len(pts)
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, np.inf)
    # Soft-min of distances: use -softmax(-dists_sq) scaled
    # soft_min(x) = -log(sum(exp(-x/beta)))/beta ... but for distances we want min
    # Let's use: soft_min(d) ≈ - (1/beta) * log(sum(exp(-d^2/beta)))  ... hmm
    # Better: soft_min of values v_i is -1/beta * log(sum(exp(-v_i/beta)))
    # We have squared distances, so let's work with squared distances
    # min_dist = sqrt(min_dists_sq)
    # Let's do soft-min on dists_sq directly
    # soft_min_sq = -1/beta * log(sum(exp(-dists_sq/beta)))
    # Then soft_min_dist = sqrt(soft_min_sq) ... not quite right but close enough for gradient
    # Actually let's just do soft-min on the distances themselves
    dists = np.sqrt(np.where(mask, dists_sq, np.inf))
    # soft-min of dists
    min_dists = np.where(mask, dists, np.inf)
    # For numerical stability
    min_val = np.min(min_dists)
    exp_terms = np.exp(-(min_dists - min_val) / beta)
    exp_terms = np.where(mask, exp_terms, 0.0)
    soft_min_d = min_val - (1.0/beta) * np.log(np.sum(exp_terms))
    
    # Boundary distances
    margins = CUBE_HALF - np.abs(pts)
    # soft-min of margins
    min_margin = np.min(margins)
    exp_m = np.exp(-(margins - min_margin) / beta)
    soft_min_m = min_margin - (1.0/beta) * np.log(np.sum(exp_m))
    
    # Final soft score: min(soft_min_d, soft_min_m) / 2
    # Use soft-min of the two
    v1 = soft_min_d
    v2 = soft_min_m
    m = min(v1, v2)
    soft_min_final = m - (1.0/beta) * np.log(np.exp(-(v1-m)/beta) + np.exp(-(v2-m)/beta))
    return soft_min_final / 2.0

def _gradient_refine(points, max_iter=5000, lr=0.001, beta=50.0, time_limit=60.0):
    """
    Gradient-based refinement using finite differences on the soft-min score.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points, 0.1)
    current_score = _soft_min_score(points, beta)
    
    for it in range(max_iter):
        if time.time() - start_t > time_limit:
            break
        
        # Compute gradient via finite differences
        n = len(points)
        grad = np.zeros_like(points)
        eps = 1e-5
        for i in range(n):
            for k in range(3):
                orig = points[i, k]
                points[i, k] = orig + eps
                s_plus = _soft_min_score(points, beta)
                points[i, k] = orig - eps
                s_minus = _soft_min_score(points, beta)
                points[i, k] = orig
                grad[i, k] = (s_plus - s_minus) / (2 * eps)
        
        # Update
        points = points + lr * grad
        points = np.clip(points, -CUBE_HALF, CUBE_HALF)
        
        new_score = _grid_score(points, 0.1)
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_pts = points.copy()
            lr *= 0.995
        else:
            lr *= 0.99
            if lr < 1e-6:
                break
    
    return best_pts, best_score

def _refine_jitter(points, r_init, rng, max_iter=20000, step=0.002, step_decay=0.98, time_limit=60.0):
    """
    Hill-climbing refinement with small random jitter.
    Returns best points found and its score.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points, r_init)
    step_size = step
    it = 0
    while it < max_iter:
        if time.time() - start_t > time_limit:
            break
        # Propose a small perturbation to a random point
        idx = rng.integers(0, len(points))
        delta = rng.uniform(-step_size, step_size, size=3)
        new_pt = best_pts[idx].copy() + delta
        # Ensure inside cube
        new_pt = np.clip(new_pt, -CUBE_HALF, CUBE_HALF)
        candidate = best_pts.copy()
        candidate[idx] = new_pt
        cand_score = _grid_score(candidate, r_init)
        if cand_score > best_score + 1e-14:
            best_pts = candidate
            best_score = cand_score
            it += 1
            step_size *= step_decay
        else:
            it += 1
            step_size = max(step_size * 0.99, 1e-5)
            if it % 500 == 0:
                step_size = max(step_size * 0.95, 1e-5)
    return best_pts, best_score

def _fcc_lattice():
    """
    Generate a proper FCC lattice and select 100 points with largest boundary margin.
    FCC basis vectors: a1 = (0, 0.5, 0.5), a2 = (0.5, 0, 0.5), a3 = (0.5, 0.5, 0)
    Lattice points: i*a1 + j*a2 + k*a3 for integers i,j,k
    """
    # Generate FCC points in a range
    points = []
    # Use a range of indices
    for i in range(-2, 5):
        for j in range(-2, 5):
            for k in range(-2, 5):
                x = 0.5 * (j + k)
                y = 0.5 * (i + k)
                z = 0.5 * (i + j)
                # Shift to center in [-0.5, 0.5]
                x -= 1.5
                y -= 1.5
                z -= 1.5
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    points.append((x, y, z))
    
    points = np.array(points)
    if len(points) < 100:
        # Fallback to cubic grid
        nx, ny, nz = 5, 4, 5
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return points
    
    # Compute boundary margin for each point
    margins = CUBE_HALF - np.abs(points)
    min_margins = np.min(margins, axis=1)
    
    # Select 100 points with largest minimum margin
    idx = np.argsort(-min_margins)[:100]
    selected = points[idx]
    
    return selected

def _body_centered_lattice():
    """
    Construct a BCC-like arrangement for 100 points in a unit cube.
    """
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = []
    
    # FCC lattice
    try:
        fcc = _fcc_lattice()
        if len(fcc) == 100:
            configs.append(("fcc", fcc))
    except:
        pass
    
    # BCC-like cubic grid
    bcc = _body_centered_lattice()
    if len(bcc) == 100:
        configs.append(("bcc", bcc))
    
    # Simple cubic grid 5x5x4
    m5 = 5
    ticks5 = (np.arange(m5) + 0.5) / m5 - 0.5
    grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
    if len(grid5) >= 100:
        configs.append(("cubic", grid5[:100]))
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if time.time() - start_time > DEADLINE_S - 30:
            break
        if len(pts) != 100:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts, 0.1)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _body_centered_lattice()
        best_score = _grid_score(best_pts, 0.1)
    
    # Refinement phase
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 5 - elapsed
    
    if remaining > 10:
        # Phase 1: Gradient refinement
        grad_time = min(30.0, remaining * 0.4)
        if grad_time > 5:
            grad_pts, grad_score = _gradient_refine(best_pts, max_iter=2000, lr=0.001, beta=50.0, time_limit=grad_time)
            if grad_score > best_score:
                best_score = grad_score
                best_pts = grad_pts.copy()
        
        # Phase 2: Jitter refinement with multiple restarts
        elapsed2 = time.time() - start_time
        remaining2 = DEADLINE_S - 5 - elapsed2
        if remaining2 > 5:
            max_iter_per_refine = 5000
            n_restarts = max(1, int(remaining2 / (0.1 * max_iter_per_refine)))
            for restart in range(min(n_restarts, 8)):
                if time.time() - start_time > DEADLINE_S - 5:
                    break
                rng_r = np.random.default_rng(1000 + restart)
                if restart == 0:
                    pts_to_refine = best_pts.copy()
                else:
                    pts_to_refine = best_pts + rng_r.normal(0, 0.005, size=(100, 3))
                    pts_to_refine = np.clip(pts_to_refine, -0.5, 0.5)
                
                refined_pts, refined_score = _refine_jitter(pts_to_refine, best_score, rng_r, max_iter=max_iter_per_refine, time_limit=min(15.0, remaining2/2))
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined_pts.copy()
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts, best_score)
    
    return best_pts
# EVOLVE-BLOCK-END
