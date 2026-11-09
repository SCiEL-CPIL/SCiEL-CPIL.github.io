"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0  # safety margin before 120 s evaluator kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def _cubic_grid(m):
    """Return (m**3, 3) array of cubic grid points in [-0.5, 0.5]^3."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid

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

def _refine_jitter(points, r_init, rng, max_iter=20000, step=0.002, step_decay=0.98):
    """
    Hill-climbing refinement with small random jitter.
    Returns best points found and its score.
    """
    best_pts = points.copy()
    best_score = _grid_score(points, r_init)
    step_size = step
    it = 0
    while it < max_iter:
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

def _smooth_score(points, r_target, beta=50.0):
    """
    Smooth surrogate for the min-distance objective using soft-min.
    r = min(min_pairwise_dist/2, min_boundary_margin)
    We approximate min with a soft-min: -log-sum-exp(-beta * values) / beta
    """
    n = len(points)
    # Pairwise distances
    d = points[:, None, :] - points[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    # Use sqrt for actual distances
    dists = np.sqrt(dists_sq)
    # Boundary margins
    margins = CUBE_HALF - np.abs(points)
    min_margins = np.min(margins, axis=1)
    
    # Combine all constraints: pairwise distances and boundary margins
    # We want min of all these
    all_vals = np.concatenate([dists.flatten(), min_margins])
    # Soft-min approximation
    # For large beta, -log(sum(exp(-beta*x)))/beta ≈ min(x)
    # Use log-sum-exp for numerical stability
    min_val = np.min(all_vals)
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - min_val))))
    soft_min = min_val - np.log(np.sum(np.exp(-beta * (all_vals - min_val)))) / beta
    
    return soft_min / 2.0

def _gradient_refine(points, max_iter=500, lr=0.001, beta=50.0):
    """
    Gradient-based refinement using the smooth surrogate.
    """
    best_pts = points.copy()
    best_score = _grid_score(points, 0.1)
    
    for it in range(max_iter):
        # Compute gradient of smooth score w.r.t. each point
        # d(soft_min)/dx_i = -sum_j exp(-beta*(x_j - min)) * (-beta) * dx_j/dx_i / sum exp(-beta*(x_j - min))
        # This is complex; use numerical gradient instead for simplicity
        
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(best_pts)
        for i in range(len(best_pts)):
            for j in range(3):
                perturbed = best_pts.copy()
                perturbed[i, j] += eps
                perturbed = np.clip(perturbed, -CUBE_HALF, CUBE_HALF)
                f_plus = _smooth_score(perturbed, best_score, beta)
                
                perturbed2 = best_pts.copy()
                perturbed2[i, j] -= eps
                perturbed2 = np.clip(perturbed2, -CUBE_HALF, CUBE_HALF)
                f_minus = _smooth_score(perturbed2, best_score, beta)
                
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient ascent (we want to maximize the score)
        new_pts = best_pts + lr * grad
        new_pts = np.clip(new_pts, -CUBE_HALF, CUBE_HALF)
        
        new_score = _grid_score(new_pts, best_score)
        if new_score > best_score + 1e-15:
            best_pts = new_pts
            best_score = new_score
            lr *= 0.99
        else:
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return best_pts, best_score

def _body_centered_lattice():
    """
    Construct a BCC-like arrangement for 100 points in a unit cube.
    We place points in a 5x4x5 cubic grid with offsets to approximate BCC.
    """
    # Use a 5x4x5 grid = 100 points
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _fcc_lattice():
    """
    Generate FCC lattice points and select the 100 with best boundary margin.
    FCC basis vectors: a1=(0, 0.5, 0.5), a2=(0.5, 0, 0.5), a3=(0.5, 0.5, 0)
    Lattice points: n1*a1 + n2*a2 + n3*a3 for integer n1,n2,n3
    """
    # Generate FCC lattice points in a region slightly larger than the cube
    # FCC lattice: points at (i, j, k) where i+j+k is even, scaled by 0.5
    # This gives spacing of 1/sqrt(2) in 3D
    points = []
    # Range: need enough points to select 100 from within [-0.5, 0.5]^3
    # With spacing ~0.5, we need about 2-3 points per axis
    # Let's generate a 4x4x4 grid of FCC unit cells
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                # FCC: corner + face centers
                # Corner: (i, j, k)
                # Face centers: (i+0.5, j+0.5, k), (i+0.5, j, k+0.5), (i, j+0.5, k+0.5)
                candidates = [
                    (i, j, k),
                    (i + 0.5, j + 0.5, k),
                    (i + 0.5, j, k + 0.5),
                    (i, j + 0.5, k + 0.5)
                ]
                for p in candidates:
                    # Scale to fit in [-0.5, 0.5] - FCC nearest neighbor distance is 1/sqrt(2)
                    # We want the packing to fit, so scale down
                    # Actually, let's just generate in [-2, 2] and select those in [-0.5, 0.5]
                    points.append(p)
    
    points = np.array(points)
    # Scale: FCC lattice with unit spacing has NN distance 1/sqrt(2) ≈ 0.707
    # We need to fit in [-0.5, 0.5], so scale by 0.5/0.707 ≈ 0.707
    # Actually, let's just select points that fall in [-0.5, 0.5] and scale appropriately
    # The raw FCC lattice has points at integer and half-integer positions
    # In [-0.5, 0.5], we have points like (0,0,0), (0.5,0.5,0), etc.
    
    # Filter points in [-0.5, 0.5]
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    inner_points = points[mask]
    
    if len(inner_points) < 100:
        # Need to generate more - use a denser lattice
        # Let's try scaling: generate in [-2, 2] and scale to [-0.5, 0.5]
        points2 = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    candidates = [
                        (i, j, k),
                        (i + 0.5, j + 0.5, k),
                        (i + 0.5, j, k + 0.5),
                        (i, j + 0.5, k + 0.5)
                    ]
                    for p in candidates:
                        points2.append(p)
        points2 = np.array(points2)
        # Scale by 0.25 to fit in [-1, 1], then we can select from [-0.5, 0.5]
        points2 = points2 * 0.25
        mask2 = np.all(np.abs(points2) <= 0.5, axis=1)
        inner_points = points2[mask2]
    
    if len(inner_points) < 100:
        # Fallback to cubic grid
        return _body_centered_lattice()
    
    # Compute boundary margin for each point
    margins = CUBE_HALF - np.abs(inner_points)
    min_margin = np.min(margins, axis=1)
    
    # Sort by margin (largest first) and select top 100
    indices = np.argsort(-min_margin)
    selected = inner_points[indices[:100]]
    
    return selected

def construct():
    start_time = time.time()
    
    # Strategy 1: Body-centered lattice (4x5x5 grid with offsets)
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    configs = [
        ("cubic_5x4x5", _body_centered_lattice()),
    ]
    
    # Also try the original simple cubic grid
    m5 = 5
    ticks5 = (np.arange(m5) + 0.5) / m5 - 0.5
    grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
    configs.append(("cubic_5x5x4", grid5[:100]))
    
    # FCC-like
    try:
        fcc = _fcc_lattice()
        if len(fcc) >= 100:
            configs.append(("fcc_like", fcc[:100]))
    except:
        pass
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if time.time() - start_time > DEADLINE_S - 30:
            break
        if len(pts) != 100:
            continue
        # Ensure valid
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts, 0.1)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _body_centered_lattice()
        best_score = _grid_score(best_pts, 0.1)
    
    # Refine with hill climbing
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 5 - elapsed  # 5s safety margin
    if remaining > 10:
        # Multiple restarts
        max_iter_per_refine = 10000
        n_restarts = max(1, int(remaining / (0.05 * max_iter_per_refine)))
        for restart in range(min(n_restarts, 10)):
            if time.time() - start_time > DEADLINE_S - 5:
                break
            rng = np.random.default_rng(1000 + restart)
            if restart == 0:
                pts_to_refine = best_pts.copy()
            else:
                # Perturb the best solution
                pts_to_refine = best_pts + rng.normal(0, 0.01, size=(100, 3))
                pts_to_refine = np.clip(pts_to_refine, -0.5, 0.5)
            
            refined_pts, refined_score = _refine_jitter(pts_to_refine, best_score, rng, max_iter=max_iter_per_refine)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    # Final high-precision polish with gradient-based refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 3 - elapsed
    if remaining > 5:
        # Use gradient refinement for final polish
        max_grad_iter = min(200, int(remaining * 10))
        polished_pts, polished_score = _gradient_refine(best_pts, max_iter=max_grad_iter, lr=0.0005, beta=100.0)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts, best_score)
    
    return best_pts
# EVOLVE-BLOCK-END
