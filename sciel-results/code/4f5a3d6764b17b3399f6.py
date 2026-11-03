"""Rewritten program for the 50-circle packing task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _score(c):
    """Compute the packing radius for centers c of shape (N, 2)."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall constraints
    wall = 0.5 - np.abs(c).max()
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    return float(min(wall, min_pair))


def _radius_fast(c):
    """Fast radius computation."""
    c = np.asarray(c, dtype=float)
    wall = 0.5 - np.abs(c).max()
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(min(wall, dist.min() / 2.0))


def _soft_min_radius_neg(x, beta=50.0):
    """
    Negative soft-min radius objective for L-BFGS-B.
    x is a flat array of shape (2*N,).
    Uses log-sum-exp for the pairwise term and a soft wall term.
    """
    c = x.reshape(-1, 2)
    
    # Wall term: soft-min of (0.5 - |c_i|) over all coordinates
    # We want to maximize min(0.5 - |c_ik|), so we minimize -soft_min
    # 0.5 - |c_ik| is the distance to the wall for coordinate k of circle i
    wall_dists = 0.5 - np.abs(c)  # shape (N, 2)
    # soft min over all 2N values
    flat_walls = wall_dists.ravel()
    # We want the minimum wall distance, so we use -soft_max(-flat_walls)
    # soft_max(-w) = -logsumexp(-(-w)/beta)*beta ... let's be careful
    # soft_min(w) = -1/beta * log(sum(exp(-beta*w)))
    # For numerical stability:
    w_min = flat_walls.min()
    soft_min_wall = w_min - np.log(np.sum(np.exp(-beta * (flat_walls - w_min)))) / beta
    
    # Pairwise term: soft-min of dist_ij / 2
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    p_min = pair_vals.min()
    # Exclude diagonal (inf) from logsumexp
    # Create a mask
    mask = ~np.eye(N, dtype=bool)
    flat_pairs = pair_vals[mask]
    fp_min = flat_pairs.min()
    soft_min_pair = fp_min - np.log(np.sum(np.exp(-beta * (flat_pairs - fp_min)))) / beta
    
    # The radius is min(soft_min_wall, soft_min_pair)
    # We want to maximize this, so minimize the negative
    # Use soft-min of the two values
    r1 = soft_min_wall
    r2 = soft_min_pair
    r_min = min(r1, r2)
    # soft min of two values
    r_soft = r_min - np.log(np.exp(-beta * (r1 - r_min)) + np.exp(-beta * (r2 - r_min))) / beta
    
    return -r_soft


def _soft_min_radius_neg_grad(x, beta=50.0):
    """Gradient of the soft-min radius objective."""
    c = x.reshape(-1, 2)
    
    # Wall term gradient
    wall_dists = 0.5 - np.abs(c)
    flat_walls = wall_dists.ravel()
    w_min = flat_walls.min()
    exp_w = np.exp(-beta * (flat_walls - w_min))
    sum_exp_w = np.sum(exp_w)
    # d(soft_min_wall)/d(wall_dists) = -exp_w / sum_exp_w
    grad_wall = -exp_w / sum_exp_w
    # d(wall_dists)/d(c) = -sign(c)
    grad_wall_c = grad_wall.reshape(-1, 2) * (-np.sign(c))
    
    # Pairwise term gradient
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    mask = ~np.eye(N, dtype=bool)
    flat_pairs = pair_vals[mask]
    fp_min = flat_pairs.min()
    exp_p = np.exp(-beta * (flat_pairs - fp_min))
    sum_exp_p = np.sum(exp_p)
    
    # d(soft_min_pair)/d(pair_vals_ij) = -exp_p / sum_exp_p
    # d(pair_vals_ij)/d(c_i) = -diff_ij / (2 * dist_ij) ... wait
    # pair_vals_ij = dist_ij / 2
    # d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij = diff_ij / dist_ij
    # d(pair_vals_ij)/d(c_i) = diff_ij / (2 * dist_ij)
    # d(pair_vals_ij)/d(c_j) = -diff_ij / (2 * dist_ij)
    
    grad_pair_c = np.zeros_like(c)
    # Reconstruct the full gradient from the flat values
    # This is complex; let's compute it directly
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = c[i] - c[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-12:
                continue
            # Find the index in flat_pairs for (i,j)
            # The mask is upper triangle + lower triangle
            # Let's just compute the contribution
            # The weight for pair (i,j) in the flat array
            # This is getting complex. Let's use a different approach.
    
    # Actually, let's just use the analytic gradient more carefully
    # soft_min_pair = fp_min - (1/beta) * log(sum(exp(-beta*(pair_vals - fp_min))))
    # d(soft_min_pair)/d(pair_vals_k) = -(1/beta) * (-beta * exp(-beta*(pair_vals_k - fp_min))) / sum(...)
    # = exp(-beta*(pair_vals_k - fp_min)) / sum(...)
    # Wait, derivative of -log(sum(exp(-beta*(x - x_min)))) w.r.t x_k:
    # = -1/sum * d(sum)/dx_k = -1/sum * (-beta * exp(-beta*(x_k - x_min))) = beta * exp(-beta*(x_k - x_min))/sum
    # So d(soft_min)/dx_k = beta * exp(-beta*(x_k - x_min))/sum
    # But soft_min = x_min - (1/beta)*log(sum(...))
    # d(x_min)/dx_k is 0 (almost everywhere)
    # d(soft_min)/dx_k = -(1/beta) * d(log(sum(exp(-beta*(x - x_min)))))/dx_k
    # = -(1/beta) * sum_j (-beta * exp(-beta*(x_j - x_min)) * delta_{jk}) / sum
    # = exp(-beta*(x_k - x_min)) / sum
    
    # So the weight is positive.
    # d(soft_min_pair)/d(c_i) = sum_j weight_ij * d(pair_vals_ij)/d(c_i)
    # = sum_j weight_ij * (c_i - c_j) / (2 * dist_ij)
    
    # Let's compute this properly
    grad_pair_c = np.zeros_like(c)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = c[i] - c[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-12:
                continue
            # Compute weight
            pv = dist_ij / 2.0
            w = np.exp(-beta * (pv - fp_min)) / sum_exp_p
            # Contribution to c_i
            grad_pair_c[i] += w * d / (2.0 * dist_ij)
            # Contribution to c_j
            grad_pair_c[j] += w * (-d) / (2.0 * dist_ij)
    
    # Combine wall and pair gradients
    # r_soft = soft_min(r1, r2) where r1 = soft_min_wall, r2 = soft_min_pair
    # This is a soft min of two values, so the gradient is a weighted average
    r1 = w_min - np.log(sum_exp_w) / beta
    r2 = fp_min - np.log(sum_exp_p) / beta
    r_min = min(r1, r2)
    exp_r1 = np.exp(-beta * (r1 - r_min))
    exp_r2 = np.exp(-beta * (r2 - r_min))
    sum_exp_r = exp_r1 + exp_r2
    w1 = exp_r1 / sum_exp_r
    w2 = exp_r2 / sum_exp_r
    
    # d(soft_min)/d(c) = w1 * grad_wall_c + w2 * grad_pair_c
    # But we're returning the negative, so negate
    grad_total = -(w1 * grad_wall_c + w2 * grad_pair_c)
    
    return grad_total.ravel()


def _make_hexagonal_grid():
    """Create a hexagonal packing initial configuration."""
    # For 50 circles, use a hexagonal arrangement
    # Rows of circles in a hexagonal pattern
    # Spacing: dx = 2r, dy = 2r * sqrt(3)/2 = r*sqrt(3)
    # Let's use a grid of rows
    # Try different row counts
    best_c = None
    best_score = -1
    
    for n_rows in range(5, 11):
        # Distribute 50 circles into n_rows
        base = N // n_rows
        extra = N % n_rows
        row_sizes = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        # Calculate positions
        c = np.zeros((N, 2))
        idx = 0
        for i, n_cols in enumerate(row_sizes):
            # Row i is at y position
            # Center the rows vertically
            y = (i - (n_rows - 1) / 2.0) * 0.15  # approximate spacing
            # Offset odd rows
            x_offset = (i % 2) * 0.07
            for j in range(n_cols):
                x = (j - (n_cols - 1) / 2.0) * 0.14 + x_offset
                c[idx, 0] = x
                c[idx, 1] = y
                idx += 1
        
        # Clip to square
        c = np.clip(c, -0.45, 0.45)
        s = _score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
    
    return best_c


def _make_grid_rect():
    """Create a rectangular grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_refine_smooth(c, deadline, max_iter=100):
    """
    Refine the packing using L-BFGS-B with the soft-min objective.
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 120:
        return c
    
    c = c.copy().astype(float)
    x0 = c.ravel()
    
    try:
        res = minimize(
            _soft_min_radius_neg,
            x0,
            jac=_soft_min_radius_neg_grad,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-10, 'maxls': 20}
        )
        if res.fun < _soft_min_radius_neg(x0):
            c = res.x.reshape(-1, 2)
            c = np.clip(c, -0.5, 0.5)
    except Exception:
        pass
    
    return c


def _relax_hard_disk(c, max_iters=300):
    """
    Hard-disk relaxation: iteratively move circles to increase the minimum distance.
    """
    c = c.copy().astype(float)
    for _ in range(max_iters):
        # Compute current radius
        wall = 0.5 - np.abs(c).max()
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = dist.min() / 2.0
        r = min(wall, min_pair)
        
        if r < 1e-15:
            break
        
        # Find the most constrained pair or wall
        if wall <= min_pair:
            # Wall is the bottleneck
            idx = np.unravel_index(np.abs(c).argmax(), c.shape)
            c[idx] -= np.sign(c[idx]) * 1e-7
        else:
            # Pair is the bottleneck
            i, j = np.unravel_index(dist.argmin(), dist.shape)
            if i == j:
                continue
            d = c[i] - c[j]
            norm = np.linalg.norm(d)
            if norm < 1e-15:
                d = np.random.randn(2) * 1e-7
                norm = np.linalg.norm(d)
            d = d / norm
            # Move them apart
            gap = (2 * r - norm) / 2.0
            if gap > 0:
                c[i] += d * gap * 0.3
                c[j] -= d * gap * 0.3
        
        # Clip to square
        c = np.clip(c, -0.5, 0.5)
    
    return c


def _voronoi_style_refine(c, deadline):
    """
    Refine by moving each circle to the center of its Voronoi cell (approximated).
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 120:
        return c
    
    c = c.copy().astype(float)
    for _ in range(50):
        if time.time() - deadline[0] > 120:
            break
        for i in range(N):
            # Find the closest other circle
            diffs = c - c[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            j = dists.argmin()
            # Move slightly away from j
            d = c[i] - c[j]
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d = d / norm
                # Move by a small fraction
                c[i] += d * 0.001
            # Also move away from walls
            for k in range(2):
                if c[i, k] > 0:
                    c[i, k] -= 0.001
                elif c[i, k] < 0:
                    c[i, k] += 0.001
        c = np.clip(c, -0.5, 0.5)
    
    return c


def _local_search(c, deadline, n_steps=1000):
    """
    Local search (hill climbing) with random perturbations.
    """
    c = c.copy().astype(float)
    best_score = _score(c)
    
    for step in range(n_steps):
        if time.time() - deadline[0] > 120:
            break
        
        # Random perturbation
        c_new = c.copy()
        i = np.random.randint(N)
        scale = 0.01 * (1 + step / 500.0)
        c_new[i] += np.random.randn(2) * scale
        c_new = np.clip(c_new, -0.5, 0.5)
        
        s = _score(c_new)
        if s > best_score:
            c = c_new
            best_score = s
    
    return c


def construct():
    deadline = [time.time()]
    total_time_limit = 150.0
    
    # Phase 1: Generate multiple initial configurations
    c_hex = _make_hexagonal_grid()
    c_grid = _make_grid_rect()
    
    best_c = c_hex.copy()
    best_score = _score(best_c)
    
    s_grid = _score(c_grid)
    if s_grid > best_score:
        best_c = c_grid.copy()
        best_score = s_grid
    
    # Phase 2: Hard-disk relaxation
    c = _relax_hard_disk(best_c, max_iters=200)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 3: Smooth refinement with L-BFGS-B
    c = _local_refine_smooth(best_c, deadline, max_iter=150)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 4: Local search (hill climbing)
    c = _local_search(best_c, deadline, n_steps=2000)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 5: Another round of smooth refinement
    c = _local_refine_smooth(best_c, deadline, max_iter=100)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 6: Final local search
    c = _local_search(best_c, deadline, n_steps=1000)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 7: Final polish
    c = _local_refine_smooth(best_c, deadline, max_iter=50)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
