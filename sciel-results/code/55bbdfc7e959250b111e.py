"""
Combined packer: Hex-lattice seed + Soft-min L-BFGS-B + Active-set local polishing.
"""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # Safety margin before the 150 s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance using KD-tree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * np.min(dists[:, 1])
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing, rng):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Add small noise to break symmetry
                x += rng.normal(0, spacing * 0.05)
                y += rng.normal(0, spacing * 0.05)
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _soft_min_obj_and_grad(x, beta, active_pairs=None, active_bounds=None):
    """
    Compute soft-min objective and gradient.
    If active_pairs is provided, only those pairs are considered.
    If active_bounds is provided, only those points' boundary constraints are considered.
    """
    pts = x.reshape(N, 2)
    
    # Identify which constraints are active
    if active_pairs is None:
        # All pairs
        iu = np.triu_indices(N, 1)
        active_pairs = iu
    if active_bounds is None:
        # All bounds
        active_bounds = np.arange(N)
    else:
        active_bounds = np.array(active_bounds)

    # Boundary terms
    norms = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - norms[active_bounds]
    boundary_grads = -pts[active_bounds] / norms[active_bounds, np.newaxis]
    
    # Pairwise terms
    i_idx, j_idx = active_pairs
    diff = pts[i_idx] - pts[j_idx]
    pair_dists = np.linalg.norm(diff, axis=1)
    
    # Avoid division by zero
    mask = pair_dists > 1e-10
    pair_dists_safe = np.where(mask, pair_dists, 1.0)
    unit_pairs = diff / pair_dists_safe[:, np.newaxis]
    unit_pairs[~mask] = 0.0

    # Combine all terms
    # We want to maximize min(distances).
    # Objective to minimize: (1/beta) * log(sum(exp(-beta * d)))
    
    # For numerical stability, we shift by the max distance
    all_dists = np.concatenate([pair_dists, boundary_terms])
    max_d = np.max(all_dists)
    
    exp_terms = np.exp(-beta * (all_dists - max_d))
    log_sum_exp = np.log(np.sum(exp_terms)) + beta * max_d
    obj_val = log_sum_exp / beta
    
    # Gradient
    # d(obj)/dx = (1/beta) * sum_j (exp(-beta*d_j)/Z) * d(d_j)/dx
    # Note: The shift max_d cancels out in the ratio exp(-beta*(d_j - max_d)) / sum(...)
    
    Z = np.sum(exp_terms)
    if Z == 0:
        return obj_val, np.zeros_like(x)
    
    weights = exp_terms / Z
    
    # Split weights back
    w_pairs = weights[:len(pair_dists)]
    w_bounds = weights[len(pair_dists):]
    
    grad = np.zeros_like(pts)
    
    # Pairwise gradient contribution
    # For each pair (i, j), weight w:
    # grad[i] += w * unit_ij
    # grad[j] += -w * unit_ij
    np.add.at(grad, i_idx, w_pairs[:, np.newaxis] * unit_pairs)
    np.add.at(grad, j_idx, -w_pairs[:, np.newaxis] * unit_pairs)
    
    # Boundary gradient contribution
    # For each point k in active_bounds, weight w:
    # grad[k] += w * (-x_k / ||x_k||)
    grad[active_bounds] += w_bounds[:, np.newaxis] * boundary_grads
    
    return obj_val, grad.flatten()


def _polish(points, beta, max_iter=500, lr=1e-4, time_limit=10.0):
    """
    Local polishing using gradient descent on active constraints.
    """
    start_time = time.time()
    pts = points.copy()
    r_current = _score(pts)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find active constraints: pairs with distance < 2 * r_current * 1.1 (or similar)
        # and points near boundary
        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=2)
        
        # Pairs
        min_dist = np.min(dists[:, 1])
        threshold = min_dist * 1.05 # Slightly larger than min dist to include near-min pairs
        # Actually, for polishing, we want to push apart the closest pairs.
        # Let's just use the closest pair for each point? No, that might ignore other constraints.
        # Better: Use a fixed number of nearest neighbors, e.g., 6 (hexagonal)
        # Or all pairs within a certain radius.
        
        # Let's use all pairs with distance < 2 * r_current + epsilon
        # r_current is the radius, so distance is 2*r.
        # Let's use a threshold relative to the current best radius.
        # If we don't know the exact global min, use the local min.
        
        # For simplicity and speed, let's just use the closest 6 neighbors for each point.
        # This is O(N * log N) to build tree, O(N * K) to query.
        
        # Re-query with K=7 (self + 6 neighbors)
        dists_k, idx_k = tree.query(pts, k=7)
        
        active_pairs_set = set()
        for i in range(N):
            for j_idx in idx_k[i, 1:]: # Skip self
                if i < j_idx:
                    active_pairs_set.add((i, j_idx))
                else:
                    active_pairs_set.add((j_idx, i))
        
        active_pairs = np.array(list(active_pairs_set))
        if len(active_pairs) == 0:
            break
            
        # Active bounds: points with norm > 1 - 2*r_current
        norms = np.linalg.norm(pts, axis=1)
        active_bounds = np.where(norms > 1.0 - 2.0 * r_current - 0.01)[0]
        if len(active_bounds) == 0:
            active_bounds = np.arange(N) # Fallback
        
        # Compute objective and gradient
        try:
            obj, grad = _soft_min_obj_and_grad(pts.flatten(), beta, active_pairs, active_bounds)
            grad = grad.reshape(N, 2)
            
            # Gradient step
            pts -= lr * grad
            
            # Project inside unit circle
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, None] * (1.0 - 1e-6)
                
        except Exception:
            break
            
        # Check improvement
        r_new = _score(pts)
        if r_new > r_current:
            r_current = r_new
        else:
            # If no improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
                
    return pts, r_current


def _local_polish_exact(pts, max_iters=200, tol=1e-10, time_limit=10.0):
    """
    Local optimization using exact min (not soft-min) with steepest descent.
    Focuses on improving the minimum constraint.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all constraint values
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        # Pair constraints: dists/2
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        bound_vals = boundary_dists
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        min_val = np.min(all_vals)
        
        if min_val < tol:
            break
        
        # Find active constraints (within tolerance of minimum)
        active_mask = np.abs(all_vals - min_val) < 1e-6
        active_pairs = np.where(active_mask[:len(ii)])[0]
        active_bounds = np.where(active_mask[len(ii):])[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build gradient for the minimum constraint
        # Use subgradient: average of gradients of active constraints
        grad = np.zeros_like(pts)
        
        # Pairwise active constraints
        if len(active_pairs) > 0:
            for idx in active_pairs:
                i, j = ii[idx], jj[idx]
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # grad_i = (pts[i] - pts[j]) / (2*d)
                # grad_j = (pts[j] - pts[i]) / (2*d)
                direction = (pts[i] - pts[j]) / d
                grad[i] += direction / (2.0 * len(active_pairs))
                grad[j] -= direction / (2.0 * len(active_pairs))
        
        # Boundary active constraints
        if len(active_bounds) > 0:
            for idx in active_bounds:
                i = active_bounds[idx]
                r = radii[i]
                if r < 1e-10:
                    continue
                # grad_i = -pts[i] / r
                direction = -pts[i] / r
                grad[i] += direction / (2.0 * len(active_bounds))
        
        # Normalize gradient
        grad_norm = np.sqrt(np.sum(grad**2))
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Line search for step size
        best_step = 0.0
        best_obj = min_val
        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        for step in step_sizes:
            new_pts = pts + step * grad
            new_score = _score(new_pts)
            if new_score > best_obj:
                best_obj = new_score
                best_step = step
        
        if best_step == 0.0:
            # Try smaller steps
            for step in [0.0001, 0.0005]:
                new_pts = pts + step * grad
                new_score = _score(new_pts)
                if new_score > best_obj:
                    best_obj = new_score
                    best_step = step
        
        if best_step == 0.0:
            break
        
        pts = pts + best_step * grad
    
    return pts


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial hex-lattice seed
    # Estimate spacing for 300 points in unit circle
    # Area ~ pi * r^2. N * pi * (r/2)^2 ~ pi * 1^2 -> r ~ 2/sqrt(N)
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    # Generate initial seed
    pts = _lattice_seed(spacing, rng)
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    # Optimization Phases
    # Phase 1: L-BFGS-B with annealed beta
    # Phase 2: Local Polishing (soft-min based)
    # Phase 3: Local Polishing (exact min based, from inspiration)
    
    # Beta schedule
    betas = [10, 50, 100, 500, 1000, 5000]
    
    for beta in betas:
        if time.time() - t0 > DEADLINE - 20:
            break
            
        x0 = best_pts.flatten()
        
        # Define objective and gradient for this beta
        # Note: L-BFGS-B requires the objective and gradient to be consistent.
        # We use the full set of constraints here for global search.
        
        def obj(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
            
        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g
            
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxfun': 100, 'gtol': 1e-5})
            if res.success or res.fun < obj(x0):
                new_pts = res.x.reshape(N, 2)
                # Validate and project inside
                norms = np.linalg.norm(new_pts, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / norms[mask, None] * (1.0 - 1e-6)
                    
                r = _score(new_pts)
                if r > best_r:
                    best_r = r
                    best_pts = new_pts
        except Exception:
            pass
            
        # Check time
        if time.time() - t0 > DEADLINE - 10:
            break
            
    # Phase 2: Local Polishing (soft-min based)
    # Use a high beta for polishing to focus on the minimum
    if time.time() - t0 < DEADLINE - 5:
        polished_pts, polished_r = _polish(best_pts, beta=1000, max_iter=1000, lr=1e-4, time_limit=10.0)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts
            
        # Try polishing again with a different beta or step size if time permits
        if time.time() - t0 < DEADLINE - 2:
            polished_pts2, polished_r2 = _polish(best_pts, beta=5000, max_iter=500, lr=1e-5, time_limit=5.0)
            if polished_r2 > best_r:
                best_r = polished_r2
                best_pts = polished_pts2

    # Phase 3: Exact min polishing (from inspiration)
    if time.time() - t0 < DEADLINE - 1:
        polished_pts3 = _local_polish_exact(best_pts, max_iters=100, time_limit=5.0)
        r3 = _score(polished_pts3)
        if r3 > best_r:
            best_r = r3
            best_pts = polished_pts3

    # Final check
    final_r = _score(best_pts)
    if final_r <= 0.0:
        # Fallback
        best_pts = _lattice_seed(spacing, rng)
        
    return best_pts
