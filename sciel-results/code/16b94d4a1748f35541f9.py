"""Packing 50 equal circles in the unit square [-0.5, 0.5]^2 using Binary Search and L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5

def _score(pts):
    """Return the achievable common radius for a centre array (N,2)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary term
    d_bound = np.min(HALF - np.abs(pts), axis=1)
    r = float(np.min(d_bound))
    # Pairwise term
    if N > 1:
        # Calculate pairwise distances
        dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        dmin = float(np.min(dist))
        r = min(r, dmin / 2.0)
    return r

def _valid(pts):
    return (pts.shape == (N, 2)
            and np.all(np.isfinite(pts))
            and np.all(pts >= -0.5 - 1e-9)
            and np.all(pts <= 0.5 + 1e-9))

def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        # Fallback to grid if hex doesn't have enough points
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)

def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)

def _energy_and_grad(x, r, mu=1.0):
    """
    Calculate energy and gradient for L-BFGS-B.
    x is a flat array of shape (2N,).
    r is the target radius.
    mu is the weight for boundary violations.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Overlap violations: max(0, 2r - dist)^2
    # We only care about pairs where dist < 2r
    overlap = np.maximum(0.0, 2.0 * r - dist)
    overlap[~np.isfinite(overlap)] = 0.0 # Just in case, though diag is inf
    
    # Energy from overlaps
    E_pair = np.sum(overlap ** 2)
    
    # Gradient of E_pair with respect to pts
    # d(E_pair)/d(pts_i) = sum_{j != i} 2 * overlap_ij * d(overlap_ij)/d(pts_i)
    # overlap_ij = max(0, 2r - dist_ij)
    # d(overlap_ij)/d(pts_i) = -d(dist_ij)/d(pts_i) if dist_ij < 2r else 0
    # d(dist_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
    
    grad_pts = np.zeros_like(pts)
    # Mask for pairs that are overlapping
    mask = dist < 2.0 * r
    # Set diagonal to False to avoid self-interaction
    np.fill_diagonal(mask, False)
    
    if np.any(mask):
        # Compute gradients only for overlapping pairs
        # For each pair (i, j) where mask[i,j] is True:
        # Contribution to grad[i] += 2 * overlap[i,j] * (pts[i] - pts[j]) / dist[i,j]
        # Contribution to grad[j] += 2 * overlap[i,j] * (pts[j] - pts[i]) / dist[i,j]
        
        # Vectorized approach:
        # inv_dist = 1 / dist where mask is true, 0 otherwise
        inv_dist = np.where(mask, 1.0 / np.where(dist > 1e-12, dist, 1e-12), 0.0)
        
        # Force magnitude for each pair
        # F_ij = 2 * overlap_ij * inv_dist_ij
        F = 2.0 * overlap * inv_dist
        
        # grad_pts[i] = sum_j F_ij * (pts[i] - pts[j])
        # This is equivalent to sum_j F_ij * pts[i] - sum_j F_ij * pts[j]
        # sum_j F_ij is a vector of sums over j for each i
        
        sum_F = np.sum(F, axis=1)
        # sum_j F_ij * pts[j]
        F_pts = F @ pts
        
        grad_pts = sum_F[:, None] * pts - F_pts
        
    # Boundary violations
    # Constraint: |x_k| <= HALF - r
    # Violation: max(0, |x_k| - (HALF - r))^2
    limit = HALF - r
    abs_pts = np.abs(pts)
    viol = np.maximum(0.0, abs_pts - limit)
    E_bound = mu * np.sum(viol ** 2)
    
    # Gradient of E_bound
    # d/dx (max(0, |x| - limit)^2) = 2 * viol * sign(x)
    grad_bound = mu * 2.0 * viol * np.sign(pts)
    
    grad_pts += grad_bound
    
    total_energy = E_pair + E_bound
    total_grad = grad_pts.flatten()
    
    return total_energy, total_grad

def _is_feasible(x, r, tol=1e-6):
    """Check if configuration x is feasible for radius r."""
    pts = x.reshape(N, 2)
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    min_bound = np.min(HALF - np.abs(pts))
    
    return (min_dist >= 2.0 * r - tol) and (min_bound >= r - tol)

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial bounds
    r_low = 0.06
    r_high = 0.08
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    for it in range(35):
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # Use multiple seeds: previous best, hex, grid
        seeds = [best_pts.copy(), _hex_seed(r_mid), _grid_seed(r_mid)]
        
        # Add some random jitter to seeds
        for seed in seeds:
            seed = seed + rng.normal(0, 0.01, size=(N, 2))
            seed = np.clip(seed, -HALF + r_mid, HALF - r_mid)
            
        feasible_found = False
        best_feasible_pts = None
        best_feasible_energy = np.inf
        
        for seed in seeds:
            if time.time() > deadline - 5.0:
                break
                
            x0 = seed.flatten()
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _energy_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                
                energy = res.fun
                x_opt = res.x
                
                if energy < best_feasible_energy:
                    best_feasible_energy = energy
                    best_feasible_pts = x_opt.copy()
                    
                # Check if it's actually feasible
                if _is_feasible(x_opt, r_mid, tol=1e-5):
                    feasible_found = True
                    break # Found a feasible one, no need to try other seeds for this r_mid
                
            except Exception as e:
                pass
        
        if feasible_found:
            r_low = r_mid
            best_pts = best_feasible_pts.reshape(N, 2).copy()
            # Update best_r based on the actual score, which might be slightly higher than r_mid
            # But for the binary search logic, we just need to know r_mid is feasible
            # We can update best_pts to the one that gives the best score
            # Actually, the feasible point for r_mid is a valid configuration.
            # Its score is at least r_mid.
            # Let's compute the score to see if it's better than our current best_r
            current_score = _score(best_pts)
            if current_score > best_r:
                best_r = current_score
        else:
            r_high = r_mid
        
        # If we have time, we can try to polish the current best_pts
        # But the binary search will naturally find higher radii if possible.
        
    # Final refinement: Try to push the current best_pts a bit more
    # Use the current best_r as a target and try to improve it slightly
    if time.time() < deadline - 10:
        target_r = best_r + 0.001
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                args=(target_r,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x
            if _is_feasible(x_opt, target_r, tol=1e-5):
                best_pts = x_opt.reshape(N, 2)
                best_r = _score(best_pts)
        except:
            pass

    # Final safety check
    best_pts = np.clip(best_pts, -0.5, 0.5)
    if not _valid(best_pts):
        best_pts = _grid_initial() # Fallback, though _grid_seed is better defined above
        # Actually, let's just use a simple grid fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pts

# Helper for fallback
def _grid_initial():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()
