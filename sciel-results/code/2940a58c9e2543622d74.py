from scipy.spatial.distance import cdist
import numpy as np
from scipy.optimize import minimize
import time

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 115  # Safety margin

def _calculate_radius(pts):
    """
    Calculate the maximum radius for a set of points.
    pts: (N, 2) array.
    Returns radius or 0 if invalid.
    """
    if len(pts) != N:
        return 0.0
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
       
    # Min distance to wall
    wall_dist = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    
    # Min pairwise distance
    # Using cdist is fast for N=200
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    r = min(wall_dist, min_pair / 2.0)
    return r if r > 0 else 0.0

def _penalty_func(pts_flat, r):
    """
    Penalty function for L-BFGS-B.
    Minimizes the amount of overlap.
    pts_flat: (2N,) array.
    r: target radius.
    """
    pts = pts_flat.reshape(-1, 2)
    
    # Wall penalty
    # If point is closer than r to wall, penalty is (r - dist)^2
    # dist to left wall: x + 0.5
    # dist to right wall: 0.5 - x
    # dist to bottom wall: y + 0.5
    # dist to top wall: 0.5 - y
    
    dist_left = pts[:, 0] + 0.5
    dist_right = 0.5 - pts[:, 0]
    dist_bottom = pts[:, 1] + 0.5
    dist_top = 0.5 - pts[:, 1]
    
    # We want dist >= r.
    # Penalty if dist < r: (r - dist)^2
    pen_wall = 0.0
    pen_wall += np.sum(np.maximum(0, r - dist_left) ** 2)
    pen_wall += np.sum(np.maximum(0, r - dist_right) ** 2)
    pen_wall += np.sum(np.maximum(0, r - dist_bottom) ** 2)
    pen_wall += np.sum(np.maximum(0, r - dist_top) ** 2)
    
    # Pairwise penalty
    # dist >= 2r
    # Penalty if dist < 2r: (2r - dist)^2
    # Vectorized calculation of all pair distances
    # N=200, 200x200 matrix.
    # cdist is C-optimized.
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # Only upper triangle to avoid double counting?
    # Summing all is fine, just 2x the value.
    # We want dist >= 2r.
    pen_pair = np.sum(np.maximum(0, 2*r - D) ** 2)
    
    return pen_wall + pen_pair

def _grad_penalty_func(pts_flat, r):
    """
    Gradient of the penalty function.
    """
    pts = pts_flat.reshape(-1, 2)
    N_pts = len(pts)
    grad = np.zeros_like(pts)
    
    # Wall gradient
    # d/dx (r - (x+0.5))^2 = 2(r - (x+0.5)) * (-1) = -2(r - x - 0.5)
    # If r > x+0.5, grad_x = 2(x+0.5-r). This pushes x to the right (increases x).
    # Wait. Penalty is (r - dist)^2.
    # If dist < r, we want to increase dist.
    # dist_left = x + 0.5. Increase x increases dist_left.
    # d/dx (r - (x+0.5))^2 = -2(r - x - 0.5).
    # If r > x+0.5, term is positive? No.
    # Let u = r - (x+0.5). u > 0.
    # d/dx u^2 = 2u * du/dx = 2u * (-1) = -2u.
    # So grad_x += -2(r - x - 0.5) = 2(x + 0.5 - r).
    # This is negative if x+0.5 < r. So it pushes x to the right?
    # Minimization moves in direction -grad.
    # If grad is negative, -grad is positive. So x increases. Correct.
    
    mask_left = pts[:, 0] + 0.5 < r
    if np.any(mask_left):
        grad[mask_left, 0] += 2 * (pts[mask_left, 0] + 0.5 - r)
        
    mask_right = 0.5 - pts[:, 0] < r
    if np.any(mask_right):
        # dist_right = 0.5 - x. Increase dist_right means decrease x.
        # u = r - (0.5 - x). du/dx = 1.
        # d/dx u^2 = 2u.
        # grad_x += 2(r - 0.5 + x).
        # If 0.5-x < r, then x > 0.5-r.
        # grad is positive? r - 0.5 + x. If x is close to 0.5, this is positive.
        # Minimization moves in -grad (negative). So x decreases. Correct.
        grad[mask_right, 0] += 2 * (r - 0.5 + pts[mask_right, 0])
        
    mask_bottom = pts[:, 1] + 0.5 < r
    if np.any(mask_bottom):
        grad[mask_bottom, 1] += 2 * (pts[mask_bottom, 1] + 0.5 - r)
        
    mask_top = 0.5 - pts[:, 1] < r
    if np.any(mask_top):
        grad[mask_top, 1] += 2 * (r - 0.5 + pts[mask_top, 1])
        
    # Pairwise gradient
    # D is N x N.
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # We need d/dx_i sum_{j != i} max(0, 2r - D_ij)^2
    # Let u_ij = 2r - D_ij.
    # If u_ij > 0, term is u_ij^2.
    # d/dx_i u_ij^2 = 2 u_ij * d/dx_i (2r - D_ij) = -2 u_ij * d/dx_i D_ij.
    # D_ij = ||p_i - p_j||.
    # d/dx_i D_ij = (p_i - p_j) / D_ij.
    # So grad_i += -2 * (2r - D_ij) * (p_i - p_j) / D_ij for all j where D_ij < 2r.
    
    # Vectorized:
    # We want to compute sum over j.
    # Create mask where D < 2r
    mask = D < 2*r
    np.fill_diagonal(mask, False)
    
    if np.any(mask):
        # We need to sum -2 * (2r - D) * (p_i - p_j) / D for each i.
        # This is a bit tricky to vectorize fully without loops for the matrix structure,
        # but N=200 is small.
        # Let's use a loop over i for clarity and sufficient speed.
        for i in range(N_pts):
            # Indices j where mask[i,j] is True
            js = np.where(mask[i, :])[0]
            if len(js) == 0: continue
            
            # D[i, js]
            dists = D[i, js]
            # (p_i - p_j)
            diffs = pts[i] - pts[js] # Shape (len(js), 2)
            # Term: -2 * (2r - dist) * diff / dist
            # Note: (p_i - p_j) points from j to i.
            # We want to push i away from j.
            # Force on i should be in direction (p_i - p_j).
            # Gradient of potential energy (repulsive) is in direction (p_i - p_j).
            # Let's re-verify sign.
            # Penalty P = (2r - D)^2.
            # dP/dp_i = 2(2r-D) * d(2r-D)/dp_i = 2(2r-D) * (-(p_i-p_j)/D) = -2(2r-D)(p_i-p_j)/D.
            # So grad += -2(2r-D)(p_i-p_j)/D.
            # Since 2r-D > 0, and D > 0, the sign is opposite to (p_i-p_j).
            # So grad points from i to j?
            # Minimization moves in -grad.
            # -grad points from j to i. So it pushes i away from j. Correct.
            
            coeffs = -2 * (2*r - dists) / dists
            # coeffs shape (len(js),)
            # diffs shape (len(js), 2)
            grad_i = np.sum(coeffs[:, np.newaxis] * diffs, axis=0)
            grad[i] += grad_i
            
    return grad.flatten()

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # Helper to run the squeezing optimization
    def run_squeeze(init_pts, r_start, r_end, max_iter=50):
        nonlocal best_r, best_pts
        
        if time.time() - t0 > _TIME_LIMIT:
            return
            
        pts = init_pts.copy()
        r = r_start
        
        # We will try to increase r in small steps
        # Or use a single optimization with a large r and let it fail?
        # Better: incremental increase.
        
        n_steps = 20
        dr = (r_end - r_start) / n_steps
        
        for step in range(n_steps):
            if time.time() - t0 > _TIME_LIMIT:
                break
                
            r_target = r_start + (step + 1) * dr
            
            # Define bounds for L-BFGS-B
            # Points must be inside [-0.5, 0.5]
            # Actually, if they go outside, wall penalty is huge.
            # But L-BFGS-B needs bounds to be efficient.
            # Let's use bounds [-0.5, 0.5] for all coordinates.
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            # Initial guess is current pts
            x0 = pts.flatten()
            
            try:
                res = minimize(
                    _penalty_func, 
                    x0, 
                    args=(r_target,), 
                    method='L-BFGS-B', 
                    jac=_grad_penalty_func,
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-8:
                    # Accept this radius
                    r = r_target
                    pts = res.x.reshape(-1, 2)
                    
                    # Check actual radius
                    r_actual = _calculate_radius(pts)
                    if r_actual > best_r:
                        best_r = r_actual
                        best_pts = pts.copy()
                else:
                    # If it didn't converge to low penalty, maybe we can't reach r_target
                    # Check if the resulting config is still valid and better
                    pts_candidate = res.x.reshape(-1, 2)
                    r_candidate = _calculate_radius(pts_candidate)
                    if r_candidate > best_r:
                        best_r = r_candidate
                        best_pts = pts_candidate.copy()
                        # Update pts to this candidate for next iteration?
                        # It's safer to stick to the last successful r
                        pts = pts_candidate
                    # Break or continue?
                    # If we can't squeeze to r_target, we probably can't go much further.
                    # But maybe with a different start?
                    # For now, break the squeeze loop for this init.
                    break
                    
            except Exception:
                break
                
        # After squeezing, do a final local polish with the current best r
        if best_pts is not None and time.time() - t0 < _TIME_LIMIT:
            r_current = best_r
            # Try to optimize at r_current + small epsilon
            r_polish = r_current * 1.01
            x0 = best_pts.flatten()
            try:
                res = minimize(
                    _penalty_func, 
                    x0, 
                    args=(r_polish,), 
                    method='L-BFGS-B', 
                    jac=_grad_penalty_func,
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < 1e-8:
                    pts_new = res.x.reshape(-1, 2)
                    r_new = _calculate_radius(pts_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_pts = pts_new
            except Exception:
                pass

    # Generate initial configurations
    
    # 1. Hexagonal Grid
    # We want ~200 points.
    # Hex grid spacing d.
    # Area per point ~ d^2 * sqrt(3)/2.
    # 200 * d^2 * 0.866 ~ 1 => d ~ sqrt(1/173) ~ 0.076.
    # Radius ~ 0.038.
    
    def make_hex_grid(spacing):
        pts = []
        # Generate points in a large area, then clip
        # Rows
        num_rows = int(1.0 / (spacing * np.sqrt(3)/2)) + 2
        num_cols = int(1.0 / spacing) + 2
        
        for i in range(-1, num_rows + 1):
            y = i * spacing * np.sqrt(3) / 2
            offset = 0 if i % 2 == 0 else spacing / 2
            for j in range(-1, num_cols + 1):
                x = j * spacing + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) > N:
            # Keep closest to center
            dists = np.sum(pts**2, axis=1)
            indices = np.argsort(dists)[:N]
            pts = pts[indices]
        elif len(pts) < N:
            # Add random points in empty space?
            # For simplicity, just return what we have, or pad with random
            # If < N, we need to add points.
            # Let's add random points inside the square.
            while len(pts) < N:
                p = np.random.uniform(-0.5, 0.5, 2)
                pts = np.vstack([pts, p])
                
        return pts

    # Try a few spacings
    spacings = [0.07, 0.072, 0.074, 0.076, 0.078]
    
    for sp in spacings:
        if time.time() - t0 > _TIME_LIMIT:
            break
        init_pts = make_hex_grid(sp)
        # Estimate initial radius
        r_init = _calculate_radius(init_pts)
        # Squeeze from r_init to r_init * 1.2
        run_squeeze(init_pts, r_init, r_init * 1.2)
        
    # 2. Square Grid
    def make_square_grid(spacing):
        pts = []
        num = int(1.0 / spacing) + 2
        for i in range(-1, num + 1):
            for j in range(-1, num + 1):
                x = i * spacing
                y = j * spacing
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.sum(pts**2, axis=1)
            indices = np.argsort(dists)[:N]
            pts = pts[indices]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.5, 0.5, 2)
                pts = np.vstack([pts, p])
        return pts

    sq_spacings = [0.07, 0.072, 0.074]
    for sp in sq_spacings:
        if time.time() - t0 > _TIME_LIMIT:
            break
        init_pts = make_square_grid(sp)
        r_init = _calculate_radius(init_pts)
        run_squeeze(init_pts, r_init, r_init * 1.2)
        
    # Fallback
    if best_pts is None:
        # Simple grid
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts
