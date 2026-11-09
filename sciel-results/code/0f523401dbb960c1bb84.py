"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validity_check(x, y, r, tol=1e-9):
    """Returns True if configuration is valid, False otherwise"""
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized overlap check
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(dist < r[i_idx] + r[j_idx] - tol):
        return False
    return True

def _make_valid(x, y, r, tol=1e-9):
    """Ensure configuration is valid by shrinking radii if needed"""
    r = r.copy()
    # First ensure bounds
    for i in range(len(r)):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        if r[i] > max_r:
            r[i] = max(0, max_r)
    
    i_idx, j_idx = np.triu_indices(len(r), k=1)
    
    # Iteratively fix overlaps
    for _ in range(500):
        # Vectorized overlap calculation
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        overlaps = r[i_idx] + r[j_idx] - dist
        
        max_overlap = np.max(overlaps)
        if max_overlap <= tol:
            break
        
        # Find the pair with max overlap
        max_pair_idx = np.argmax(overlaps)
        max_i = i_idx[max_pair_idx]
        max_j = j_idx[max_pair_idx]
        
        # Shrink the larger circle by half the overlap plus a small buffer
        shrink = max_overlap / 2 + 1e-7
        if r[max_i] >= r[max_j]:
            r[max_i] = max(0, r[max_i] - shrink)
        else:
            r[max_j] = max(0, r[max_j] - shrink)
    
    # Final safety: if still invalid, scale down all radii
    if not _validity_check(x, y, r, tol):
        r *= 0.95
        for _ in range(50):
            if _validity_check(x, y, r, tol):
                break
            r *= 0.9
        
    return x, y, r

def _maximize_radii(x, y, r, iterations=50, tol=1e-10):
    """Try to increase each radius to its maximum possible value"""
    r = r.copy()
    for _ in range(iterations):
        changed = False
        for i in range(len(r)):
            # Max radius from boundary
            r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
            # Max radius from neighbors
            r_neighbor = 0.5
            for j in range(len(r)):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_neighbor = min(r_neighbor, max(0, dist - r[j] - tol))
            
            new_r = min(r_bound, r_neighbor)
            if new_r > r[i] + 1e-12:
                r[i] = new_r
                changed = True
        
        if not changed:
            break
    
    # Ensure validity after maximization
    if not _validity_check(x, y, r, 1e-9):
        x, y, r = _make_valid(x, y, r)
    
    return x, y, r

def _hexagonal_lattice(n, margin=0.05):
    """
    Generate n points on a hexagonal lattice inside [margin, 1-margin]^2.
    Returns array of shape (n, 2).
    """
    points = []
    # Hexagonal lattice with spacing s
    # For a hex lattice, horizontal spacing = s, vertical spacing = s * sqrt(3)/2
    # We want to pack as many as possible in the square.
    
    # Let's try to fit a grid of hex points.
    # Number of rows and columns.
    # Approximate area per point ~ s^2 * sqrt(3)/2
    # Total area ~ (1 - 2*margin)^2
    # s ~ sqrt(2 * area / (n * sqrt(3)))
    
    side = 1.0 - 2.0 * margin
    s = np.sqrt(2.0 * side * side / (n * np.sqrt(3)))
    
    row_spacing = s * np.sqrt(3) / 2
    col_spacing = s
    
    # Determine number of rows and columns
    n_rows = int(np.floor(side / row_spacing)) + 1
    n_cols = int(np.floor(side / col_spacing)) + 1
    
    # Generate points
    for i in range(n_rows):
        y = margin + i * row_spacing
        offset = (i % 2) * (col_spacing / 2.0)
        for j in range(n_cols):
            x = margin + j * col_spacing + offset
            if margin <= x <= 1.0 - margin and margin <= y <= 1.0 - margin:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n points, randomly sample n
    if len(points) > n:
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        while len(points) < n:
            points = np.vstack([points, np.array([np.random.uniform(margin, 1-margin), 
                                                   np.random.uniform(margin, 1-margin)])])
    
    return points[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_result = None
    best_score = -np.inf
    
    # Precompute indices for vectorized overlap calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        We want to maximize sum(r), so we minimize -sum(r).
        We add penalties for boundary violations and circle overlaps.
        Vectorized for speed.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure radii are positive
        r_pos = np.maximum(r, 1e-6)
        
        # 1. Boundary Penalties
        pen_x_left = np.maximum(0, r_pos - x)**2
        pen_x_right = np.maximum(0, x - (1 - r_pos))**2
        pen_y_bottom = np.maximum(0, r_pos - y)**2
        pen_y_top = np.maximum(0, y - (1 - r_pos))**2
        boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # 2. Overlap Penalties (Vectorized)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx*dx + dy*dy)
        overlaps = np.maximum(0, r_pos[i_idx] + r_pos[j_idx] - d)
        overlap_pen = np.sum(overlaps**2)
        
        # Weights: High weight to ensure validity is prioritized
        w_pen = 100.0
        
        return -np.sum(r_pos) + w_pen * (boundary_pen + overlap_pen)
    
    def optimize_start(x0, y0, r0):
        """
        Run L-BFGS-B optimization from a given start.
        Returns the best valid configuration found.
        """
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Try to repair by making valid
            x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
            
            # Try to maximize radii further
            x_opt, y_opt, r_opt = _maximize_radii(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None

    def polish(x, y, r):
        """
        High-precision polish pass.
        Runs L-BFGS-B with tighter tolerances and more iterations.
        """
        params0 = np.concatenate([x, y, r])
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Repair and maximize
            x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
            x_opt, y_opt, r_opt = _maximize_radii(x_opt, y_opt, r_opt, iterations=50)
            
            score = np.sum(r_opt)
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return np.sum(r), np.column_stack((x, y, r))
    
    def local_search(x, y, r, num_iter=20):
        """
        Simple local search: perturb positions/radii, re-optimize.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for _ in range(num_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            perturb_scale = 0.05
            x_new = x + np.random.normal(0, perturb_scale, N)
            y_new = y + np.random.normal(0, perturb_scale, N)
            r_new = r + np.random.normal(0, 0.01, N)
            
            # Keep within bounds
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            r_new = np.clip(r_new, 0, 0.5)
            
            # Optimize from perturbed state
            score, config = optimize_start(x_new, y_new, r_new)
            
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_new, y_new, r_new
                best_x = config[:, 0]
                best_y = config[:, 1]
                best_r = config[:, 2]
        
        return best_score, np.column_stack((best_x, best_y, best_r))
    
    # Try multiple random restarts with different initializations
    n_restarts = 15
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with hexagonal lattice + random perturbation
        x0 = _hexagonal_lattice(N, margin=0.05)
        # Add small random perturbation
        x0 += rng.uniform(-0.02, 0.02, size=x0.shape)
        
        # Initial radii: small
        r0 = np.full(N, 0.03)
        
        # Optimize
        score, config = optimize_start(x0[:, 0], x0[:, 1], r0)
        
        if config is not None and score > best_score:
            best_score = score
            best_result = config
        
        # If we have time, do local search and polish
        if time.time() < deadline - 20.0 and best_result is not None:
            x_cur = best_result[:, 0]
            y_cur = best_result[:, 1]
            r_cur = best_result[:, 2]
            
            # Local search
            ls_score, ls_config = local_search(x_cur, y_cur, r_cur, num_iter=10)
            
            if ls_score > best_score:
                best_score = ls_score
                best_result = ls_config
            
            # Polish
            if time.time() < deadline - 15.0:
                pol_score, pol_config = polish(best_result[:, 0], best_result[:, 1], best_result[:, 2])
                
                if pol_score > best_score:
                    best_score = pol_score
                    best_result = pol_config
    
    # If no valid result found, return a trivial valid packing
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
