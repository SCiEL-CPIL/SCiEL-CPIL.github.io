import numpy as np
import time

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Try a few different spacings/jitters to find a good start
    for attempt in range(5):
        # Estimate spacing
        # Area per circle ~ pi*r^2. Total area pi. N circles.
        # r ~ 1/sqrt(N). Spacing ~ 2r.
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.95 + 0.02 * attempt)
        
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        
        if len(rows) < n_points:
            continue
        
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        
        # Select points closest to center first
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]].copy()
        
        # Add small jitter to break symmetry
        if attempt > 0:
            selected += rng.normal(0, spacing * 0.05, selected.shape)
            # Ensure inside
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            if np.sum(mask) < n_points:
                continue
            selected = selected[mask][:n_points]
        
        # Compute radius
        norms = np.linalg.norm(selected, axis=1)
        min_bound = 1.0 - norms.max()
        
        # Quick radius check
        # Use cdist for speed in init
        from scipy.spatial.distance import cdist
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
        
    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    # Use cdist for robustness and speed in final eval
    from scipy.spatial.distance import cdist
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Initialization
    pts = _hex_init(N, seed=42)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # 2. Grid-based Repulsion Optimization
    # We minimize energy E = sum(1/d^p) + lambda * sum(1/(1-r)^p)
    # This drives points to maximize min distance.
    
    # Adaptive parameters
    momentum = 0.5
    lr = 0.01
    p_power = 4.0  # Exponent for repulsion
    cutoff_mult = 3.0  # Cutoff distance relative to current min radius
    
    # Precompute grid structure parameters
    # We will rebuild grid every few steps or dynamically
    
    for step in range(50000):
        if time.time() > deadline:
            break
            
        # Compute current min radius to set cutoffs
        norms = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - norms.max()
        
        # Estimate min pair distance quickly using grid
        # Cell size should be around current min distance
        # Estimate min distance from previous step or use a fixed heuristic
        # Let's use a dynamic cell size based on best_r so far
        cell_size = max(2.0 * best_r * cutoff_mult, 0.01)
        
        # Build Grid
        # Map points to grid cells
        grid = {}
        for i in range(N):
            cx = int(pts[i, 0] / cell_size)
            cy = int(pts[i, 1] / cell_size)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        forces = np.zeros_like(pts)
        
        # Compute Pairwise Forces
        # Iterate over all points, check neighbors in 3x3 grid block
        for i in range(N):
            cx = int(pts[i, 0] / cell_size)
            cy = int(pts[i, 1] / cell_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if j <= i:
                                continue
                            # Calculate distance
                            dx_val = pts[i, 0] - pts[j, 0]
                            dy_val = pts[i, 1] - pts[j, 1]
                            dist_sq = dx_val*dx_val + dy_val*dy_val
                            dist = np.sqrt(dist_sq)
                            
                            if dist < 1e-6:
                                # Jitter
                                forces[i, 0] += 0.01
                                forces[j, 0] -= 0.01
                                continue
                            
                            # Cutoff
                            cutoff = 2.0 * best_r * cutoff_mult
                            if dist > cutoff:
                                continue
                            
                            # Force magnitude: 1/dist^p
                            # Gradient of 1/d^p is -p * 1/d^(p+2) * unit_vec
                            # We want to MINIMIZE energy, so force is -grad
                            # Energy term: 1/d^p
                            # dE/dd = -p/d^(p+1)
                            # Force vector on i: -dE/dxi = - (dE/dd) * (xi-xj)/d
                            # = - (-p/d^(p+1)) * (xi-xj)/d = p/d^(p+2) * (xi-xj)
                            
                            # We want to push them apart.
                            # Force on i is in direction (xi-xj).
                            f_mag = p_power / (dist ** (p_power + 2.0))
                            forces[i, 0] += f_mag * dx_val
                            forces[i, 1] += f_mag * dy_val
                            forces[j, 0] -= f_mag * dx_val
                            forces[j, 1] -= f_mag * dy_val
        
        # Compute Boundary Forces
        # Energy term for boundary: 1/(1-r)^p
        # dE/dr = p/(1-r)^(p+1)
        # Force on i: -dE/dxi = - (dE/dr) * (xi/r)
        # = - p/(1-r)^(p+1) * xi/r
        # This pushes inward.
        
        cutoff_bound = 1.0 - 2.0 * best_r * cutoff_mult
        for i in range(N):
            r = norms[i]
            if r < 1e-6:
                continue
            if r > cutoff_bound:
                # Only apply if close to boundary
                # Force magnitude
                f_mag = p_power / ((1.0 - r) ** (p_power + 1.0))
                forces[i, 0] -= f_mag * pts[i, 0] / r
                forces[i, 1] -= f_mag * pts[i, 1] / r
        
        # Update positions
        # Adaptive learning rate
        # If forces are large, reduce lr
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1000:
            lr_eff = lr * (1000.0 / max_force)
        else:
            lr_eff = lr
        
        # Momentum update
        # We don't store velocity here to save memory, but simple momentum helps
        # For simplicity and stability, we use direct gradient step with damping
        # Actually, let's add a velocity buffer
        if not hasattr(construct, '_vel'):
            construct._vel = np.zeros_like(pts)
        
        construct._vel = momentum * construct._vel + (1 - momentum) * forces
        pts += lr_eff * construct._vel
        
        # Project inside circle
        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms >= 1.0
        if np.any(mask):
            pts[mask] *= (1.0 - 1e-5) / new_norms[mask, np.newaxis]
        
        # Compute new radius periodically
        if step % 100 == 0:
            current_r = _compute_radius(pts)
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
                # Increase cutoff slightly to allow global rearrangement?
                # Or decrease to focus on local?
                # Usually, as r increases, the "critical" distance increases.
                # We keep cutoff relative to best_r.
                
        # Annealing: Reduce learning rate over time
        if step < 5000:
            lr = 0.02
        elif step < 10000:
            lr = 0.01
        elif step < 20000:
            lr = 0.005
        else:
            lr = 0.001

    # Final Polish with L-BFGS on the best found configuration
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            
            def obj_and_grad(x):
                pts_x = x.reshape(-1, 2)
                # Soft min objective
                norms = np.linalg.norm(pts_x, axis=1)
                boundary = 1.0 - norms
                
                # Pairwise distances
                diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
                d = np.linalg.norm(diff, axis=2)
                iu = np.triu_indices(N, k=1)
                pair_half = d[iu] / 2.0
                
                all_vals = np.concatenate([pair_half, boundary])
                T = 0.001
                neg_vals = -all_vals / T
                max_neg = np.max(neg_vals)
                exp_shifted = np.exp(neg_vals - max_neg)
                log_sum = np.log(np.sum(exp_shifted)) + max_neg
                obj = -T * log_sum
                
                # Gradient
                weights = exp_shifted / np.sum(exp_shifted)
                grad_all = -weights
                
                grad_obj = np.zeros_like(x)
                n_pairs = len(pair_half)
                
                safe_dists = np.where(d > 1e-10, d, 1.0)
                unit_vecs = diff / safe_dists[:, :, np.newaxis]
                
                W = np.zeros((N, N))
                W[iu] = grad_all[:n_pairs]
                W[iu[1], iu[0]] = grad_all[:n_pairs]
                
                grad_pairs = 0.5 * np.einsum('ij,ijc->ic', W, unit_vecs)
                
                safe_norms = np.where(norms > 1e-10, norms, 1.0)
                grad_boundary = -pts_x / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
                
                grad_pts = grad_pairs + grad_boundary
                return obj, grad_pts.ravel()
            
            result = minimize(
                lambda x: obj_and_grad(x)[0],
                best_pts.ravel(),
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_pts = result.x.reshape(-1, 2)
            if np.all(np.linalg.norm(new_pts, axis=1) < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    return best_pts
