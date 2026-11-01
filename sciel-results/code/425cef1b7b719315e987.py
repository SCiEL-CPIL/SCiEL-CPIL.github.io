"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def _calc_radius(pts):
    """Compute the maximum valid radius for given point positions."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_dense_hex(n):
    """
    Generate a dense hexagonal lattice and select n points optimally.
    Uses multiple spacing values and selects the best subset.
    """
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings
    spacings = np.linspace(0.065, 0.082, 25)
    
    for d in spacings:
        row_h = d * np.sqrt(3.0) / 2.0
        col_w = d
        
        # Generate all lattice points in [-0.5-d, 0.5+d]^2
        y_vals = np.arange(-0.5 - row_h, 0.5 + row_h + row_h, row_h)
        all_pts = []
        
        for j, y in enumerate(y_vals):
            if y < -0.5 - d or y > 0.5 + d:
                continue
            offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
            x_start = -0.5 - col_w + offset
            x_vals = np.arange(x_start, 0.5 + col_w + offset + col_w, col_w)
            for x in x_vals:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    all_pts.append([x, y])
        
        if len(all_pts) < n:
            continue
        
        all_pts = np.array(all_pts)
        
        # Score each candidate point by how "central" it is
        dist_to_bnd = np.minimum(
            0.5 - np.abs(all_pts[:, 0]),
            0.5 - np.abs(all_pts[:, 1])
        )
        
        # Sort by distance to boundary (descending) - take most central points
        idx = np.argsort(dist_to_bnd)[::-1]
        selected = all_pts[idx[:n]]
        
        score = _calc_radius(selected)
        if score > best_score:
            best_score = score
            best_points = selected.copy()
    
    if best_points is None:
        # Fallback: regular grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    return best_points

def _qp_optimize(points, max_iter=5000, lr_init=0.008, lr_min=0.00005, beta=0.999):
    """
    Optimize point positions to maximize minimum distance using
    projected gradient descent with adaptive radius increase.
    """
    n = len(points)
    r = _calc_radius(points)
    points = points.copy()
    
    for it in range(max_iter):
        lr = max(lr_init * (beta ** it), lr_min)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # --- Pairwise gradient contribution ---
        mask = dists < 2.0 * r
        safe_dists = np.where(dists > 1e-10, dists, 1e-10)
        
        # Violation amount
        viol = np.where(mask, 2.0 * r - dists, 0.0)
        
        # Weight for each pair: proportional to violation squared
        weights = np.zeros_like(dists)
        weights[mask] = (viol[mask] ** 2) / safe_dists[mask]
        
        # Gradient from pairwise constraints
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
        
        # --- Boundary gradient contribution ---
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Active boundary constraint per point
        active_x = bx <= by
        
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        
        # Gradient: push away from boundary
        grad_x = 2.0 * viol_x * np.sign(points[:, 0])
        grad_y = 2.0 * viol_y * np.sign(points[:, 1])
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        # Update positions
        points = points - lr * total_grad
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check current radius
        current_r = _calc_radius(points)
        
        # Adaptive radius update
        if current_r >= r:
            r = min(current_r + 0.0001, current_r + 1e-6)
        else:
            r = max(current_r, r - 0.00015)
    
    return points

def _local_search(points, n_iter=2000, step_init=0.01, step_decay=0.9995):
    """
    Local search: try small random perturbations and keep improvements.
    """
    points = points.copy()
    current_r = _calc_radius(points)
    step = step_init
    
    for it in range(n_iter):
        step *= step_decay
        if step < 1e-6:
            break
        
        # Try a few random perturbations
        best_new_points = None
        best_new_r = current_r
        
        n_trials = 5
        for _ in range(n_trials):
            trial = points + np.random.randn(len(points), 2) * step
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            
            trial_r = _calc_radius(trial)
            if trial_r > best_new_r + 1e-10:
                best_new_r = trial_r
                best_new_points = trial.copy()
        
        if best_new_points is not None:
            points = best_new_points
            current_r = best_new_r
    
    return points

def construct():
    # Step 1: Generate initial hexagonal lattice
    points = _generate_dense_hex(N)
    
    # Step 2: Optimize using QP-style projected gradient descent
    points = _qp_optimize(points, max_iter=5000)
    
    # Step 3: Local search for fine-tuning
    points = _local_search(points, n_iter=2000, step_init=0.008, step_decay=0.9995)
    
    # Step 4: Final polish with smaller gradient steps
    points = _qp_optimize(points, max_iter=2000, lr_init=0.003, lr_min=0.00002)
    
    return points
# EVOLVE-BLOCK-END
