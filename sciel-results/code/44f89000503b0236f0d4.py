"""
Pack 200 equal circles in a unit square using a hybrid approach:
1. Force-directed relaxation with repulsive potentials to escape local minima
2. Projected gradient descent with momentum on the exact radius objective
3. Nelder-Mead simplex for final local refinement without gradients
"""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 140  # Leave 10s safety margin before 150s kill

def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0
    
    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _force_directed_phase(pts, time_limit, k_rep=100.0, k_wall=50.0, dt=0.001):
    """
    Force-directed relaxation: repel overlapping circles and push away from walls.
    Uses a soft repulsive potential to smoothly move circles apart.
    """
    t_start = time.time()
    pts = pts.copy()
    prev_energy = np.inf
    
    while time.time() - t_start < time_limit:
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        D = cdist(pts, pts)
        np.fill_diagonal(D, 0.0)
        
        # Repulsive force: F = k * (1/d^2) * direction, only when d < threshold
        threshold = 0.1  # Only repel when close
        mask = (D > 1e-10) & (D < threshold)
        
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = D[i, j]
                    f_mag = k_rep / (d * d)
                    direction = (pts[i] - pts[j]) / d
                    forces[i] += f_mag * direction
                    forces[j] -= f_mag * direction
        
        # Wall repulsion
        margin = 0.05
        # Left wall
        close_left = pts[:, 0] < -0.5 + margin
        forces[close_left, 0] += k_wall * (-0.5 + margin - pts[close_left, 0])
        # Right wall
        close_right = pts[:, 0] > 0.5 - margin
        forces[close_right, 0] -= k_wall * (pts[close_right, 0] - (0.5 - margin))
        # Bottom wall
        close_bottom = pts[:, 1] < -0.5 + margin
        forces[close_bottom, 1] += k_wall * (-0.5 + margin - pts[close_bottom, 1])
        # Top wall
        close_top = pts[:, 1] > 0.5 - margin
        forces[close_top, 1] -= k_wall * (pts[close_top, 1] - (0.5 - margin))
        
        # Update positions
        pts += dt * forces
        
        # Project back into the square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Check convergence
        current_radius = _get_radius(pts)
        if abs(current_radius - prev_energy) < 1e-10:
            break
        prev_energy = current_radius
        
        # Decay dt slightly for convergence
        dt *= 0.999
    
    return pts

def _projected_gradient_phase(pts, time_limit, lr=0.001, momentum=0.9):
    """
    Projected gradient descent with momentum to maximize the radius.
    The objective is the minimum of all wall distances and half pairwise distances.
    We use a smoothed approximation via the soft-min.
    """
    t_start = time.time()
    pts = pts.copy()
    vel = np.zeros_like(pts)
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    beta = 50.0
    
    while time.time() - t_start < time_limit:
        # Compute soft-min objective and its gradient
        # Wall terms
        wall_vals = np.concatenate([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1]
        ])
        
        # Pair terms
        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        dists = np.sqrt(dx**2 + dy**2)
        pair_vals = dists / 2.0
        
        # Combine all constraint values
        C = np.concatenate([wall_vals, pair_vals])
        
        # Soft min
        y = -beta * C
        max_y = np.max(y)
        shifted = y - max_y
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0 or not np.isfinite(sum_exp):
            break
            
        weights = exp_shifted / sum_exp
        soft_min_val = -(max_y + np.log(sum_exp)) / beta
        
        # Gradient of soft min w.r.t. pts
        grad_pts = np.zeros_like(pts)
        
        # Wall gradients
        w_wall = weights[:4*N]
        grad_pts[:, 0] += w_wall[0:N] - w_wall[N:2*N]
        grad_pts[:, 1] += w_wall[2*N:3*N] - w_wall[3*N:4*N]
        
        # Pair gradients
        w_pair = weights[4*N:]
        valid = dists > 1e-10
        dx_safe = np.where(valid, dx, 1.0)
        dy_safe = np.where(valid, dy, 1.0)
        dists_safe = np.where(valid, dists, 1.0)
        
        grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
        grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
        grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
        grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
        
        np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * w_pair)
        np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * w_pair)
        np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * w_pair)
        np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * w_pair)
        
        # We want to maximize soft_min, so gradient ascent
        # But we're minimizing -soft_min, so the gradient for descent is -grad_pts
        # Actually, we want to move in the direction that increases soft_min
        # The gradient of soft_min w.r.t. pts is (1/beta) * sum(w_i * dC_i/dpts)
        # So grad_pts as computed above is sum(w_i * dC_i/dpts)
        # The gradient of soft_min is grad_pts / beta
        
        # For gradient ascent on soft_min: pts += lr * grad_pts / beta
        # But we also need to handle the projection
        
        vel = momentum * vel + lr * grad_pts / beta
        pts += vel
        
        # Project into square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Track best
        current_r = _get_radius(pts)
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
        
        # Increase beta gradually for sharper optimization
        beta *= 1.01
    
    return best_pts

def _nelder_mead_phase(pts, time_limit, max_iter=200):
    """
    Nelder-Mead simplex method for final local refinement.
    Operates on a subset of points or uses a simplified objective.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    # Nelder-Mead in 400D is too slow. Instead, we do coordinate-wise
    # local search: for each point, try small perturbations in x and y.
    
    np.random.seed(42)
    step_size = 0.005
    
    while time.time() - t_start < time_limit:
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small positive and negative steps
                for sign in [1, -1]:
                    pts_new = pts.copy()
                    pts_new[i, coord] += sign * step_size
                    pts_new[i, coord] = np.clip(pts_new[i, coord], -0.5, 0.5)
                    
                    r_new = _get_radius(pts_new)
                    if r_new > best_r + 1e-12:
                        pts = pts_new
                        best_r = r_new
                        best_pts = pts.copy()
                        improved = True
                        break
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_pts

def _generate_hex_initial(jitter=0.0):
    """
    Generate a hexagonal lattice packing with optional jitter.
    """
    # Target spacing for ~200 circles in unit square
    # Area per circle ~ 1/200 = 0.005, so spacing ~ sqrt(0.005/sqrt(3)/2) ~ 0.075
    s = 0.0732
    points = []
    row_h = s * np.sqrt(3) / 2
    
    y = -0.5 + row_h / 2
    row_idx = 0
    while y < 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s / 2 + shift
        while x < 0.5:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
    
    pts = np.array(points)
    
    # Add jitter
    if jitter > 0:
        pts += np.random.uniform(-jitter, jitter, pts.shape)
    
    # Filter points inside the square
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        # Fallback: generate more points
        s2 = 0.070
        points2 = []
        row_h2 = s2 * np.sqrt(3) / 2
        y = -0.6 + row_h2 / 2
        row_idx = 0
        while y < 0.6:
            shift = (s2 / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points2.append([x, y])
                x += s2
            y += row_h2
            row_idx += 1
        pts = np.array(points2)
        mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
               (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
        pts = pts[mask]
        if len(pts) < N:
            return np.random.uniform(-0.4, 0.4, (N, 2))
    
    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Generate diverse initializations
    inits = []
    
    # Init 1: Standard hex
    pts_hex1 = _generate_hex_initial(jitter=0.0)
    if pts_hex1 is not None and len(pts_hex1) == N:
        inits.append(pts_hex1)
    
    # Init 2: Jittered hex
    pts_hex2 = _generate_hex_initial(jitter=0.01)
    if pts_hex2 is not None and len(pts_hex2) == N:
        inits.append(pts_hex2)
    
    # Init 3: More jittered hex
    pts_hex3 = _generate_hex_initial(jitter=0.02)
    if pts_hex3 is not None and len(pts_hex3) == N:
        inits.append(pts_hex3)
    
    # Init 4: Random
    pts_rand = np.random.uniform(-0.45, 0.45, (N, 2))
    inits.append(pts_rand)
    
    num_inits = len(inits)
    if num_inits == 0:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        best_r = _get_radius(best_pts)
        return best_pts
    
    # Allocate time: 50% for force-directed, 35% for gradient, 15% for Nelder-Mead
    time_force = _TIME_LIMIT * 0.50
    time_grad = _TIME_LIMIT * 0.35
    time_nm = _TIME_LIMIT * 0.15
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 10 - elapsed
        if remaining <= 5:
            break
        
        # Allocate time for this init
        time_per_init = min(remaining / max(1, num_inits - i), 25)
        
        # Phase 1: Force-directed relaxation
        t_force = min(time_force / num_inits, time_per_init * 0.5)
        if t_force > 1:
            pts_fd = _force_directed_phase(init_pts, t_force)
        else:
            pts_fd = init_pts
        
        # Phase 2: Projected gradient descent
        t_grad = min(time_grad / num_inits, time_per_init * 0.35)
        elapsed = time.time() - t0
        if time.time() - t0 < _TIME_LIMIT - 15 and t_grad > 1:
            pts_pg = _projected_gradient_phase(pts_fd, t_grad)
        else:
            pts_pg = pts_fd
        
        # Phase 3: Nelder-Mead local search
        t_nm = min(time_nm / num_inits, time_per_init * 0.15)
        elapsed = time.time() - t0
        if time.time() - t0 < _TIME_LIMIT - 10 and t_nm > 0.5:
            pts_nm = _nelder_mead_phase(pts_pg, t_nm)
        else:
            pts_nm = pts_pg
        
        r_final = _get_radius(pts_nm)
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_nm
    
    # Final polish on the best configuration
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > 2:
            # Quick gradient polish
            pts_polished = _projected_gradient_phase(best_pts, remaining * 0.6, lr=0.0005, momentum=0.8)
            r_polished = _get_radius(pts_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_pts = pts_polished
            
            # Quick Nelder-Mead polish
            elapsed = time.time() - t0
            remaining = _TIME_LIMIT - 5 - elapsed
            if remaining > 1:
                pts_nm2 = _nelder_mead_phase(best_pts, remaining * 0.4)
                r_nm2 = _get_radius(pts_nm2)
                if r_nm2 > best_r:
                    best_r = r_nm2
                    best_pts = pts_nm2
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    
    # Final safety check: ensure all points are inside the square
    best_pts[:, 0] = np.clip(best_pts[:, 0], -0.5, 0.5)
    best_pts[:, 1] = np.clip(best_pts[:, 1], -0.5, 0.5)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
