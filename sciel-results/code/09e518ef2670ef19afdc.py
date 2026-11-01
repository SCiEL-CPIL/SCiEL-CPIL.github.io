"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, selecting most central points."""
    best_c = None
    best_r = 0
    
    # Try a range of spacings
    for d in np.linspace(0.055, 0.080, 25):
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        points = []
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            pts = np.array(points)
            # Select the most central N points
            dist_to_bnd = np.minimum(
                0.5 - np.abs(pts[:, 0]),
                0.5 - np.abs(pts[:, 1])
            )
            idx = np.argsort(dist_to_bnd)[::-1]
            c = pts[idx[:N]]
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _qp_optimize(points, max_time=30.0, lr_init=0.003, lr_decay=0.9995, lr_min=0.00002, r_step=0.00008):
    """
    QP-style optimization: iteratively increase target radius r and
    project points to satisfy pairwise and boundary constraints.
    """
    start_time = time.time()
    n = len(points)
    r = _compute_radius(points)
    points = points.copy()
    
    best_points = points.copy()
    best_r = r
    
    for it in range(100000):
        if time.time() > start_time + max_time:
            break
        
        lr = max(lr_init * (lr_decay ** it), lr_min)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        
        # --- Pairwise gradient contribution ---
        threshold = 2.0 * r
        mask = dists < threshold
        safe_dists = np.where(dists > 1e-10, dists, 1e-10)
        
        viol = np.where(mask, threshold - dists, 0.0)
        
        weights = np.zeros_like(dists)
        weights[mask] = viol[mask] / safe_dists[mask]
        
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
        
        # --- Boundary gradient contribution ---
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        active_x = bx <= by
        
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        
        grad_x = viol_x * np.sign(points[:, 0])
        grad_y = viol_y * np.sign(points[:, 1])
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        points = points - lr * total_grad
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        current_r = _compute_radius(points)
        
        if current_r >= r:
            r = current_r + r_step * 0.5
            r = min(r, current_r + 1e-6)
        else:
            r = max(current_r, r - r_step)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    return best_points, best_r


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    p = 20
    mask = dists < 0.15
    if np.any(mask):
        soft_min = np.sum(dists[mask]**p)**(-1.0/p)
    else:
        soft_min = min_dist
    
    p_b = 20
    soft_boundary = np.sum(boundary_dists**p_b)**(-1.0/p_b)
    
    r_pair = soft_min / 2.0
    r_bound = soft_boundary
    
    T = temperature
    soft_min_combined = -T * np.logaddexp(-r_pair/T, -r_bound/T)
    
    return -soft_min_combined


def _local_optimize(c0, max_time=30.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    temperatures = [0.01, 0.015, 0.02, 0.03]
    if temperature_schedule is not None:
        temperatures = temperature_schedule
    
    for T in temperatures:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params):
            return _energy(params, temperature=T)
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                c0 = c_new.copy()
        except Exception:
            continue
    
    return best_c, best_r


def _soft_contact_dynamics(c0, max_time=20.0):
    """
    Rigid-body dynamics with soft-contact potential.
    Uses symplectic Euler with adaptive time-stepping.
    """
    start_time = time.time()
    n = len(c0)
    c = c0.copy()
    v = np.zeros_like(c)
    
    # Potential: high-order polynomial penalty for overlap
    # Force: repulsive when dist < 2r, attractive when slightly above
    r_target = _compute_radius(c)
    
    dt = 0.001
    damping = 0.995
    k_repel = 50.0
    k_attract = 5.0
    r_ref = r_target * 2.0  # reference distance for interaction
    
    best_c = c.copy()
    best_r = r_target
    
    for it in range(100000):
        if time.time() > start_time + max_time:
            break
        
        # Compute forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(c)
        
        # Pairwise soft-contact forces
        # Repulsive when dist < r_ref, mild attractive when r_ref < dist < 1.5*r_ref
        mask_repel = dists < r_ref
        mask_attract = (dists >= r_ref) & (dists < 1.5 * r_ref)
        
        safe_dists = np.where(dists > 1e-10, dists, 1e-10)
        
        # Repulsive force: k * (r_ref - dist)^3 / dist^2 * direction
        viol = np.where(mask_repel, r_ref - dists, 0.0)
        repel_mag = k_repel * viol**3 / (safe_dists**2 + 1e-12)
        repel_mag[~mask_repel] = 0.0
        
        # Attractive force: k * (dist - r_ref) / dist^2 * direction (mild)
        attract_mag = k_attract * np.where(mask_attract, dists - r_ref, 0.0) / (safe_dists**2 + 1e-12)
        attract_mag[~mask_attract] = 0.0
        
        # Net force magnitude (repulsive positive, attractive negative)
        net_mag = repel_mag - attract_mag
        
        # Force direction: (c_i - c_j) / dist
        force_dirs = diff / (safe_dists[:, :, np.newaxis] + 1e-12)
        
        # Sum forces
        forces = np.sum(net_mag[:, :, np.newaxis] * force_dirs, axis=1)
        
        # Boundary forces: push away from walls
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        
        wall_force_x = np.where(bx < r_target, k_repel * (r_target - bx)**2 * np.sign(c[:, 0]), 0.0)
        wall_force_y = np.where(by < r_target, k_repel * (r_target - by)**2 * np.sign(c[:, 1]), 0.0)
        
        forces[:, 0] += wall_force_x
        forces[:, 1] += wall_force_y
        
        # Symplectic Euler update
        v = v * damping + forces * dt
        c = c + v * dt
        
        # Project into square
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        
        # Check if settled
        current_r = _compute_radius(c)
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
        
        # Adaptive time step: reduce if velocity is high
        max_v = np.max(np.abs(v))
        if max_v > 0.05:
            dt *= 0.9
        elif max_v < 0.001:
            dt = min(dt * 1.05, 0.005)
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization with central selection
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: QP optimization (primary method)
    if time.time() < deadline - 50:
        max_qp_time = min(40.0, deadline - time.time() - 15)
        c_qp, r_qp = _qp_optimize(c_hex, max_time=max_qp_time)
        if r_qp > best_radius:
            best_radius = r_qp
            best_c = c_qp.copy()
    
    # Step 3: L-BFGS local optimization
    if time.time() < deadline - 30:
        max_lbfgs_time = min(25.0, deadline - time.time() - 10)
        c_lbfgs, r_lbfgs = _local_optimize(best_c, max_time=max_lbfgs_time)
        if r_lbfgs > best_radius:
            best_radius = r_lbfgs
            best_c = c_lbfgs.copy()
    
    # Step 4: Soft-contact dynamics refinement
    if time.time() < deadline - 15:
        max_dyn_time = min(15.0, deadline - time.time() - 5)
        c_dyn, r_dyn = _soft_contact_dynamics(best_c, max_time=max_dyn_time)
        if r_dyn > best_radius:
            best_radius = r_dyn
            best_c = c_dyn.copy()
    
    # Step 5: Final QP polish
    if time.time() < deadline - 8:
        max_qp2_time = min(10.0, deadline - time.time() - 3)
        c_qp2, r_qp2 = _qp_optimize(best_c, max_time=max_qp2_time, lr_init=0.001, lr_decay=0.9998, lr_min=0.00001, r_step=0.00003)
        if r_qp2 > best_radius:
            best_radius = r_qp2
            best_c = c_qp2.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
