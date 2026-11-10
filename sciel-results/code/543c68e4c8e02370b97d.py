import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 135.0  # seconds, leaving 15s margin before the 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint: min_i min_k (0.5 - |c_ik|)
    wall = np.min(0.5 - np.abs(centers))
    if wall <= 0:
        return 0.0
    # Pairwise constraint: min_{i<j} |c_i - c_j| / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    min_pair = np.min(dist[i_idx, j_idx]) / 2.0
    return float(min(wall, min_pair))

def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective: minimize -log(sum(exp(-beta * d_ij))) approximately.
    We use the sum of inverse squared distances as a proxy that's easier to handle.
    Actually, let's use: sum_{i<j} 1/d_ij^2 + wall_penalty
    This pushes all distances up.
    
    Alternatively, use the log-sum-exp trick for min distance:
    f(x) = -1/beta * log(sum_{i<j} exp(-beta * d_ij))
    As beta -> inf, this approaches -min(d_ij).
    We minimize this (i.e., maximize min distance).
    
    But for numerical stability, let's use:
    f(x) = sum_{i<j} exp(-beta * d_ij) + wall_term
    Minimizing this pushes distances up.
    
    Actually, the cleanest differentiable objective for maximizing min distance is:
    f(x) = sum_{i<j} (1/d_ij)^2 + sum_i wall_penalty_i
    This is smooth and has nice gradients.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise: sum 1/d^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Set diagonal to inf to avoid division by zero
    np.fill_diagonal(dist_sq, np.inf)
    
    # We want to minimize sum of 1/d^2 for i<j
    # But this can be very large if any distance is small.
    # Let's use a softer version: sum 1/(d^2 + eps)
    eps = 1e-10
    inv_d2 = 1.0 / (dist_sq + eps)
    # Only i<j
    i_idx, j_idx = np.triu_indices(N, k=1)
    obj = np.sum(inv_d2[i_idx, j_idx])
    
    # Wall penalty: for each coordinate, if |c| > 0.5 - margin, penalize
    # We want centers to stay within [-0.5+R, 0.5-R] but since R is unknown,
    # let's use a soft wall at 0.5.
    # Penalty: sum over all coords of max(0, |c| - 0.45)^2
    # This keeps points away from walls.
    margin = 0.45
    wall_viol = np.maximum(0.0, np.abs(centers) - margin)
    wall_obj = np.sum(wall_viol ** 2)
    
    total_obj = obj + 100.0 * wall_obj
    
    # Gradient
    grad_centers = np.zeros_like(centers)
    
    # d/d(c_i) [1/(d_ij^2 + eps)] = -2 * (c_i - c_j) / (d_ij^2 + eps)^2
    # where d_ij^2 = |c_i - c_j|^2
    # grad_i += sum_j -2 * (c_i - c_j) / (d_ij^2 + eps)^2
    # grad_j += sum_i  2 * (c_i - c_j) / (d_ij^2 + eps)^2
    
    # Vectorized:
    # For each pair (i,j), contribution to grad_i is -2 * diff_ij / (d_ij^2 + eps)^2
    # where diff_ij = c_i - c_j
    
    inv_d2_sq = 1.0 / (dist_sq + eps) ** 2
    # Multiply by diff
    grad_term = -2.0 * diff * inv_d2_sq[:, :, np.newaxis]
    
    # Sum over j for each i (excluding self)
    grad_centers = np.sum(grad_term, axis=1)
    
    # Wall gradient
    grad_centers += 200.0 * wall_viol * np.sign(centers)
    
    grad = grad_centers.ravel()
    
    return total_obj, grad

def _energy_and_grad(x, R):
    """
    Energy for fixed radius R: sum of squared overlaps.
    """
    centers = x.reshape(N, 3)
    energy = 0.0
    grad_centers = np.zeros_like(centers)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx]
    
    target = 2.0 * R
    viol = np.maximum(0.0, target - d_ij)
    energy += np.sum(viol ** 2)
    
    mask = viol > 0
    if np.any(mask):
        d_masked = d_ij[mask]
        viol_masked = viol[mask]
        i_masked = i_idx[mask]
        j_masked = j_idx[mask]
        vecs = centers[i_masked] - centers[j_masked]
        coeff = -2.0 * viol_masked / d_masked
        np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
        np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
    
    # Wall
    L = 0.5 - R
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0.0, abs_c - L)
    energy += np.sum(wall_viol ** 2)
    grad_centers += 2.0 * wall_viol * np.sign(centers)
    
    grad = grad_centers.ravel()
    return energy, grad

def _generate_initial_guess(seed=0):
    """Generate diverse initial guesses."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    
    strategy = seed % 5
    
    if strategy == 0:
        # Cubic grid
        n_per_side = 4  # 4^3 = 64, take 50
        xs = np.linspace(-0.4, 0.4, n_per_side)
        ys = np.linspace(-0.4, 0.4, n_per_side)
        zs = np.linspace(-0.4, 0.4, n_per_side)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        idx = rng.choice(len(grid), N, replace=False)
        centers = grid[idx]
        
    elif strategy == 1:
        # BCC-like
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.45
                    y = (j - 1) * 0.45
                    z = (k - 1) * 0.45
                    pts.append([x, y, z])
                    if i < 2 and j < 2 and k < 2:
                        pts.append([x + 0.225, y + 0.225, z + 0.225])
        pts = np.array(pts)
        if len(pts) >= N:
            idx = rng.choice(len(pts), N, replace=False)
            centers = pts[idx]
        else:
            centers[:len(pts)] = pts
            centers[len(pts):] = rng.uniform(-0.4, 0.4, (N - len(pts), 3))
            
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    elif strategy == 3:
        # FCC-like
        pts = []
        # FCC: corners + face centers
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    # Face centers
                    if i < 2:
                        pts.append([x + 0.2, y, z])
                    if j < 2:
                        pts.append([x, y + 0.2, z])
                    if k < 2:
                        pts.append([x, y, z + 0.2])
        pts = np.array(pts)
        # Remove duplicates by rounding
        pts = np.round(pts, 6)
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) >= N:
            idx = rng.choice(len(unique_pts), N, replace=False)
            centers = unique_pts[idx]
        else:
            centers[:len(unique_pts)] = unique_pts
            centers[len(unique_pts):] = rng.uniform(-0.4, 0.4, (N - len(unique_pts), 3))
            
    else:
        # Spiral
        t = np.linspace(0, 8 * np.pi, N)
        r = 0.35 * t / (8 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = (t / (8 * np.pi) - 0.5) * 0.8
        centers = np.column_stack((x, y, z))
        centers += rng.normal(0, 0.03, (N, 3))

    centers = np.clip(centers, -0.45, 0.45)
    return centers

def _optimize_soft_min(centers, beta=50.0, max_iter=200):
    """Optimize using soft-min objective."""
    x0 = centers.ravel()
    bounds = [(-0.45, 0.45)] * (N * 3)
    
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers

def _try_fit_radius(centers, R, max_iter=150):
    """Try to fit spheres of radius R."""
    x0 = centers.ravel()
    bounds = [(-0.5 + R, 0.5 - R)] * (N * 3)
    
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            args=(R,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
        )
        min_energy = res.fun
        if min_energy < 1e-6:
            return True, res.x.reshape(N, 3)
        else:
            return False, res.x.reshape(N, 3)
    except Exception:
        return False, centers

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    def update_best(centers):
        nonlocal best_radius, best_centers
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Phase 1: Soft-min optimization from multiple restarts
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        centers = _generate_initial_guess(seed=restart)
        
        # Soft-min optimization
        optimized = _optimize_soft_min(centers, beta=50.0, max_iter=100)
        update_best(optimized)
        
        # Second pass with higher beta
        if time.time() - start_time < DEADLINE - 15:
            optimized2 = _optimize_soft_min(optimized, beta=100.0, max_iter=100)
            update_best(optimized2)
    
    # Phase 2: Binary search on radius with energy minimization
    if time.time() - start_time < DEADLINE - 10:
        # Start from the best configuration found
        if best_centers is not None:
            # Try to push the radius higher
            current_R = best_radius
            step = 0.0005
            stall_count = 0
            max_stalls = 8
            
            while time.time() - start_time < DEADLINE - 5:
                next_R = current_R + step
                
                success, new_centers = _try_fit_radius(best_centers, next_R, max_iter=100)
                
                if success:
                    current_R = next_R
                    best_centers = new_centers
                    update_best(best_centers)
                    stall_count = 0
                else:
                    # Try relaxing
                    relax_R = max(0.05, current_R - step * 0.5)
                    success2, relaxed = _try_fit_radius(best_centers, relax_R, max_iter=50)
                    if success2:
                        # Try next_R again from relaxed state
                        success3, new_centers2 = _try_fit_radius(relaxed, next_R, max_iter=50)
                        if success3:
                            current_R = next_R
                            best_centers = new_centers2
                            update_best(best_centers)
                            stall_count = 0
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
                        
                    if stall_count > max_stalls:
                        break
    
    # Phase 3: Final local refinement
    if best_centers is not None and time.time() - start_time < DEADLINE - 3:
        # One more soft-min pass
        final = _optimize_soft_min(best_centers, beta=200.0, max_iter=50)
        update_best(final)
        
        # Try to fit at the current best radius
        if best_radius > 0.1:
            success, refined = _try_fit_radius(best_centers, best_radius, max_iter=50)
            if success:
                update_best(refined)
    
    if best_centers is None:
        # Fallback
        best_centers = _generate_initial_guess(seed=99)
    
    return best_centers
