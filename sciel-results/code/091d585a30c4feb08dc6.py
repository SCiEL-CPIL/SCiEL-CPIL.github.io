import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 130.0  # seconds, leaving 20s margin before the 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
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
    return min(wall, min_pair)

def _energy_and_grad(x, R, weight_pair=1.0, weight_wall=1.0):
    """
    Energy: sum of squared violations for sphere packing with radius R.
    x is flat array of shape (N*3,).
    Returns (energy, gradient).
    """
    centers = x.reshape(N, 3)
    energy = 0.0
    grad_centers = np.zeros_like(centers)
    
    # Pairwise overlaps
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)  # avoid div by zero
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx]
    
    target = 2.0 * R
    viol = np.maximum(0.0, target - d_ij)
    
    energy += weight_pair * np.sum(viol ** 2)
    
    mask = viol > 0
    if np.any(mask):
        d_masked = d_ij[mask]
        viol_masked = viol[mask]
        i_masked = i_idx[mask]
        j_masked = j_idx[mask]
        
        vecs = centers[i_masked] - centers[j_masked]
        coeff = -2.0 * weight_pair * viol_masked / d_masked
        
        np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
        np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
    
    # Wall overlaps
    L = 0.5 - R
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0.0, abs_c - L)
    
    energy += weight_wall * np.sum(wall_viol ** 2)
    grad_centers += 2.0 * weight_wall * wall_viol * np.sign(centers)
    
    return energy, grad_centers.ravel()

def _soft_min_score(x, R):
    """
    Smooth approximation of the minimum distance to boundaries.
    Uses a soft-min to make the objective differentiable.
    x is flat array (N*3,).
    Returns (score, gradient).
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx] / 2.0  # radius equivalent
    
    # Combine all distances into one array
    # Wall distances: N*3 values
    # Pair distances: 1225 values
    all_dists = np.concatenate([wall_dists.ravel(), d_ij])
    
    # Soft min: -beta * log(sum(exp(-d/beta)))
    beta = 0.01
    exp_terms = np.exp(-all_dists / beta)
    score = -beta * np.log(np.sum(exp_terms))
    
    # Gradient of soft min w.r.t. each distance
    weights = exp_terms / np.sum(exp_terms)
    
    grad_centers = np.zeros_like(centers)
    
    # Wall gradient
    wall_weights = weights[:N*3].reshape(N, 3)
    grad_centers -= wall_weights * np.sign(centers)  # d(0.5-|c|)/dc = -sign(c)
    
    # Pair gradient
    pair_weights = weights[N*3:]  # 1225 values
    i_masked = i_idx
    j_masked = j_idx
    
    vecs = centers[i_masked] - centers[j_masked]
    d_masked = dist[i_masked, j_masked]
    
    # d(d_ij/2)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    # So grad_i += -pair_weight * (c_i - c_j) / (2 * d_ij)
    # Wait, let me re-derive.
    # score = -beta * log(sum(exp(-d_k/beta)))
    # d(score)/d(d_k) = exp(-d_k/beta) / sum(exp(-d_m/beta)) = w_k
    # d(score)/d(c_i) = sum_k w_k * d(d_k)/d(c_i)
    # For pair k=(i,j): d(d_k)/d(c_i) = (c_i - c_j)/d_ij (for d_ij, not d_ij/2)
    # But d_k = d_ij/2, so d(d_k)/d(c_i) = (c_i - c_j)/(2*d_ij)
    
    coeff = -pair_weights / (2.0 * d_masked)
    np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
    np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
    
    return score, grad_centers.ravel()

def _generate_initial_guess(seed=0):
    """Generate diverse initial guesses."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    strategy = seed % 6
    
    if strategy == 0:
        # Simple cubic grid + jitter
        xs = np.linspace(-0.4, 0.4, 3)
        ys = np.linspace(-0.4, 0.4, 4)
        zs = np.linspace(-0.4, 0.4, 4)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        centers[:48] = grid
        centers[48:] = rng.uniform(-0.3, 0.3, (2, 3))
        
    elif strategy == 1:
        # BCC-like lattice
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    if i < 2 and j < 2 and k < 2:
                        pts.append([x + 0.2, y + 0.2, z + 0.2])
        centers[:35] = np.array(pts)
        centers[35:] = rng.uniform(-0.35, 0.35, (15, 3))
        
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    elif strategy == 3:
        # Helical / Spiral
        t = np.linspace(0, 10 * np.pi, N)
        r = 0.3 * t / (10 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = (t / (10 * np.pi) - 0.5) * 0.8
        centers = np.column_stack((x, y, z))
        centers += rng.normal(0, 0.05, (N, 3))
        
    elif strategy == 4:
        # Two-layer structure
        # Bottom layer: 5x5 grid
        # Top layer: 5x4 grid offset
        bottom = []
        for i in range(5):
            for j in range(5):
                x = -0.4 + i * 0.2
                y = -0.4 + j * 0.2
                bottom.append([x, y, -0.2])
        top = []
        for i in range(5):
            for j in range(4):
                x = -0.3 + i * 0.2
                y = -0.3 + j * 0.2
                top.append([x, y, 0.2])
        centers[:25] = np.array(bottom)
        centers[25:] = np.array(top)
        centers += rng.normal(0, 0.02, (N, 3))
        
    else:
        # FCC-like arrangement
        # Face-centered cubic lattice points
        pts = []
        # Corner and face centers of a 2x2x2 grid of cells
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.35
                    y = (j - 1) * 0.35
                    z = (k - 1) * 0.35
                    pts.append([x, y, z])
                    # Face centers
                    if i < 2:
                        pts.append([x + 0.175, y, z])
                    if j < 2:
                        pts.append([x, y + 0.175, z])
                    if k < 2:
                        pts.append([x, y, z + 0.175])
        # Take first 50
        pts = np.array(pts[:50])
        centers = pts + rng.normal(0, 0.03, (N, 3))
    
    centers = np.clip(centers, -0.45, 0.45)
    return centers

def _try_fit_radius(centers, R, max_iter=200, tol=1e-10):
    """
    Try to fit spheres of radius R starting from centers.
    Returns (success, new_centers, min_energy).
    """
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
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 30, 'iprint': -1}
        )
        min_energy = res.fun
        if min_energy < 1e-4:  # Threshold for "valid"
            return True, res.x.reshape(N, 3), min_energy
        else:
            return False, res.x.reshape(N, 3), min_energy
    except Exception:
        return False, centers, 1.0

def _lloyd_iteration(centers, R, max_iter=50):
    """
    Lloyd's algorithm: iteratively move each center to the Chebyshev center
    of its Voronoi cell (approximated by nearest neighbors).
    This is a heuristic that can improve packings.
    """
    centers = centers.copy()
    for it in range(max_iter):
        new_centers = centers.copy()
        for i in range(N):
            # Find the 8 nearest neighbors
            dists = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
            dists[i] = np.inf
            nearest_idx = np.argsort(dists)[:8]
            
            # Compute the centroid of nearest neighbors
            centroid = np.mean(centers[nearest_idx], axis=0)
            
            # Move toward centroid but constrain to valid region
            L = 0.5 - R
            new_center = (centers[i] + centroid) / 2.0
            new_center = np.clip(new_center, -L, L)
            
            # Check if this move improves the local minimum distance
            old_min_dist = np.inf
            for j in range(N):
                if j == i:
                    continue
                d = np.linalg.norm(centers[i] - centers[j])
                old_min_dist = min(old_min_dist, d)
            old_wall = np.min(0.5 - np.abs(centers[i]))
            old_score = min(old_min_dist / 2.0, old_wall)
            
            new_min_dist = np.inf
            for j in range(N):
                if j == i:
                    continue
                d = np.linalg.norm(new_center - centers[j])
                new_min_dist = min(new_min_dist, d)
            new_wall = np.min(0.5 - np.abs(new_center))
            new_score = min(new_min_dist / 2.0, new_wall)
            
            if new_score >= old_score - 1e-6:
                new_centers[i] = new_center
        
        if np.allclose(new_centers, centers, atol=1e-8):
            break
        centers = new_centers
    
    return centers

def construct():
    """
    Construct optimal sphere packing.
    """
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    num_restarts = 12
    base_radius = 0.08
    radius_step = 0.0005
    
    for restart_idx in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Generate initial guess
        centers = _generate_initial_guess(seed=restart_idx)
        
        # First, fit base radius
        current_R = base_radius
        success, new_centers, _ = _try_fit_radius(centers, current_R, max_iter=100)
        if not success:
            current_R = 0.05
            success, new_centers, _ = _try_fit_radius(centers, current_R, max_iter=100)
            if not success:
                continue
        
        local_best_R = current_R
        local_centers = new_centers
        
        # Apply Lloyd's iteration to improve
        local_centers = _lloyd_iteration(local_centers, local_best_R, max_iter=20)
        local_best_R = _compute_radius(local_centers)
        
        # Now incrementally increase radius
        stall_count = 0
        max_stalls = 8
        
        while time.time() - start_time < DEADLINE - 5:
            if local_best_R > best_radius:
                best_radius = local_best_R
                best_centers = local_centers.copy()
                
            next_R = local_best_R + radius_step
            
            success, new_centers, energy = _try_fit_radius(local_centers, next_R, max_iter=150)
            
            if success:
                local_best_R = next_R
                local_centers = new_centers
                stall_count = 0
                
                # Apply Lloyd's iteration periodically
                if int(local_best_R * 10000) % 10 == 0:
                    local_centers = _lloyd_iteration(local_centers, local_best_R, max_iter=10)
                    local_best_R = _compute_radius(local_centers)
            else:
                # Failed to fit next_R
                # Try relaxation
                relax_R = local_best_R - radius_step * 0.5
                if relax_R > 0.05:
                    success, new_centers, _ = _try_fit_radius(local_centers, relax_R, max_iter=50)
                    if success:
                        local_centers = new_centers
                        # Try next_R again
                        success2, new_centers2, _ = _try_fit_radius(new_centers, next_R, max_iter=100)
                        if success2:
                            local_best_R = next_R
                            local_centers = new_centers2
                            stall_count = 0
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
                    
            if stall_count > max_stalls:
                break
    
    # Final check
    if best_centers is None:
        centers = _generate_initial_guess(seed=99)
        return centers
        
    return best_centers
