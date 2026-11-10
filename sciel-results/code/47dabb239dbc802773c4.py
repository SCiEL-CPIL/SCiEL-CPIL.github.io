"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance from each center to nearest wall
    # Walls are at -0.5 and 0.5
    wall_dists = np.min(np.array([
        centers[:, 0] + 0.5,
        -centers[:, 0] + 0.5,
        centers[:, 1] + 0.5,
        -centers[:, 1] + 0.5,
        centers[:, 2] + 0.5,
        -centers[:, 2] + 0.5,
    ]), axis=1)
    
    if np.any(wall_dists < -1e-10):
        return 0.0
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = 0.5 * min(min_pair, min_wall)
    return r if r > 0 else 0.0

def _softmin_grad(vals, beta):
    """
    Compute soft-min of vals and its gradient.
    softmin(vals, beta) = -1/beta * log(sum(exp(-beta * vals)))
    grad_i = exp(-beta * vals_i) / sum(exp(-beta * vals))
    """
    # Shift for numerical stability
    vals_shifted = vals - np.min(vals)
    exp_vals = np.exp(-beta * vals_shifted)
    sum_exp = np.sum(exp_vals)
    # Soft-min value
    soft_min_val = -1.0 / beta * np.log(sum_exp) + np.min(vals)
    # Gradient w.r.t vals
    grad = exp_vals / sum_exp
    return soft_min_val, grad

def _lj_force_directed_step(centers, step_size, target_dist, temperature):
    """
    Perform a Lennard-Jones-style force-directed step with simulated annealing.
    Uses LJ repulsion for close pairs and gentle attraction for far pairs,
    plus wall confinement forces.
    """
    centers = centers.copy()
    forces = np.zeros_like(centers)
    
    # Lennard-Jones parameters
    sigma = target_dist  # equilibrium distance
    epsilon = 1.0
    
    # Sphere-sphere LJ forces
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                diff = np.random.normal(0, 1e-6, 3)
                d = np.linalg.norm(diff)
            
            # LJ force: F = 24*epsilon*(2*sigma^6/d^7 - sigma^12/d^11) * r_hat
            # Simplified: only use repulsive part for d < sigma, gentle attraction for sigma < d < 2*sigma
            if d < 2.0 * sigma:
                sig6 = (sigma / d) ** 6
                sig12 = sig6 * sig6
                # Force magnitude (repulsive when d < sigma, attractive when d > sigma)
                mag = 24.0 * epsilon * (2.0 * sig6 - sig12) / d
                force = diff / d * mag
                forces[i] += force
                forces[j] -= force
    
    # Wall confinement: harmonic potential pushing spheres toward center
    # This keeps spheres inside the cube and encourages dense packing
    wall_strength = 50.0
    for axis in range(3):
        # Push away from walls if too close
        dist_left = centers[:, axis] + 0.5
        dist_right = 0.5 - centers[:, axis]
        
        # If within r_target of wall, push inward
        r_margin = target_dist * 0.6
        mask_left = dist_left < r_margin
        mask_right = dist_right < r_margin
        
        if np.any(mask_left):
            forces[mask_left, axis] += (r_margin - dist_left[mask_left]) * wall_strength
        if np.any(mask_right):
            forces[mask_right, axis] -= (r_margin - dist_right[mask_right]) * wall_strength
    
    # Add small random perturbation (simulated annealing)
    if temperature > 0:
        forces += np.random.normal(0, temperature, size=(N, 3))
    
    # Apply forces with adaptive step size
    centers += step_size * forces
    
    # Clip to cube to ensure validity
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers

def _polish_local(centers, max_iter=200, tol=1e-14):
    """
    High-precision local polish using gradient ascent on the minimum separation.
    Uses a very high beta soft-min to approximate the true min.
    """
    centers = centers.copy()
    beta = 5000.0
    
    # Precompute indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for iteration in range(max_iter):
        # Compute current minimum
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        W = np.zeros((N, 6))
        W[:, 0] = centers[:, 0] + 0.5
        W[:, 1] = -centers[:, 0] + 0.5
        W[:, 2] = centers[:, 1] + 0.5
        W[:, 3] = -centers[:, 1] + 0.5
        W[:, 4] = centers[:, 2] + 0.5
        W[:, 5] = -centers[:, 2] + 0.5
        min_wall = np.min(W)
        
        current_min = min(min_pair, min_wall)
        
        # Compute gradient of soft-min
        V_pairs = dists[i_idx, j_idx]
        V_walls_flat = W.flatten()
        V_all = np.concatenate([V_pairs, V_walls_flat])
        
        val, grad_all = _softmin_grad(V_all, beta)
        
        grad_flat = np.zeros((N, 3))
        
        # Pair contributions
        grad_pair = grad_all[:len(V_pairs)]
        diffs = centers[i_idx] - centers[j_idx]
        norms = np.linalg.norm(diffs, axis=1)
        norms[norms == 0] = 1e-10
        unit_vecs = diffs / norms[:, np.newaxis]
        
        np.add.at(grad_flat, i_idx, grad_pair[:, None] * unit_vecs)
        np.add.at(grad_flat, j_idx, -grad_pair[:, None] * unit_vecs)
        
        # Wall contributions
        grad_wall_flat = grad_all[len(V_pairs):]
        grad_wall = grad_wall_flat.reshape(N, 6)
        
        grad_flat[:, 0] += grad_wall[:, 0] - grad_wall[:, 1]
        grad_flat[:, 1] += grad_wall[:, 2] - grad_wall[:, 3]
        grad_flat[:, 2] += grad_wall[:, 4] - grad_wall[:, 5]
        
        # Gradient ascent step with adaptive step size
        grad_norm = np.linalg.norm(grad_flat)
        if grad_norm < tol:
            break
        
        # Adaptive step: larger when gradient is small, smaller when large
        step_size = 1e-4 * current_min / (grad_norm + 1e-10)
        step_size = min(step_size, 1e-3)
        
        new_centers = centers + step_size * grad_flat
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        # Check if improvement
        new_dists = cdist(new_centers, new_centers)
        np.fill_diagonal(new_dists, np.inf)
        new_min_pair = np.min(new_dists)
        
        new_W = np.zeros((N, 6))
        new_W[:, 0] = new_centers[:, 0] + 0.5
        new_W[:, 1] = -new_centers[:, 0] + 0.5
        new_W[:, 2] = new_centers[:, 1] + 0.5
        new_W[:, 3] = -new_centers[:, 1] + 0.5
        new_W[:, 4] = new_centers[:, 2] + 0.5
        new_W[:, 5] = -new_centers[:, 2] + 0.5
        new_min_wall = np.min(new_W)
        
        new_min = min(new_min_pair, new_min_wall)
        
        if new_min > current_min:
            centers = new_centers
        else:
            # Halve step size
            step_size *= 0.5
            new_centers = centers + step_size * grad_flat
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            centers = new_centers
    
    return centers

def construct():
    """Construct centers for N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Initial Grid Placement
    # Try multiple grid configurations
    grid_configs = [
        (4, 5, 5),  # 4*5*5 = 100
        (5, 4, 5),
        (5, 5, 4),
    ]
    
    best_centers = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Optimization Function using Soft-Min
    def objective_and_grad(x_flat, beta):
        centers = x_flat.reshape(N, 3)
        
        # Pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Get unique pairs i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Wall distances
        W = np.zeros((N, 6))
        W[:, 0] = centers[:, 0] + 0.5
        W[:, 1] = -centers[:, 0] + 0.5
        W[:, 2] = centers[:, 1] + 0.5
        W[:, 3] = -centers[:, 1] + 0.5
        W[:, 4] = centers[:, 2] + 0.5
        W[:, 5] = -centers[:, 2] + 0.5
        
        V_pairs = pair_dists
        V_walls_flat = W.flatten()
        V_all = np.concatenate([V_pairs, V_walls_flat])
        
        val, grad_all = _softmin_grad(V_all, beta)
        obj = -val
        
        grad_flat = np.zeros_like(x_flat)
        
        # 1. Pair contributions
        grad_pair = grad_all[:len(V_pairs)]
        diffs = centers[i_idx] - centers[j_idx]
        norms = np.linalg.norm(diffs, axis=1)
        norms[norms == 0] = 1e-8
        unit_vecs = diffs / norms[:, np.newaxis]
        
        np.add.at(grad_flat, i_idx, grad_pair[:, None] * 0.5 * unit_vecs)
        np.add.at(grad_flat, j_idx, -grad_pair[:, None] * 0.5 * unit_vecs)
        
        # 2. Wall contributions
        grad_wall_flat = grad_all[len(V_pairs):]
        grad_wall = grad_wall_flat.reshape(N, 6)
        
        grad_flat[:, 0] += grad_wall[:, 0] - grad_wall[:, 1]
        grad_flat[:, 1] += grad_wall[:, 2] - grad_wall[:, 3]
        grad_flat[:, 2] += grad_wall[:, 4] - grad_wall[:, 5]
        
        return obj, grad_flat
    
    # Phase 1: LJ-based force-directed relaxation from multiple grid starts
    for config in grid_configs:
        if time.time() > deadline - 30.0:
            break
        
        m1, m2, m3 = config
        ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
        
        grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
        
        # Multiple perturbations from this grid
        for pert_iter in range(3):
            if time.time() > deadline - 25.0:
                break
            
            # Perturb the grid to break symmetry
            perturbation_scale = 0.03 + 0.02 * pert_iter
            perturbation = rng.normal(0, perturbation_scale, size=(N, 3))
            current_centers = grid + perturbation
            current_centers = np.clip(current_centers, -0.49, 0.49)
            
            # LJ Force-Directed Pre-processing with simulated annealing
            # Target distance based on expected optimal radius
            target_dist = 0.22  # ~2 * 0.11
            temperature = 0.01
            step_size = 0.005
            
            for fd_iter in range(100):
                if time.time() > deadline - 20.0:
                    break
                current_centers = _lj_force_directed_step(
                    current_centers, step_size, target_dist, temperature
                )
                # Cooling schedule
                temperature *= 0.97
                step_size *= 0.995
            
            # Check radius after force-directed
            fd_radius = _compute_radius(current_centers)
            
            # Phase 2: Gradient-based refinement with soft-min
            beta_schedule = [20, 100, 500, 2000]
            
            for beta in beta_schedule:
                if time.time() > deadline - 15.0:
                    break
                
                x0 = current_centers.flatten()
                bounds = [(-0.4999, 0.4999)] * (N * 3)
                
                try:
                    res = minimize(
                        lambda x: objective_and_grad(x, beta)[0],
                        x0,
                        jac=lambda x: objective_and_grad(x, beta)[1],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14}
                    )
                    
                    new_centers = res.x.reshape(N, 3)
                    new_radius = _compute_radius(new_centers)
                    
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
                        current_centers = new_centers.copy()
                        
                except Exception:
                    pass
            
            # Phase 3: High-precision local polish
            if time.time() > deadline - 5.0:
                break
            
            polished = _polish_local(current_centers, max_iter=300, tol=1e-15)
            polished_radius = _compute_radius(polished)
            
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished.copy()
    
    # Final polish on the best solution found
    if best_centers is not None and time.time() < deadline - 2.0:
        final_polished = _polish_local(best_centers, max_iter=500, tol=1e-15)
        final_radius = _compute_radius(final_polished)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_polished
    
    # Ensure validity
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        # Fallback to simple grid
        m1, m2, m3 = 4, 5, 5
        ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
        best_centers = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    
    return best_centers
# EVOLVE-BLOCK-END
