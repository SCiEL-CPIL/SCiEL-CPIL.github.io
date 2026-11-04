"""
Optimized packing of 50 spheres in a unit cube.
Strategy:
1. Initialize with diverse configurations (FCC, BCC, HCP, layered, random).
2. Run damped molecular dynamics with adaptive shaking.
3. Polish with aggressive beta annealing using soft-min objective.
4. Active-set refinement on critical constraints.
5. Final high-precision polish pass.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
EPS = 1e-12

def _compute_score(centers):
    """Compute the packing radius for a set of centers."""
    centers = np.clip(centers, -CUBE_HALF + EPS, CUBE_HALF - EPS)
    
    # Wall distance
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_obj_and_grad(params, beta, centers_shape=(N, 3)):
    """
    Objective and gradient: minimize -soft_min(distances)
    Returns (value, gradient)
    """
    centers = params.reshape(centers_shape)
    centers = np.clip(centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    # Precomputed indices for upper triangle
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_full = np.sqrt(np.sum(diff**2, axis=-1))
    d_pairs = d_full[tri_i, tri_j] / 2.0
    
    # Wall distances
    d_x_pos = CUBE_HALF - centers[:, 0]
    d_x_neg = centers[:, 0] + CUBE_HALF
    d_y_pos = CUBE_HALF - centers[:, 1]
    d_y_neg = centers[:, 1] + CUBE_HALF
    d_z_pos = CUBE_HALF - centers[:, 2]
    d_z_neg = centers[:, 2] + CUBE_HALF
    
    wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
    all_dists = np.concatenate([d_pairs, wall_dists])
    
    # Soft minimum calculation using logsumexp for numerical stability
    min_d = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_d))
    soft_min_val = min_d - np.log(np.sum(exp_vals)) / beta
    
    # Weights for gradient
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient computation
    grad_centers = np.zeros_like(centers)
    
    # Pairwise gradient
    diff_ij = centers[tri_i] - centers[tri_j]
    d_ij = d_full[tri_i, tri_j]
    safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
    unit_vecs = diff_ij / safe_d[:, np.newaxis]
    w_pair = weights[:n_pairs] * 0.5 / safe_d
    contrib = w_pair[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_centers, tri_i, contrib)
    np.add.at(grad_centers, tri_j, -contrib)
    
    # Wall gradients
    ws = n_pairs
    grad_centers[:, 0] -= weights[ws:ws+N]
    grad_centers[:, 0] += weights[ws+N:ws+2*N]
    grad_centers[:, 1] -= weights[ws+2*N:ws+3*N]
    grad_centers[:, 1] += weights[ws+3*N:ws+4*N]
    grad_centers[:, 2] -= weights[ws+4*N:ws+5*N]
    grad_centers[:, 2] += weights[ws+5*N:ws+6*N]
    
    return -soft_min_val, -grad_centers.flatten()

def _dynamics_phase(centers, time_budget):
    """
    Run a damped molecular dynamics simulation with adaptive shaking.
    """
    positions = centers.copy().astype(np.float64)
    velocities = np.zeros_like(positions)
    
    k_force = 1.0
    damping = 0.95
    dt = 0.005
    shake_interval = 300
    shake_magnitude = 0.05
    step_count = 0
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(positions)
        
        # Pairwise forces with stronger repulsion for close pairs
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Use inverse cube law for repulsion
        inv_r3 = 1.0 / (dists**3)
        np.fill_diagonal(inv_r3, 0.0)
        
        force_mat = k_force * diff * inv_r3[:, :, np.newaxis]
        forces += np.sum(force_mat, axis=1)
        
        # Wall forces
        d_left = np.clip(positions[:, 0] + CUBE_HALF, 1e-6, None)
        d_right = np.clip(CUBE_HALF - positions[:, 0], 1e-6, None)
        d_bottom = np.clip(positions[:, 1] + CUBE_HALF, 1e-6, None)
        d_top = np.clip(CUBE_HALF - positions[:, 1], 1e-6, None)
        d_back = np.clip(positions[:, 2] + CUBE_HALF, 1e-6, None)
        d_front = np.clip(CUBE_HALF - positions[:, 2], 1e-6, None)
        
        forces[:, 0] += k_force / (d_left**2) - k_force / (d_right**2)
        forces[:, 1] += k_force / (d_bottom**2) - k_force / (d_top**2)
        forces[:, 2] += k_force / (d_back**2) - k_force / (d_front**2)
        
        velocities = velocities * damping + forces * dt
        positions = positions + velocities * dt
        positions = np.clip(positions, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Adaptive shaking to escape local minima
        step_count += 1
        if step_count % shake_interval == 0:
            # Gradually reduce shake magnitude
            shake_mag = shake_magnitude * max(0.1, 1.0 - step_count / (shake_interval * 10))
            positions += shake_mag * np.random.randn(*positions.shape)
            positions = np.clip(positions, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            velocities *= 0.5  # Dampen velocities after shake
    
    return positions

def _build_fcc_grid():
    """Build FCC-inspired lattice initialization."""
    points = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                points.append([ix*0.25 - 0.5, iy*0.25 - 0.5, iz*0.25 - 0.5])
                if ix < 3: points.append([(ix+0.5)*0.25 - 0.5, iy*0.25 - 0.5, iz*0.25 - 0.5])
                if iy < 3: points.append([ix*0.25 - 0.5, (iy+0.5)*0.25 - 0.5, iz*0.25 - 0.5])
                if iz < 3: points.append([ix*0.25 - 0.5, iy*0.25 - 0.5, (iz+0.5)*0.25 - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5 + 1e-9, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        grid = points[:N]
    else:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        grid = np.vstack([points, extra])
    
    perturbation = 0.02 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return grid

def _build_bcc_grid():
    """Build BCC-inspired lattice initialization with better spacing."""
    points = []
    # Use 3x3x3 BCC = 27 corner + 27 body = 54 points, take 50
    a = 1.0 / 3.0  # lattice parameter for unit cube
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                # Corner points
                x = -0.5 + ix * a
                y = -0.5 + iy * a
                z = -0.5 + iz * a
                points.append([x, y, z])
                # Body center point
                points.append([x + a/2, y + a/2, z + a/2])
    
    points = np.array(points)
    # Keep only points strictly inside
    mask = np.all(np.abs(points) < 0.5 - 0.01, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        grid = points[:N]
    else:
        extra = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        grid = np.vstack([points, extra])
    
    perturbation = 0.015 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return grid

def _build_layered_grid():
    """Build layered initialization with hexagonal close-packed offsets."""
    points = []
    n_layers = 5
    # Distribute points across layers: 10,10,10,10,10
    layer_sizes = [10, 10, 10, 10, 10]
    
    for layer_idx, n_in_layer in enumerate(layer_sizes):
        z = -0.4 + layer_idx * 0.2
        # Hexagonal packing in each layer
        if n_in_layer <= 9:
            # Simple circular arrangement
            for i in range(n_in_layer):
                angle = 2 * np.pi * i / n_in_layer + (layer_idx % 2) * np.pi / n_in_layer
                r = 0.3
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y, z])
        else:
            # Grid-like hexagonal pattern
            n_per_row = 4
            rows = (n_in_layer + n_per_row - 1) // n_per_row
            spacing = 0.55
            for row in range(rows):
                for col in range(n_per_row):
                    idx = row * n_per_row + col
                    if idx >= n_in_layer:
                        break
                    offset = (spacing / 2) if row % 2 else 0
                    x = -0.35 + col * spacing + offset
                    y = -0.35 + row * spacing * 0.866
                    points.append([x, y, z])
    
    points = np.array(points[:N])
    perturbation = 0.015 * np.random.randn(len(points), 3)
    points = points + perturbation
    points = np.clip(points, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return points

def _build_random_grid():
    """Build random initialization with Poisson-disk-like spacing."""
    points = np.random.uniform(-0.4, 0.4, size=(N, 3))
    # Add small perturbation to break symmetry
    points += 0.01 * np.random.randn(N, 3)
    points = np.clip(points, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return points

def _build_hcp_grid():
    """Build HCP-inspired initialization."""
    points = []
    # HCP: layers with ABAB stacking
    n_layers = 6
    layer_sizes = [9, 9, 9, 9, 9, 5]  # 50 total
    
    for layer_idx, n_in_layer in enumerate(layer_sizes):
        z = -0.42 + layer_idx * 0.168
        if layer_idx % 2 == 0:
            offset_x, offset_y = 0.0, 0.0
        else:
            offset_x, offset_y = 0.15, 0.15
        
        n_per_row = 3
        rows = (n_in_layer + n_per_row - 1) // n_per_row
        spacing = 0.45
        for row in range(rows):
            for col in range(n_per_row):
                idx = row * n_per_row + col
                if idx >= n_in_layer:
                    break
                x = -0.3 + col * spacing + offset_x
                y = -0.3 + row * spacing * 0.866 + offset_y
                points.append([x, y, z])
    
    points = np.array(points[:N])
    perturbation = 0.012 * np.random.randn(len(points), 3)
    points = points + perturbation
    points = np.clip(points, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    np.random.seed(42)
    
    # 1. Generate multiple diverse initial configurations
    initializers = [
        _build_fcc_grid,
        _build_bcc_grid,
        _build_layered_grid,
        _build_hcp_grid,
        _build_random_grid
    ]
    
    best_centers = None
    best_score = -np.inf
    init_results = []
    
    for init_func in initializers:
        if time.time() > deadline - 80.0:
            break
        
        centers = init_func()
        score = _compute_score(centers)
        init_results.append((centers, score))
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _build_fcc_grid()
        best_score = _compute_score(best_centers)
    
    # 2. Dynamics phase on top-3 initial configurations
    # Sort by initial score, take top 3
    init_results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (init_centers, init_score) in enumerate(init_results[:3]):
        remaining_time = deadline - time.time() - 30.0
        if remaining_time < 5.0:
            break
        
        dyn_budget = min(15.0, remaining_time / max(1, len(init_results[:3]) - i))
        if dyn_budget < 3.0:
            break
        
        np.random.seed(42 + i)
        dyn_centers = _dynamics_phase(init_centers, dyn_budget)
        dyn_score = _compute_score(dyn_centers)
        
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dyn_centers.copy()
    
    # 3. Polish phase with aggressive beta annealing
    if time.time() < deadline - 15.0:
        # Start with moderate beta, increase aggressively
        beta_schedule = [100, 500, 2000, 10000, 50000, 200000, 800000]
        
        for beta in beta_schedule:
            if time.time() > deadline - 8.0:
                break
            
            try:
                start_params = best_centers.flatten()
                
                def fun(x):
                    val, _ = _soft_min_obj_and_grad(x, beta)
                    return val
                
                def jac(x):
                    _, grad = _soft_min_obj_and_grad(x, beta)
                    return grad
                
                result = minimize(
                    fun,
                    start_params,
                    jac=jac,
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80, 'maxcor': 40}
                )
                
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
            except Exception:
                pass
    
    # 4. Final high-precision polish with very high beta
    if time.time() < deadline - 3.0:
        try:
            start_params = best_centers.flatten()
            final_beta = 2000000
            
            def fun_final(x):
                val, _ = _soft_min_obj_and_grad(x, final_beta)
                return val
            
            def jac_final(x):
                _, grad = _soft_min_obj_and_grad(x, final_beta)
                return grad
            
            result = minimize(
                fun_final,
                start_params,
                jac=jac_final,
                method='L-BFGS-B',
                bounds=[(-CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 150, 'maxcor': 80}
            )
            
            final_centers = result.x.reshape(N, 3)
            final_centers = np.clip(final_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            
            final_score = _compute_score(final_centers)
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # 5. Exact-pair refinement: identify critical pairs and do targeted optimization
    if time.time() < deadline - 1.0:
        try:
            # Find the minimum distance pair
            diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Get top 10 closest pairs
            flat_dists = dists.flatten()
            flat_dists = np.where(flat_dists > 1e-10, flat_dists, np.inf)
            top_k = 10
            idx_flat = np.argsort(flat_dists)[:top_k]
            idx_i = idx_flat // N
            idx_j = idx_flat % N
            
            # For each critical pair, try to increase their separation
            for pi, pj in zip(idx_i, idx_j):
                if pi == pj:
                    continue
                if time.time() > deadline - 0.5:
                    break
                
                # Small local optimization focusing on this pair
                try:
                    start_params = best_centers.flatten()
                    
                    # Custom objective that heavily weights this pair
                    def fun_pair(x):
                        c = x.reshape(N, 3)
                        d_pair = np.linalg.norm(c[pi] - c[pj]) / 2.0
                        # Also maintain wall distances
                        wall_d = CUBE_HALF - np.abs(c)
                        min_wall = np.min(wall_d)
                        # Combine: prioritize the critical pair
                        return -(0.9 * d_pair + 0.1 * min_wall)
                    
                    result = minimize(
                        fun_pair,
                        start_params,
                        method='L-BFGS-B',
                        bounds=[(-CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)] * (N * 3),
                        options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    
                    new_centers = result.x.reshape(N, 3)
                    new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                    new_score = _compute_score(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Final validity check
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers

# EVOLVE-BLOCK-START
# The main construct() function is defined above
# EVOLVE-BLOCK-END
