"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _jellyfish_to_centers(params):
    """
    Convert jellyfish parameters to sphere centers.
    params layout:
      [0:24]   - 8 anchor points (3D each)
      [24:300] - 100 lattice offsets (3D each)
    
    The base lattice is a 5x4x5 grid. Each sphere's position is:
      pos_i = anchor_blend_i + offset_i
    where anchor_blend_i is a weighted blend of the 8 anchors based on
    the sphere's position in the lattice (bilinear/trilinear interpolation).
    """
    anchors = params[0:24].reshape(8, 3)
    offsets = params[24:24 + N*3].reshape(N, 3)
    
    # Define the 5x4x5 lattice base positions
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    # Create lattice points and compute normalized coordinates [0,1] for each axis
    lattice_pts = []
    for z in zs:
        for y in ys:
            for x in xs:
                lattice_pts.append((x, y, z))
    lattice_pts = np.array(lattice_pts[:N])  # Take first N points
    
    # Normalized coordinates in [0, 1] for trilinear interpolation
    # x_norm: 0 at -0.4, 1 at 0.4
    x_norm = (lattice_pts[:, 0] - (-0.4)) / 0.8
    y_norm = (lattice_pts[:, 1] - (-0.4)) / 0.8
    z_norm = (lattice_pts[:, 2] - (-0.4)) / 0.8
    
    # Trilinear interpolation weights for 8 corners
    # Corner indices: (x,y,z) where x,y,z in {0,1}
    # 0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
    # 4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
    w000 = (1-x_norm)*(1-y_norm)*(1-z_norm)
    w100 = x_norm*(1-y_norm)*(1-z_norm)
    w010 = (1-x_norm)*y_norm*(1-z_norm)
    w110 = x_norm*y_norm*(1-z_norm)
    w001 = (1-x_norm)*(1-y_norm)*z_norm
    w101 = x_norm*(1-y_norm)*z_norm
    w011 = (1-x_norm)*y_norm*z_norm
    w111 = x_norm*y_norm*z_norm
    
    weights = np.stack([w000, w100, w010, w110, w001, w101, w011, w111], axis=1)  # (N, 8)
    
    # Blend anchors: (N, 8) @ (8, 3) -> (N, 3)
    anchor_blend = weights @ anchors
    
    # Final positions
    centers = anchor_blend + offsets
    return centers

def _centers_to_jellyfish(centers):
    """
    Inverse: given centers, estimate jellyfish parameters.
    This is an approximation - we solve for anchors and offsets.
    """
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    lattice_pts = []
    for z in zs:
        for y in ys:
            for x in xs:
                lattice_pts.append((x, y, z))
    lattice_pts = np.array(lattice_pts[:N])
    
    x_norm = (lattice_pts[:, 0] - (-0.4)) / 0.8
    y_norm = (lattice_pts[:, 1] - (-0.4)) / 0.8
    z_norm = (lattice_pts[:, 2] - (-0.4)) / 0.8
    
    w000 = (1-x_norm)*(1-y_norm)*(1-z_norm)
    w100 = x_norm*(1-y_norm)*(1-z_norm)
    w010 = (1-x_norm)*y_norm*(1-z_norm)
    w110 = x_norm*y_norm*(1-z_norm)
    w001 = (1-x_norm)*(1-y_norm)*z_norm
    w101 = x_norm*(1-y_norm)*z_norm
    w011 = (1-x_norm)*y_norm*z_norm
    w111 = x_norm*y_norm*z_norm
    
    weights = np.stack([w000, w100, w010, w110, w001, w101, w011, w111], axis=1)
    
    # Simple approach: set offsets to zero, solve for anchors via least squares
    # centers ≈ weights @ anchors
    # anchors = (W^T W)^{-1} W^T centers
    W = weights  # (N, 8)
    C = centers  # (N, 3)
    
    try:
        anchors = np.linalg.lstsq(W, C, rcond=None)[0]  # (8, 3)
    except:
        anchors = np.zeros((8, 3))
    
    # Offsets are the residuals
    anchor_blend = W @ anchors
    offsets = C - anchor_blend
    
    params = np.concatenate([anchors.flatten(), offsets.flatten()])
    return params

def _energy_and_gradient_jellyfish(params, R):
    """
    Energy and gradient in jellyfish parameter space.
    """
    centers = _jellyfish_to_centers(params)
    n = len(centers)
    
    energy = 0.0
    grad_centers = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        energy += np.sum(v_ij**2)
        
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        np.add.at(grad_centers, i_valid, -forces)
        np.add.at(grad_centers, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_k = np.argmax(abs_centers[idx])
            grad_centers[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
    
    # Backpropagate gradient from centers to params
    # params = [anchors (24), offsets (300)]
    # centers = W @ anchors + offsets
    # d(energy)/d(offsets) = grad_centers
    # d(energy)/d(anchors) = W^T @ grad_centers
    
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    lattice_pts = []
    for z in zs:
        for y in ys:
            for x in xs:
                lattice_pts.append((x, y, z))
    lattice_pts = np.array(lattice_pts[:N])
    
    x_norm = (lattice_pts[:, 0] - (-0.4)) / 0.8
    y_norm = (lattice_pts[:, 1] - (-0.4)) / 0.8
    z_norm = (lattice_pts[:, 2] - (-0.4)) / 0.8
    
    w000 = (1-x_norm)*(1-y_norm)*(1-z_norm)
    w100 = x_norm*(1-y_norm)*(1-z_norm)
    w010 = (1-x_norm)*y_norm*(1-z_norm)
    w110 = x_norm*y_norm*(1-z_norm)
    w001 = (1-x_norm)*(1-y_norm)*z_norm
    w101 = x_norm*(1-y_norm)*z_norm
    w011 = (1-x_norm)*y_norm*z_norm
    w111 = x_norm*y_norm*z_norm
    
    W = np.stack([w000, w100, w010, w110, w001, w101, w011, w111], axis=1)  # (N, 8)
    
    grad_offsets = grad_centers  # (N, 3)
    grad_anchors = W.T @ grad_centers  # (8, 3)
    
    grad_params = np.concatenate([grad_anchors.flatten(), grad_offsets.flatten()])
    
    return energy, grad_params

def _minimize_energy_jellyfish(init_params, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy in jellyfish parameter space.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient_jellyfish(x, target_R)
        return val, g

    bounds = [(-0.45, 0.45)] * (24 + N*3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_params,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_params = res.x
        new_centers = _jellyfish_to_centers(new_params)
        final_energy, _ = _energy_and_gradient_jellyfish(new_params, target_R)
        return new_params, new_centers, final_energy, res.success
    except Exception:
        return init_params, _jellyfish_to_centers(init_params), np.inf, False

def _minimize_energy_direct(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Standard direct optimization in center space (for polishing).
    """
    def _energy_and_gradient(x, R):
        centers = x.reshape(-1, 3)
        n = len(centers)
        
        energy = 0.0
        grad = np.zeros_like(centers)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

        abs_centers = np.abs(centers)
        max_abs = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_abs
        
        mask_wall = wall_dists < R
        if np.any(mask_wall):
            violations_wall = R - wall_dists[mask_wall]
            i_valid_wall = np.where(mask_wall)[0]
            
            energy += np.sum(violations_wall**2)
            
            for idx in i_valid_wall:
                v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
                max_k = np.argmax(abs_centers[idx])
                grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
        
        return energy, grad.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            lambda x: _energy_and_gradient(x, target_R),
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialize with grid + noise
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    init_centers = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    init_centers = np.clip(init_centers, -0.48, 0.48)
    
    # Convert to jellyfish params
    init_params = _centers_to_jellyfish(init_centers)
    
    # Initial optimization to establish baseline
    initial_r = 0.1
    params, centers, energy, _ = _minimize_energy_jellyfish(init_params, initial_r, max_iter=300)
    best_r = _compute_radius(centers)
    best_params = params.copy()
    best_centers = centers.copy()
    
    if best_r <= 0:
        best_centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_centers)
        best_params = _centers_to_jellyfish(best_centers)
    
    # Iterative radius increase using jellyfish optimization
    current_r = best_r
    step_size = 0.0008
    consecutive_failures = 0
    max_failures = 8
    
    while time.time() < deadline - 30.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.75
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        success = False
        
        # Strategy 1: Jellyfish optimization from current best
        new_params, new_centers, energy, opt_success = _minimize_energy_jellyfish(
            best_params, target_r, max_iter=400
        )
        actual_r = _compute_radius(new_centers)
        if actual_r >= target_r - 1e-6:
            best_params = new_params
            best_centers = new_centers
            best_r = actual_r
            current_r = actual_r
            success = True
            consecutive_failures = 0
        
        # Strategy 2: Perturb anchors in jellyfish space
        if not success and time.time() < deadline - 35.0:
            perturbed_params = best_params.copy()
            # Perturb anchors more strongly
            anchor_noise = rng.normal(0, 0.02, size=(8, 3))
            perturbed_params[0:24] += anchor_noise.flatten()
            # Small offset noise
            offset_noise = rng.normal(0, 0.005, size=(N, 3))
            perturbed_params[24:] += offset_noise.flatten()
            
            new_params, new_centers, energy, opt_success = _minimize_energy_jellyfish(
                perturbed_params, target_r, max_iter=400
            )
            actual_r = _compute_radius(new_centers)
            if actual_r >= target_r - 1e-6:
                best_params = new_params
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > current_r:
                best_params = new_params
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
                consecutive_failures = 0
        
        # Strategy 3: Direct optimization from current best (escape jellyfish constraint)
        if not success and time.time() < deadline - 40.0:
            start_pos = best_centers + rng.normal(0, 0.003, size=(N, 3))
            start_pos = np.clip(start_pos, -0.48, 0.48)
            new_centers, energy, opt_success = _minimize_energy_direct(
                start_pos, target_r, max_iter=300
            )
            actual_r = _compute_radius(new_centers)
            if actual_r >= target_r - 1e-6:
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
                best_params = _centers_to_jellyfish(new_centers)
                success = True
                consecutive_failures = 0
            elif actual_r > current_r:
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
                best_params = _centers_to_jellyfish(new_centers)
                consecutive_failures = 0
        
        if not success:
            consecutive_failures += 1
            
            # Periodic polish
            if time.time() < deadline - 45.0 and consecutive_failures % 4 == 0:
                polished, _, _ = _minimize_energy_direct(best_centers, current_r, max_iter=800, ftol=1e-16, gtol=1e-14)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_centers = polished
                    best_r = polished_r
                    current_r = polished_r
                    best_params = _centers_to_jellyfish(polished)
    
    # Final high-precision polish in direct space
    if time.time() < deadline - 5.0:
        for polish_iter in range(4):
            if time.time() > deadline - 5.0:
                break
            
            target_r = current_r + 1e-5 * (polish_iter + 1)
            new_centers, _, _ = _minimize_energy_direct(
                best_centers, target_r, max_iter=1500, ftol=1e-18, gtol=1e-16
            )
            actual_r = _compute_radius(new_centers)
            if actual_r > best_r:
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
            else:
                new_centers, _, _ = _minimize_energy_direct(
                    best_centers, current_r, max_iter=2000, ftol=1e-18, gtol=1e-16
                )
                actual_r = _compute_radius(new_centers)
                if actual_r > best_r:
                    best_centers = new_centers
                    best_r = actual_r
                    current_r = actual_r
                else:
                    break
    
    # Final validation
    final_pos = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_pos)
    
    return final_pos
# EVOLVE-BLOCK-END
