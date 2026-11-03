"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    # Compute pairwise distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to walls
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_and_grad(centers, target_r):
    """Compute repulsion energy and its analytical gradient.
    
    Energy: sum of squared violations for pairs closer than 2*target_r
            + sum of squared wall violations.
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For pairs closer than 2*target_r, add (2*r - d)^2 to energy
    # Gradient w.r.t. centers[i]: -2*(2*r - d) * (centers[i] - centers[j]) / d
    close_mask = dists < 2.0 * target_r
    if np.any(close_mask):
        violations = 2.0 * target_r - dists
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(n, k=1)
        close_iu = close_mask[iu]
        if np.any(close_iu):
            vi, vj = iu[0][close_iu], iu[1][close_iu]
            dv = violations[vi, vj]  # shape (k,)
            dij = dists[vi, vj]      # shape (k,)
            diff_ij = centers[vi] - centers[vj]  # shape (k, 3)
            
            # Gradient contribution for each pair (vi, vj):
            # d/dx_i [(2r - d)^2] = -2(2r - d) * (x_i - x_j)/d
            # d/dx_j [(2r - d)^2] = -2(2r - d) * (x_j - x_i)/d = +2(2r - d) * (x_i - x_j)/d
            coeff = 2.0 * dv / dij  # shape (k,)
            
            # Accumulate gradient using np.add.at for repeated indices
            for idx in range(len(vi)):
                grad[vi[idx]] -= coeff[idx] * diff_ij[idx]
                grad[vj[idx]] += coeff[idx] * diff_ij[idx]
            
            energy += np.sum(dv**2)
    
    # Wall repulsion
    wall_dists = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0, target_r - wall_dists)
    energy += np.sum(wall_violations**2)
    
    # Gradient for wall violations:
    # d/dx_i [max(0, r - (0.5 - |x_i|))^2] 
    # = -2 * max(0, r - (0.5 - |x_i|)) * sign(x_i)  if violation > 0
    wall_viol = target_r - wall_dists
    active_wall = wall_viol > 0
    if np.any(active_wall):
        signs = np.sign(centers)
        # Where center is 0, sign is 0; handle carefully
        signs[centers == 0] = 1  # arbitrary but consistent
        grad[active_wall] -= 2.0 * wall_violations[active_wall] * signs[active_wall]
    
    return energy, grad

def _optimize_from_init(centers_init, target_r, max_iter=500, tol=1e-12):
    """Optimize from a given initial configuration using L-BFGS-B with bounds."""
    bounds = [(-0.5, 0.5)] * (len(centers_init) * 3)
    
    def objective_and_grad(x):
        c = x.reshape(-1, 3)
        e, g = _energy_and_grad(c, target_r)
        return e, g.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            centers_init.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'disp': False}
        )
        return result.x.reshape(-1, 3), result.success, result.fun
    except Exception:
        return centers_init, False, np.inf

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _face_centered_init(n):
    """Initialize with points near face centers and corners for better spread."""
    rng = np.random.RandomState(42)
    centers = []
    # Add some points near each face center
    for axis in range(3):
        for sign in [-1, 1]:
            for _ in range(2):
                p = np.zeros(3)
                p[axis] = sign * 0.3
                p += rng.uniform(-0.1, 0.1, size=3)
                centers.append(p)
    # Fill remaining with random points
    while len(centers) < n:
        centers.append(rng.uniform(-0.45, 0.45, size=3))
    return np.array(centers[:n])

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    inits.append(_grid_init(N))
    inits.append(_face_centered_init(N))
    for seed in range(30):
        inits.append(_random_init(N, seed=seed))
    
    # Evaluate initial radii and track best
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        r_est = _compute_radius(init)
        if r_est > best_radius:
            best_radius = r_est
            best_centers = init.copy()
    
    # Sort initializations by initial radius (best first)
    scored = sorted([( _compute_radius(init), init) for init in inits], key=lambda x: -x[0])
    
    # Optimize from top initializations
    for r_est, init in scored:
        if time.time() > deadline - 30.0:
            break
        if r_est < best_radius * 0.9:  # skip poor initializations
            continue
        
        target_r = max(best_radius, r_est)
        try:
            optimized, success, energy = _optimize_from_init(
                init, target_r, max_iter=300, tol=1e-12
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            continue
    
    # Final polish: multiple refinement passes from the best solution
    if best_centers is not None:
        for polish_iter in range(8):
            if time.time() > deadline - 15.0:
                break
            
            # Slight perturbation to escape local minima
            noise_scale = 0.005 * (1.0 - polish_iter / 10.0)
            perturbed = best_centers + np.random.randn(N, 3) * noise_scale
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            try:
                optimized, success, energy = _optimize_from_init(
                    perturbed, best_radius, max_iter=200, tol=1e-13
                )
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Final high-precision polish on the best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            optimized, success, energy = _optimize_from_init(
                best_centers, best_radius, max_iter=500, tol=1e-14
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _grid_init(N)
    
    # Final validation: ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
