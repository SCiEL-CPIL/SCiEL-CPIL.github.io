"""
Optimization of 100 equal spheres in a unit cube.
Uses a barrier method with L-BFGS-B optimization to maximize the minimum radius.
"""
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
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R, beta=50.0):
    """
    Calculates a smooth surrogate energy and gradient for a target radius R.
    Uses soft-min via log-sum-exp to smooth the nonsmooth min objective.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Pairwise terms - smooth version
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # For each pair, violation = max(0, 2R - dist)
    # Smooth approximation: use softplus-like function
    # violation_smooth = (1/beta) * log(1 + exp(beta*(2R - dist)))
    # But we want squared violation for energy. Let's use:
    # energy_pair = sum over i<j of softplus(beta*(2R - dist_ij))^2 / beta^2
    
    # Actually, let's use a simpler smooth approach:
    # For pairs where dist < 2R, the violation is (2R - dist)
    # We smooth this with: v_smooth = (1/beta) * log(1 + exp(beta*(2R - dist)))
    # Energy = sum(v_smooth^2)
    
    # Compute for all pairs, then take upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Soft violation: smooth max(0, 2R - d)
    arg = beta * (2.0 * R - d_ij)
    # Use log(1+exp(arg)) but clip to avoid overflow
    arg = np.clip(arg, -50, 50)
    soft_viol = (1.0 / beta) * np.log1p(np.exp(arg))
    
    energy = np.sum(soft_viol**2)
    
    # Gradient of soft_viol w.r.t. d_ij:
    # d(soft_viol)/d(d_ij) = -exp(arg) / (1 + exp(arg)) = -sigmoid(arg)
    sigmoid_arg = 1.0 / (1.0 + np.exp(-arg))
    d_soft_viol_dd = -sigmoid_arg / beta
    
    # d(energy)/d(centers[i]) = sum over j of 2*soft_viol_ij * d_soft_viol_dd_ij * d(d_ij)/d(centers[i])
    # d(d_ij)/d(centers[i]) = (centers[i] - centers[j]) / d_ij
    
    grad = np.zeros_like(centers)
    
    # Vectorized gradient computation
    vec_ij = centers[i_idx] - centers[j_idx]
    d_safe = np.where(d_ij > 1e-10, d_ij, 1e-10)
    
    # factor_ij = 2 * soft_viol * d_soft_viol_dd / d_ij
    factor = 2.0 * soft_viol * d_soft_viol_dd / d_safe
    
    # grad[i] += sum_j factor_ij * vec_ij
    # grad[j] -= sum_i factor_ij * vec_ij
    
    # Use np.add.at for accumulation
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        f = factor[k]
        v = vec_ij[k]
        grad[i] += f * v
        grad[j] -= f * v
    
    # 2. Wall terms - smooth version
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    # Soft violation for walls: smooth max(0, R - wall_dist)
    arg_wall = beta * (R - wall_dists)
    arg_wall = np.clip(arg_wall, -50, 50)
    soft_viol_wall = (1.0 / beta) * np.log1p(np.exp(arg_wall))
    
    energy += np.sum(soft_viol_wall**2)
    
    # Gradient for wall term
    sigmoid_wall = 1.0 / (1.0 + np.exp(-arg_wall))
    d_soft_viol_dd_wall = -sigmoid_wall / beta
    
    # d(wall_dist)/d(centers[i,k]) = -sign(centers[i,k]) if centers[i,k] is the max abs component
    for i in range(n):
        max_val = max_abs[i]
        for k in range(3):
            if np.isclose(abs_centers[i, k], max_val):
                sign_k = np.sign(centers[i, k])
                if sign_k == 0:
                    sign_k = 1.0
                grad[i, k] += 2.0 * soft_viol_wall[i] * d_soft_viol_dd_wall[i] * (-sign_k)
                break
    
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, beta=50.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    Returns the optimized centers.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def _polish(centers, max_iter=500):
    """
    Final high-precision polish pass to maximize the radius.
    """
    r_current = _compute_radius(centers)
    if r_current <= 0:
        return centers
    
    # Use a very smooth objective and tight tolerances
    beta = 100.0
    
    def objective_and_grad(x):
        # Maximize r = min(min_pair_dist/2, min_wall_dist)
        # Equivalent to minimizing -r, but r is nonsmooth.
        # Use the energy approach with target_R = r_current + small_delta
        # Actually, let's directly optimize for larger radius
        c = x.reshape(-1, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        min_wall = np.min(wall_dists)
        r = min(min_pair, min_wall)
        # We want to maximize r, so minimize -r
        # But -r is nonsmooth. Use the energy surrogate with target slightly above current.
        target = r_current * 1.001
        val, g = _energy_and_gradient(x, target, beta)
        return val, g
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        polished = res.x.reshape(-1, 3)
        if _compute_radius(polished) > r_current:
            return polished
    except Exception:
        pass
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_radius = 0.0
    best_centers = None
    
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # Multiple grid configurations
    grid_545 = make_grid(5, 4, 5)
    grid_455 = make_grid(4, 5, 5)
    grid_554 = make_grid(5, 5, 4)
    
    # BCC-like initialization: face-centered positions
    def make_bcc_like():
        # 4x4x4 grid of 64 points, plus 36 more
        # Actually, let's use a different approach
        rng_local = np.random.RandomState(123)
        pts = rng_local.uniform(-0.45, 0.45, size=(N, 3))
        return pts
    
    rng = np.random.RandomState(42)
    random_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    inits = [grid_545, grid_455, grid_554, random_centers]
    
    for i in range(8):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
            
        r_est = _compute_radius(init)
        
        target_R = r_est * 1.005
        if target_R < 0.05:
            target_R = 0.05
            
        current_centers = init.copy()
        
        while time.time() < deadline - 10.0:
            optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=150, beta=50.0)
            
            r_achieved = _compute_radius(optimized_centers)
            
            if r_achieved > best_radius:
                best_radius = r_achieved
                best_centers = optimized_centers.copy()
            
            if energy < 1e-8:
                target_R *= 1.003
                current_centers = optimized_centers
                if target_R > 0.115:
                    target_R *= 1.0005
            else:
                # Try a slightly smaller target to relax
                target_R *= 0.99
                if target_R < r_est * 0.95:
                    break
        
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()
    
    # Final polish pass
    if best_centers is not None and time.time() < deadline - 5.0:
        polished = _polish(best_centers, max_iter=300)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
    
    if best_centers is None:
        best_centers = grid_545
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        
    return best_centers
# EVOLVE-BLOCK-END
