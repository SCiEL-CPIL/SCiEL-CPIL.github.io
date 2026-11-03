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

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Uses vectorized operations for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
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

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        # Vectorized wall gradient
        for idx in i_valid_wall:
            v = violations_wall[list(i_valid_wall).index(idx)]
            max_k = np.argmax(abs_centers[idx])
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _soft_min_objective_and_grad(x, beta=100.0):
    """
    Smooth surrogate for maximizing the minimum distance using soft-min.
    Maximizes soft_min(min_pair_dist/2, min_wall_dist).
    Vectorized gradient computation.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # All constraints: pairwise/2 and wall distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    all_constraints = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: -log(sum(exp(-beta * x))) / beta
    scaled = -beta * all_constraints
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Objective: soft_min
    objective = -(np.log(sum_exp) + max_scaled) / beta
    
    # Gradient - vectorized pairwise part
    grad = np.zeros_like(centers)
    
    # Pairwise gradient (vectorized)
    pair_weights = weights[:len(pair_dists)]
    if len(pair_weights) > 0:
        # Filter out zero weights and tiny distances for efficiency
        valid_mask = pair_weights > 1e-10
        if np.any(valid_mask):
            pw = pair_weights[valid_mask]
            ii = i_idx[valid_mask]
            jj = j_idx[valid_mask]
            dd = dists[ii, jj]
            valid_d = dd > 1e-12
            if np.any(valid_d):
                pw = pw[valid_d]
                ii = ii[valid_d]
                jj = jj[valid_d]
                dd = dd[valid_d]
                
                vecs = centers[ii] - centers[jj]
                factors = beta * pw / dd
                forces = factors[:, np.newaxis] * vecs
                
                np.add.at(grad, ii, forces)
                np.add.at(grad, jj, -forces)
    
    # Wall gradient (vectorized)
    wall_weights = weights[len(pair_dists):]
    valid_wall = wall_weights > 1e-10
    if np.any(valid_wall):
        ww = wall_weights[valid_wall]
        wi = np.where(valid_wall)[0]
        max_k = np.argmax(abs_centers[wi], axis=1)
        signs = np.sign(centers[wi, max_k])
        grad[wi, max_k] -= beta * ww * signs
    
    return objective, grad.flatten()

def _soft_min_optimize(centers, beta=100.0, max_iter=500):
    """
    Optimize using soft-min objective to directly maximize radius.
    """
    def obj_and_grad(x):
        val, g = _soft_min_objective_and_grad(x, beta)
        return val, g
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers

def _energy_minimize(centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius using L-BFGS-B.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return centers, np.inf, False

def _active_set_polish(centers, R, max_iter=3000, ftol=1e-18, gtol=1e-16):
    """
    Final high-precision polish: identify active constraints and solve
    a smaller problem on just those variables/constraints.
    """
    # First, do a standard high-precision optimization
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
    except Exception:
        return centers
    
    # Check if we're at the boundary and try to push further
    # by slightly increasing R and re-optimizing
    for delta in [1e-6, 5e-7, 2e-7, 1e-7]:
        test_R = R + delta
        def test_obj_and_grad(x):
            val, g = _energy_and_gradient(x, test_R)
            return val, g
        
        try:
            res2 = minimize(
                test_obj_and_grad,
                new_centers.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
            )
            test_centers = res2.x.reshape(N, 3)
            test_r = _compute_radius(test_centers)
            if test_r >= test_R - 1e-8:
                new_centers = test_centers
                R = test_R
        except Exception:
            pass
    
    return new_centers

def _radius_increase_search(centers, current_r, deadline, rng):
    """
    Iterative radius increase with multiple strategies.
    """
    best_pos = centers.copy()
    best_r = current_r
    current_r = best_r
    step_size = 0.0003
    
    consecutive_failures = 0
    max_failures = 4
    
    T_initial = 0.01
    T_min = 0.001
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.6
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        success = False
        
        # Strategy 1: Direct energy minimization
        max_iter = 400 if current_r < 0.11 else 600
        new_pos, energy, opt_success = _energy_minimize(best_pos, target_r, max_iter=max_iter)
        actual_r = _compute_radius(new_pos)
        if actual_r >= target_r - 1e-6:
            best_pos = new_pos
            best_r = actual_r
            current_r = actual_r
            success = True
            consecutive_failures = 0
        
        # Strategy 2: Soft-min optimization (smoother landscape)
        if not success and time.time() < deadline - 20.0:
            new_pos = _soft_min_optimize(best_pos, beta=80.0, max_iter=300)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > current_r + 1e-6:
                # Accept improvement
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                consecutive_failures = 0
        
        # Strategy 3: Perturbation + optimization
        if not success and time.time() < deadline - 25.0:
            start_pos = best_pos + rng.normal(0, 0.004, size=(N, 3))
            start_pos = np.clip(start_pos, -0.49, 0.49)
            new_pos, energy, opt_success = _energy_minimize(start_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > current_r + 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                consecutive_failures = 0
        
        # Strategy 4: Simulated annealing escape
        if not success and time.time() < deadline - 30.0:
            T = T_initial * (T_min / T_initial) ** (consecutive_failures / max_failures)
            sa_pos = best_pos.copy()
            current_energy, _ = _energy_and_gradient(sa_pos, current_r)
            for _ in range(80):
                perturbation = rng.normal(0, 0.015, size=sa_pos.shape)
                proposed = sa_pos + perturbation
                proposed = np.clip(proposed, -0.49, 0.49)
                proposed_energy, _ = _energy_and_gradient(proposed, current_r)
                delta_E = proposed_energy - current_energy
                if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
                    sa_pos = proposed
                    current_energy = proposed_energy
            
            new_pos, energy, opt_success = _energy_minimize(sa_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > current_r + 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                consecutive_failures = 0
        
        if not success:
            consecutive_failures += 1
            
            # Periodic polish
            if time.time() < deadline - 35.0 and consecutive_failures % 3 == 0:
                polished = _active_set_polish(best_pos, current_r, max_iter=800)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization: 5x4x5 Grid with symmetry breaking
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    # Add small noise to break symmetry
    best_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Initial optimization to get a good baseline
    initial_r = 0.1
    best_pos, energy, _ = _energy_minimize(best_pos, initial_r, max_iter=200)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)

    # 2. Main search loop
    rng_search = np.random.default_rng(123)
    best_pos, best_r = _radius_increase_search(best_pos, best_r, deadline, rng_search)

    # 3. Final High-Precision Polish Phase
    if time.time() < deadline - 5.0:
        # Multiple polish attempts with increasing precision
        for polish_iter in range(4):
            if time.time() > deadline - 5.0:
                break
            
            # Try to push slightly beyond current radius
            target_r = best_r + 1e-5 * (polish_iter + 1)
            max_iter = 1200 + 400 * polish_iter
            
            new_pos = _active_set_polish(best_pos, target_r, max_iter=max_iter, ftol=1e-17, gtol=1e-15)
            actual_r = _compute_radius(new_pos)
            
            if actual_r > best_r:
                best_pos = new_pos
                best_r = actual_r
            else:
                # Try polishing at current radius
                new_pos = _active_set_polish(best_pos, best_r, max_iter=2000, ftol=1e-18, gtol=1e-16)
                actual_r = _compute_radius(new_pos)
                if actual_r > best_r:
                    best_pos = new_pos
                    best_r = actual_r
                else:
                    break

    # Final validation
    final_pos = np.clip(best_pos, -0.5, 0.5)
    final_r = _compute_radius(final_pos)
    
    return final_pos
# EVOLVE-BLOCK-END
