"""Combined program: iterative radius increase with energy minimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

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
    return float(r)

def _energy_and_gradient(x, R, beta=50.0):
    """
    Calculates a smooth surrogate energy for target radius R.
    Uses a soft-min (log-sum-exp) over squared constraint violations
    to create a differentiable objective that approximates the hard min.
    
    E = (1/beta) * log(sum(exp(beta * v_i^2)))
    where v_i are constraint violations (positive when violated).
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Pairwise constraint violations: 2R - d_ij (positive when too close)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(sq_dists, 1e-30))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Violations for pairs (positive when d_ij < 2R)
    pair_violations = 2.0 * R - d_ij
    
    # 2. Wall constraint violations: R - (0.5 - max|c_ik|)
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    wall_violations = R - wall_dists
    
    # Combine all violations into one vector
    all_violations = np.concatenate([pair_violations, wall_violations])
    
    # Soft-min via log-sum-exp for numerical stability
    # We want to minimize the "worst" violation, so we use:
    # soft_min(v) = (1/beta) * log(sum(exp(beta * v^2)))
    # This is smooth and approximates max(v_i^2) as beta -> inf
    
    # For numerical stability, subtract max
    v_sq = all_violations ** 2
    max_v_sq = np.max(v_sq)
    
    if max_v_sq < 1e-30:
        return 0.0, np.zeros_like(x)
    
    exp_vals = np.exp(beta * (v_sq - max_v_sq))
    log_sum = np.log(np.sum(exp_vals)) + beta * max_v_sq
    energy = log_sum / beta
    
    # Gradient: dE/dx_i = (1/beta) * d/dx_i log(sum(exp(beta * v_j^2)))
    #           = sum_j [exp(beta * v_j^2) * beta * 2*v_j * dv_j/dx_i] / sum_k exp(beta * v_k^2)
    #           = sum_j [exp_vals[j] * 2*v_j * dv_j/dx_i] / sum(exp_vals)
    
    weights = exp_vals / np.sum(exp_vals)  # normalized weights
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradient contribution
    # d(pair_violation_ij)/dx_i = -(x_i - x_j)/d_ij
    # dE/dx_i = sum_j weights_ij * 2 * pair_violation_ij * (-(x_i - x_j)/d_ij)
    
    # Only compute for pairs with significant weight to save time
    pair_weights = weights[:len(pair_violations)]
    active_pairs = pair_weights > 1e-10
    
    if np.any(active_pairs):
        i_a = i_idx[active_pairs]
        j_a = j_idx[active_pairs]
        w_a = pair_weights[active_pairs]
        v_a = pair_violations[active_pairs]
        d_a = d_ij[active_pairs]
        
        # Factor: 2 * v * w / d
        factors = 2.0 * v_a * w_a / d_a
        vecs = centers[i_a] - centers[j_a]
        forces = factors[:, np.newaxis] * vecs
        
        # dE/dx_i = -sum forces (push i away from j)
        # dE/dx_j = +sum forces (push j away from i)
        np.add.at(grad, i_a, -forces)
        np.add.at(grad, j_a, forces)
    
    # Wall gradient contribution
    wall_weights = weights[len(pair_violations):]
    active_walls = wall_weights > 1e-10
    
    if np.any(active_walls):
        for idx in np.where(active_walls)[0]:
            w = wall_weights[idx]
            v = wall_violations[idx]
            max_k = np.argmax(abs_centers[idx])
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            # dE/dx_i[k] = 2 * v * w * sign_val (push away from wall)
            grad[idx, max_k] += 2.0 * v * w * sign_val
    
    return float(energy), grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12, beta=50.0):
    """
    Minimizes the smooth surrogate energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta=beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R, beta=beta)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT  # Safety margin before 150s kill
    
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
    # Start with a conservative radius
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=200, beta=30.0)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
         best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
         best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase
    current_r = best_r
    step_size = 0.0008  # Slightly larger steps for faster progress
    
    consecutive_failures = 0
    max_failures = 4
    
    rng_search = np.random.default_rng(123)
    
    # Adaptive beta: start lower for smoother landscape, increase for precision
    beta = 30.0
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.6
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.003, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.01, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.03, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Increase max_iter for higher precision as we get closer to optimum
            max_iter = 400 if current_r < 0.11 else 600
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter, beta=beta)
            
            # Check actual radius achieved
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
                break
            elif actual_r > current_r:
                # Improved but didn't reach target - accept the improvement
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
        
        if not success:
            consecutive_failures += 1
            
            # If we failed, try to polish the current best position with a slightly smaller target
            # or just re-optimize the current best to ensure it's local optimal
            if time.time() < deadline - 20.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=300, beta=beta)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r
        
        # Gradually increase beta for more precise optimization
        if current_r > 0.105:
            beta = min(80.0, beta + 5.0)

    # 3. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        # Try to push slightly beyond current best with very tight tolerances and high beta
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            target_r = best_r + 1e-6
            polished, energy, _ = _minimize_energy(best_pos, target_r, max_iter=2000, ftol=1e-16, gtol=1e-15, beta=100.0)
            polished_r = _compute_radius(polished)
            if polished_r > best_r:
                best_pos = polished
                best_r = polished_r
    
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
