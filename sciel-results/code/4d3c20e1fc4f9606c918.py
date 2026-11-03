"""
Optimization of 100 equal spheres in a unit cube.
Combines L-BFGS-B heuristic search with SLSQP constrained optimization for precise radius maximization.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
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
    
    # Vectorized computation for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        # Gradient: for each violating pair, add force
        # dE/dc_i = -2 * v_ij * (c_i - c_j) / d_ij
        # dE/c_j = 2 * v_ij * (c_i - c_j) / d_ij
        # We only iterate over violating pairs for efficiency if many are not violating
        # But for N=100, 4950 pairs is small.
        for k in range(len(v_ij)):
            if v_ij[k] <= 0:
                continue
            i = i_idx[k]
            j = j_idx[k]
            v = v_ij[k]
            d = d_ij[k]
            if d < 1e-10:
                continue
            factor = -2.0 * v / d
            vec = centers[i] - centers[j]
            force = factor * vec
            grad[i] += force
            grad[j] -= force

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        for i in range(n):
            if v_wall[i] <= 0:
                continue
            v = v_wall[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_budget=5.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > time_budget:
             pass
            
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def _slsqp_maximize_radius(centers_init, time_budget=10.0):
    """
    Uses SLSQP to maximize the radius r subject to non-overlap and wall constraints.
    Variables: x = [c_1, ..., c_N, r]
    Objective: Minimize -r
    Constraints:
    1. ||c_i - c_j|| - 2r >= 0 for all i < j
    2. 0.5 - |c_ik| - r >= 0 for all i, k
    """
    start_time = time.time()
    
    # Flatten centers and append r
    # Initial r estimate
    r_init = _compute_radius(centers_init)
    if r_init <= 0:
        r_init = 0.01
    x0 = np.concatenate([centers_init.flatten(), [r_init]])
    
    # Bounds for centers and r
    # Centers: [-0.5, 0.5]
    # r: [0, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]
    
    # Constraints for SLSQP
    # SLSQP expects constraints in the form: fun(x) >= 0 (for inequality, type='ineq')
    
    def pair_constraints(x):
        """
        Returns array of constraints: ||c_i - c_j|| - 2r
        """
        c = x[:-1].reshape(N, 3)
        r = x[-1]
        # Compute all pairwise distances
        # To save memory/time, we can compute in chunks or just use cdist
        # N=100, 100x100 matrix is small
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        # We need upper triangle values
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_ij = dists[i_idx, j_idx]
        return d_ij - 2.0 * r

    def wall_constraints(x):
        """
        Returns array of constraints: 0.5 - |c_ik| - r
        """
        c = x[:-1].reshape(N, 3)
        r = x[-1]
        # For each center, distance to nearest wall
        # max_abs = max(|cx|, |cy|, |cz|)
        # wall_dist = 0.5 - max_abs
        # constraint = wall_dist - r
        max_abs = np.max(np.abs(c), axis=1)
        wall_dists = 0.5 - max_abs
        return wall_dists - r

    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': pair_constraints},
        {'type': 'ineq', 'fun': wall_constraints}
    ]
    
    def objective(x):
        return -x[-1]

    def objective_grad(x):
        # Gradient of -r with respect to x
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad

    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        
        if time.time() - start_time > time_budget:
            # If it takes too long, we might not get the full result, 
            # but SLSQP doesn't easily interrupt. 
            # We rely on maxiter and the fact that it converges fast if close.
            pass
            
        if res.success:
            new_centers = res.x[:-1].reshape(N, 3)
            # Ensure centers are strictly inside
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            return new_centers
        else:
            # If SLSQP fails, return original or best effort
            # Sometimes SLSQP returns a valid point even if not "success"
            new_centers = res.x[:-1].reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            # Verify if it's better
            r_new = _compute_radius(new_centers)
            r_old = _compute_radius(centers_init)
            if r_new > r_old:
                return new_centers
            return centers_init
    except Exception:
        return centers_init

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy 1: Grid Initialization
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
    grid_configs = [
        (5, 4, 5),  # 100
        (4, 5, 5),  # 100
        (5, 5, 4),  # 100
        (10, 10, 1),  # 100 (flat)
        (1, 10, 10),  # 100 (flat)
        (10, 1, 10),  # 100 (flat)
    ]
    
    rng = np.random.RandomState(42)
    
    # Create a set of initial guesses
    inits = []
    for nx, ny, nz in grid_configs:
        if nx * ny * nz == N:
            inits.append(make_grid(nx, ny, nz))
    
    # Add random inits
    for i in range(5):
        rc = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(rc)
    
    # Add structured random: start from grid, add noise
    base_grid = make_grid(5, 4, 5)
    for i in range(3):
        noise_scale = 0.02 * (i + 1)
        noisy = base_grid + rng.normal(0, noise_scale, size=(N, 3))
        noisy = np.clip(noisy, -0.45, 0.45)
        inits.append(noisy)
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
            
        # 1. Calculate initial radius
        r_est = _compute_radius(init)
        
        # 2. Iteratively increase target radius using L-BFGS-B
        target_R = max(r_est * 1.005, 0.05)
        
        current_centers = init.copy()
        max_iter_per_step = 100
        
        stuck_count = 0
        while time.time() < deadline - 25.0:
            # Try to optimize for target_R
            time_budget_for_step = min(5.0, max(1.0, (deadline - 20.0 - time.time())))
            optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=max_iter_per_step, time_budget=time_budget_for_step)
            
            # Check if we achieved a valid packing for this radius
            if energy < 1e-8:
                r_achieved = _compute_radius(optimized_centers)
                
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = optimized_centers.copy()
                
                # Try to increase the radius
                increment = 0.003 if target_R < 0.11 else 0.001
                target_R *= (1.0 + increment)
                current_centers = optimized_centers
                stuck_count = 0
            else:
                # Failed to fit this radius
                r_curr = _compute_radius(current_centers)
                if r_curr > best_radius:
                    best_radius = r_curr
                    best_centers = current_centers.copy()
                
                stuck_count += 1
                
                # Try to heal with slightly smaller radius
                heal_attempts = 0
                heal_R = target_R * 0.98
                healed = False
                while heal_attempts < 3 and time.time() < deadline - 25.0:
                    h_centers, h_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50)
                    if h_energy < 1e-6:
                        current_centers = h_centers
                        target_R = heal_R * 1.002
                        healed = True
                        break
                    else:
                        heal_R *= 0.98
                        heal_attempts += 1
                        if h_energy > 1.0:
                            break
                
                if not healed or stuck_count > 5:
                    break
        
        # Update best with current optimized centers
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final high-precision polish pass using SLSQP
    # We apply this to the best candidate found so far
    if best_centers is not None and time.time() < deadline - 10.0:
        polish_time = min(15.0, deadline - 5.0 - time.time())
        polished = _slsqp_maximize_radius(best_centers, time_budget=polish_time)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
    
    # If we have time, try SLSQP on a few other good candidates from the loop?
    # The loop only kept the global best. 
    # To improve, we should store top K candidates and polish them.
    # However, given the time budget, polishing the single best is usually sufficient 
    # if the L-BFGS phase was robust.
    
    # Let's try to polish the second best if we have time?
    # The current code doesn't store history. 
    # Let's stick to polishing the single best to avoid complexity and time risks.
    
    # Final safety check
    if best_centers is None:
        best_centers = make_grid(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        # Ensure all values are finite
        if not np.all(np.isfinite(best_centers)):
            best_centers = make_grid(5, 4, 5)
        
    return best_centers
# EVOLVE-BLOCK-END
