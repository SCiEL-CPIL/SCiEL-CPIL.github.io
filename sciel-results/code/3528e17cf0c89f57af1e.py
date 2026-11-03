"""
Optimization of 100 equal spheres in a unit cube.
Uses a combination of simulated annealing and L-BFGS-B optimization.
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
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    mask = dists < 2.0 * R
    if np.any(mask):
        # Use upper triangle to avoid double counting in energy
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        m_ij = mask[i_idx, j_idx]
        
        if np.any(m_ij):
            v_ij = 2.0 * R - d_ij[m_ij]
            energy += np.sum(v_ij**2)
            
            # Gradient
            # Force on i from j: -2 * v * (c_i - c_j) / d
            # We update grad for both i and j
            for k in range(len(m_ij)):
                if not m_ij[k]:
                    continue
                i = i_idx[k]
                j = j_idx[k]
                v = v_ij[k] # This is 2R - d
                d = d_ij[k]
                factor = -2.0 * v / d
                vec = centers[i] - centers[j]
                force = factor * vec
                grad[i] += force
                grad[j] -= force

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        v_wall = (R - wall_dists)[mask_wall]
        energy += np.sum(v_wall**2)
        
        # Better loop for walls
        for i in range(n):
            if not mask_wall[i]:
                continue
            v = R - wall_dists[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            # dE/dc = 2 * (R - d_wall) * d(R - d_wall)/dc
            # d(R - d_wall)/dc = - d(d_wall)/dc
            # d(d_wall)/dc = - grad_wall_i (where grad_wall_i is d(max_abs)/dc)
            # So d(R - d_wall)/dc = grad_wall_i
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
        
        # Check time budget
        if time.time() - start_time > time_budget:
             # If we hit the time limit, we might want to interrupt, but L-BFGS-B doesn't support interruption easily.
             # We rely on max_iter being set reasonably.
             pass
            
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def _simulated_annealing(centers, target_R, max_iter=5000, start_temp=0.01, end_temp=0.001, time_budget=10.0):
    """
    Simulated annealing to find a configuration that fits within target_R.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_energy = max(0, target_R - current_radius)
    
    for i in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Random perturbation
        new_centers = centers.copy()
        idx = np.random.randint(0, N)
        new_centers[idx] += np.random.normal(0, temp, size=3)
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        new_energy = max(0, target_R - new_radius)
        
        delta_energy = new_energy - current_energy
        
        # Acceptance criterion
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / (temp + 1e-10)):
            centers = new_centers
            current_energy = new_energy
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = new_centers.copy()
                
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for the 150s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy 1: Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/ny, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # 5x4x5 = 100
    grid_centers = make_grid(5, 4, 5)
    
    # Strategy 2: Random Initialization
    rng = np.random.RandomState(42)
    random_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    # Create a set of initial guesses
    inits = [grid_centers, random_centers]
    
    # Add a few more diverse random inits
    for i in range(5):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
            
        # 1. Calculate initial radius
        r_est = _compute_radius(init)
        
        # 2. Iteratively increase target radius
        target_R = r_est * 1.01
        if target_R < 0.05:
            target_R = 0.05
            
        current_centers = init.copy()
        max_iter_per_step = 150
        min_iter_per_step = 50
        
        while time.time() < deadline - 10.0:
            # Try to optimize for target_R using simulated annealing first
            time_budget_for_sa = min(5.0, max(1.0, (deadline - 5.0 - time.time())))
            sa_centers, sa_radius = _simulated_annealing(current_centers, target_R, max_iter=2000, time_budget=time_budget_for_sa)
            
            # Check if SA achieved a valid packing for this radius
            if sa_radius >= target_R:
                r_achieved = _compute_radius(sa_centers)
                
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = sa_centers.copy()
                    
                # Try to increase the radius
                target_R *= 1.005
                current_centers = sa_centers
                
                if target_R > 0.12:
                    target_R *= 1.001
            else:
                # SA didn't achieve target_R, try L-BFGS-B
                time_budget_for_lbfgs = min(5.0, max(1.0, (deadline - 5.0 - time.time())))
                optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=100, time_budget=time_budget_for_lbfgs)
                
                # Check if we achieved a valid packing for this radius
                if energy < 1e-6:
                    r_achieved = _compute_radius(optimized_centers)
                    
                    if r_achieved > best_radius:
                        best_radius = r_achieved
                        best_centers = optimized_centers.copy()
                        
                    # Try to increase the radius
                    target_R *= 1.005
                    current_centers = optimized_centers
                    
                    if target_R > 0.12:
                        target_R *= 1.001
                else:
                    # Failed to fit this radius.
                    # We might have overshot. Try a slightly smaller R or break.
                    # If energy is high, it means we are far from the solution.
                    # Let's try to reduce target_R slightly to see if we can recover,
                    # or just break if we are stuck.
                    
                    # Check if current centers are better than best
                    r_curr = _compute_radius(current_centers)
                    if r_curr > best_radius:
                        best_radius = r_curr
                        best_centers = current_centers.copy()
                    
                    # If energy is not zero, we cannot proceed with higher R from this config.
                    # We could try to relax the constraints to get a local minimum of the energy,
                    # but that's complex. For now, we break and move to the next init.
                    # However, we can try a few times with a slightly lower R to "heal" the config.
                    
                    heal_attempts = 0
                    heal_R = target_R * 0.99
                    while heal_attempts < 3 and time.time() < deadline - 10.0:
                        h_centers, h_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50)
                        if h_energy < 1e-4:
                            current_centers = h_centers
                            target_R = heal_R * 1.002 # Try slightly higher again
                            heal_attempts = 0 # Reset counter
                        else:
                            heal_R *= 0.99
                            heal_attempts += 1
                            # If we can't even fit a smaller radius, this config is dead
                            if h_energy > 1.0:
                                break
                    
                    if heal_attempts >= 3 or (time.time() > deadline - 10.0):
                        break
        
        # Update best with current optimized centers
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final safety check
    if best_centers is None:
        best_centers = grid_centers
    else:
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
        
    return best_centers
# EVOLVE-BLOCK-END
