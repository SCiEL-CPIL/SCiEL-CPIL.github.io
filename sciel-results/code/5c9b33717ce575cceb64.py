import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    if pos.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.max(np.abs(pos), axis=1)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def energy_and_gradient(x, target_R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Violations:
    1. Pairwise distance < 2R
    2. Wall distance < R
    
    Returns: (energy, gradient_vector)
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * target_R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        v_ij = 2.0 * target_R - d_ij
        
        active_mask = v_ij > 0
        if np.any(active_mask):
            v_active = v_ij[active_mask]
            d_active = d_ij[active_mask]
            i_active = i_idx[active_mask]
            j_active = j_idx[active_mask]
            
            energy += np.sum(v_active**2)
            
            factors = -2.0 * v_active / d_active
            vecs = diff[i_active, j_active]
            
            forces_i = factors[:, np.newaxis] * vecs
            forces_j = -forces_i
            
            np.add.at(grad, i_active, forces_i)
            np.add.at(grad, j_active, forces_j)
    
    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < target_R
    if np.any(mask_wall):
        v_wall = target_R - wall_dists
        v_wall[~mask_wall] = 0.0
        
        energy += np.sum(v_wall**2)
        
        signs = np.sign(centers)
        signs[signs == 0] = 1.0
        
        is_max = np.isclose(abs_centers, max_abs[:, np.newaxis], rtol=1e-12, atol=1e-12)
        
        grad_wall = 2.0 * v_wall[:, np.newaxis] * signs * is_max
        
        grad += grad_wall * mask_wall[:, np.newaxis]
        
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_limit=10.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    Returns the optimized centers.
    """
    def objective_and_grad(x):
        val, g = energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        return res.x.reshape(-1, 3)
    except Exception:
        return init_centers

def _polish_hard_min(pos, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    High-precision polish pass using the hard-min objective (maximize min dist).
    This is used to refine a feasible solution.
    """
    def hard_min_obj_and_grad(flat_pos):
        pos = flat_pos.reshape(N, 3)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.max(np.abs(pos), axis=1)
        min_wall_val = np.min(d_walls)
        min_wall_idx = np.argmin(d_walls)
        wi = min_wall_idx
        wk = np.argmax(np.abs(pos[wi]))
        
        r_pair = min_pair_val / 2.0
        r_wall = min_wall_val
        
        grad = np.zeros_like(pos)
        
        if r_pair <= r_wall:
            d = min_pair_val
            if d > 1e-12:
                vec = (pos[i] - pos[j]) / d
                grad[i] += 0.5 * vec
                grad[j] += -0.5 * vec
            return -r_pair, -grad.flatten()
        else:
            s = np.sign(pos[wi, wk])
            if s == 0:
                s = 1.0
            grad[wi, wk] = -s
            return -r_wall, -grad.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: Grid
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
    
    best_pos = make_grid(5, 4, 5)
    best_r = compute_min_dist(best_pos)
    
    rng = np.random.default_rng(42)
    
    # Initial optimization to establish a solid baseline
    # Start with a conservative radius
    initial_r = 0.1
    best_pos = _optimize_for_radius(best_pos, initial_r, max_iter=200, time_limit=10.0)
    best_r = compute_min_dist(best_pos)
    
    if best_r <= 0:
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = compute_min_dist(best_pos)

    current_r = best_r
    step_size = 0.0005
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.005, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.02, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            new_pos = _optimize_for_radius(start_pos, target_r, max_iter=300, time_limit=10.0)
            
            actual_r = compute_min_dist(new_pos)
            
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
                break
            elif actual_r > current_r:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
        
        if not success:
            consecutive_failures += 1

    # Final Polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish_hard_min(best_pos, max_iter=500)
        final_r = compute_min_dist(polished_pos)
        if final_r > best_r:
            best_r = final_r
            best_pos = polished_pos

    return best_pos
