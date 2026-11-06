import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Finds the centers of 50 equal spheres in a unit cube to maximize the minimum radius.
    Uses a repulsive force optimization (L-BFGS-B) with multiple restarts.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Helper to calculate the radius of a configuration
    def calc_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        # min_k (0.5 - |c_ik|)
        # This is equivalent to 0.5 - max_k |c_ik|
        max_coord_abs = np.max(np.abs(c), axis=1)
        min_boundary_dist = 0.5 - np.max(max_coord_abs)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # Energy function and gradient
    def energy_and_grad(x):
        # x is flat array of shape (N*3,)
        c = x.reshape(N, 3)
        
        # Pairwise distances squared
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        # We only care about i < j
        # Create a mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # For gradient calculation, we need the full matrix but only sum over i<j
        # Energy term: sum_{i<j} 1 / dist_sq_ij
        # To be safe, add a small epsilon to dist_sq
        dist_sq_safe = np.maximum(dist_sq, 1e-12)
        
        # Energy
        e_pair = np.sum(1.0 / dist_sq_safe[mask])
        
        # Boundary term
        # sum_{i,k} 1 / (0.5 - |c_ik|)^2
        abs_c = np.abs(c)
        dist_bound = 0.5 - abs_c
        dist_bound_safe = np.maximum(dist_bound, 1e-6)
        e_bound = np.sum(1.0 / (dist_bound_safe**2))
        
        total_energy = e_pair + e_bound
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Gradient of pair term
        # d/dc_ik (1 / ||c_i - c_j||^2) = -2 * (c_ik - c_jk) / ||c_i - c_j||^4
        # We sum over all j != i.
        # Let's compute the full matrix gradient contribution
        # G_ij = -2 * diff_ij / dist_sq_safe_ij^2
        # grad_i = sum_{j != i} G_ij
        # grad_j = sum_{i != j} G_ji = - sum_{i != j} G_ij
        
        # Efficient calculation:
        # diff is (N, N, 3)
        # dist_sq_safe is (N, N)
        # inv_dist_sq_4 = 1 / dist_sq_safe^2
        inv_dist_sq_4 = 1.0 / (dist_sq_safe**2)
        
        # Multiply diff by inv_dist_sq_4
        # diff * inv_dist_sq_4[:,:,None]
        weighted_diff = diff * inv_dist_sq_4[:, :, np.newaxis]
        
        # Sum over j for each i (axis=1)
        # grad_pair[i] = sum_j -2 * weighted_diff[i,j]
        grad_pair = -2.0 * np.sum(weighted_diff, axis=1)
        
        # The diagonal terms (i==j) should be 0. 
        # dist_sq_safe[i,i] is 1e-12, so inv is huge.
        # diff[i,i] is 0. So 0 * huge = 0 (or nan if 0*inf).
        # To be safe, zero out diagonal in weighted_diff before summing?
        # Actually, diff[i,i] is exactly 0. 0 * anything is 0.
        # But 1e-12 ** -2 is 1e24. 0 * 1e24 is 0.
        # However, if there's numerical noise, it might be non-zero.
        # Let's explicitly zero the diagonal of weighted_diff.
        np.fill_diagonal(weighted_diff[:, :, 0], 0) # This doesn't work for 3D easily
        # Better:
        # weighted_diff = np.where(np.eye(N, dtype=bool)[:, :, np.newaxis], 0, weighted_diff)
        
        # Let's redo the gradient calculation more carefully to avoid diagonal issues
        # We only want i != j
        # Create a mask
        mask_3d = mask[:, :, np.newaxis]
        weighted_diff_clean = np.where(mask_3d, weighted_diff, 0)
        
        grad_pair = -2.0 * np.sum(weighted_diff_clean, axis=1)
        
        # Gradient of boundary term
        # d/dc_ik (1 / (0.5 - |c_ik|)^2) = 2 * sign(c_ik) / (0.5 - |c_ik|)^3
        # Using safe dist
        grad_bound = 2.0 * np.sign(c) / (dist_bound_safe**3)
        # If c_ik is 0, sign is 0, so gradient is 0. This is correct by symmetry.
        
        grad = grad_pair + grad_bound
        
        return total_energy, grad.flatten()

    # Objective function for scipy
    def objective(x):
        val, grad = energy_and_grad(x)
        return val, grad

    # Initialization strategies
    def init_grid_checkerboard():
        # 5x5x4 grid
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.linspace(-0.375, 0.375, 4)
        
        points = []
        for z in zs:
            for y in ys:
                for x in xs:
                    # Checkerboard: index sum even
                    # Map coords to indices
                    # x index: 0..4, y: 0..4, z: 0..3
                    # We can just generate all and filter
                    points.append((x, y, z))
        
        points = np.array(points)
        # Generate indices
        i_idx, j_idx, k_idx = np.indices((5, 5, 4))
        # Flatten
        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        k_flat = k_idx.flatten()
        
        # Checkerboard condition: (i + j + k) % 2 == 0
        mask = (i_flat + j_flat + k_flat) % 2 == 0
        checker_points = points[mask]
        
        # We need 50 points. checker_points has ~52.
        # Shuffle to pick 50? Or just take first 50?
        # Taking first 50 might be biased.
        # Let's shuffle with a fixed seed for this init
        idx = np.arange(len(checker_points))
        np.random.seed(42)
        np.random.shuffle(idx)
        selected = checker_points[idx[:N]]
        
        # Add small random noise to break symmetry
        noise = np.random.normal(0, 0.01, selected.shape)
        return selected + noise

    def init_random():
        return np.random.uniform(-0.45, 0.45, (N, 3))

    def init_bcc_subset():
        # BCC lattice points
        # Simple cubic grid with spacing 0.2
        # Add body center
        # This is complex to subset exactly to 50.
        # Just use random for other restarts.
        return None

    restarts = [
        init_grid_checkerboard,
        init_random,
        init_random,
        init_random,
        init_grid_checkerboard, # Different noise? No, same seed. 
        # Let's vary the noise seed
    ]
    
    # To vary the grid init, we can pass a seed
    def init_grid_with_seed(seed):
        def inner():
            xs = np.linspace(-0.4, 0.4, 5)
            ys = np.linspace(-0.4, 0.4, 5)
            zs = np.linspace(-0.375, 0.375, 4)
            points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
            i_idx, j_idx, k_idx = np.indices((5, 5, 4))
            i_flat = i_idx.flatten()
            j_flat = j_idx.flatten()
            k_flat = k_idx.flatten()
            mask = (i_flat + j_flat + k_flat) % 2 == 0
            checker_points = points[mask]
            idx = np.arange(len(checker_points))
            np.random.seed(seed)
            np.random.shuffle(idx)
            selected = checker_points[idx[:N]]
            noise = np.random.normal(0, 0.01, selected.shape)
            return selected + noise
        return inner

    restart_funcs = [
        init_grid_with_seed(1),
        init_grid_with_seed(2),
        init_grid_with_seed(3),
        init_random,
        init_random,
        init_random,
        init_grid_with_seed(4),
        init_grid_with_seed(5),
    ]

    bounds = [(-0.5, 0.5)] * (N * 3)

    for i, init_func in enumerate(restart_funcs):
        if time.time() - start_time > deadline:
            break
            
        x0 = init_func()
        
        # Ensure x0 is within bounds
        x0 = np.clip(x0, -0.49, 0.49)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e10: # Check for reasonable result
                c_res = res.x.reshape(N, 3)
                r_res = calc_radius(c_res)
                
                if r_res > best_radius:
                    best_radius = r_res
                    best_centers = c_res
        except Exception as e:
            pass

    # If no valid center found (should not happen), return a default grid
    if best_centers is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
