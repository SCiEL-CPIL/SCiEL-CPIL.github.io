"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(c):
        """Compute the minimum radius for a given set of centres."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(c)):
            return 0.0
        # Check bounds
        if np.any(np.abs(c) > 0.5 + 1e-12):
            return 0.0
        
        # Compute pairwise distances
        # Using broadcasting for speed
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective(c):
        """
        Objective function to minimize.
        We minimize a sum of repulsive potentials to maximize the minimum distance.
        We also add a penalty if points go out of bounds (though bounds are handled by optimizer,
        it's safer to have a soft penalty or rely on hard bounds).
        Using L-BFGS-B with hard bounds is efficient.
        """
        c = np.asarray(c, dtype=np.float64).reshape(N, 3)
        
        # Repulsive potential: sum of 1/d^2
        # This forces points apart.
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        # Avoid division by zero
        dists_sq = np.where(dists_sq < 1e-12, 1e-12, dists_sq)
        
        # Only sum upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        repulsive_energy = np.sum(1.0 / dists_sq[iu])
        
        # Boundary penalty (optional if using hard bounds, but helps gradient)
        # We want to keep points away from walls if possible, but the hard constraint is |c| <= 0.5
        # Actually, maximizing radius means we WANT points to touch walls if that's the bottleneck.
        # So we should NOT penalize being near walls in the objective, just constrain them.
        # The repulsive term will push them apart. If the optimal packing is wall-limited,
        # the points will naturally move to the walls due to repulsion from the bulk.
        
        return repulsive_energy

    def generate_bcc_init(seed):
        """
        Generate an initial configuration based on a BCC lattice.
        BCC has points at (0,0,0) and (0.5, 0.5, 0.5) in a unit cell.
        We create a superlattice and pick the N points closest to the center of the cube.
        """
        np.random.seed(seed)
        
        # Create a grid of BCC points
        # We need enough points to choose from. 50 points in a cube of side 1.
        # Volume per point ~ 1/50 = 0.02. Side ~ 0.27.
        # Let's generate a 5x5x5 grid of BCC cells. Total points 5*5*5*2 = 250.
        grid_size = 5
        a = 1.0 / grid_size  # Lattice constant
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Corner of the cell
                    x0 = (i + 0.5) * a - 0.5
                    y0 = (j + 0.5) * a - 0.5
                    z0 = (k + 0.5) * a - 0.5
                    
                    # BCC points: corner and body center
                    # Relative to cell center? No, standard BCC is (0,0,0) and (a/2, a/2, a/2)
                    # Let's define cell from (x0, y0, z0) to (x0+a, y0+a, z0+a)
                    # Center of cell: (x0 + a/2, y0 + a/2, z0 + a/2)
                    # BCC points in a cell of size a:
                    # 1. (0,0,0) relative to cell corner -> (x0, y0, z0)
                    # 2. (a/2, a/2, a/2) relative to cell corner -> (x0 + a/2, y0 + a/2, z0 + a/2)
                    
                    points.append([x0, y0, z0])
                    points.append([x0 + a/2, y0 + a/2, z0 + a/2])
        
        points = np.array(points)
        
        # Filter points that are strictly inside or on the boundary [-0.5, 0.5]
        # The grid generation above centers the grid in [-0.5, 0.5]
        # Let's verify bounds
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        # If we don't have enough points, expand grid
        if len(points) < N:
             # Fallback to random if BCC generation fails (unlikely with 5x5x5)
             return np.random.uniform(-0.5, 0.5, size=(N, 3))
        
        # Select the N points closest to the center (0,0,0)
        # This creates a dense cluster in the middle, which is a good starting point
        # for packing because it minimizes the distance to the walls for the outermost points
        # relative to the volume.
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        init_points = points[indices].copy()
        
        # Add small random perturbation to break symmetry and help optimization
        init_points += np.random.normal(0, 0.01, size=init_points.shape)
        
        # Clip to bounds
        init_points = np.clip(init_points, -0.5, 0.5)
        
        return init_points

    def generate_fcc_init(seed):
        """
        Generate an initial configuration based on an FCC lattice.
        FCC has points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in a unit cell.
        """
        np.random.seed(seed + 1000)
        grid_size = 4
        a = 1.0 / grid_size
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x0 = (i + 0.5) * a - 0.5
                    y0 = (j + 0.5) * a - 0.5
                    z0 = (k + 0.5) * a - 0.5
                    
                    # FCC points
                    points.append([x0, y0, z0])
                    points.append([x0, y0 + a/2, z0 + a/2])
                    points.append([x0 + a/2, y0, z0 + a/2])
                    points.append([x0 + a/2, y0 + a/2, z0])
        
        points = np.array(points)
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            return np.random.uniform(-0.5, 0.5, size=(N, 3))
            
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        init_points = points[indices].copy()
        
        init_points += np.random.normal(0, 0.01, size=init_points.shape)
        init_points = np.clip(init_points, -0.5, 0.5)
        
        return init_points

    best_c = None
    best_r = -np.inf

    # List of initializers
    initializers = [
        lambda s: generate_bcc_init(s),
        lambda s: generate_fcc_init(s),
        lambda s: np.random.uniform(-0.5, 0.5, size=(N, 3)),
    ]
    
    bounds = [(-0.5, 0.5)] * (N * 3)

    # Optimization Loop
    # We will try a few restarts with different seeds and methods
    for seed in range(10):
        if time.time() > deadline - 20.0:
            break
            
        # Select initializer
        init_func = initializers[seed % len(initializers)]
        c_init = init_func(seed)
        
        # Phase 1: Coarse Optimization
        # Use L-BFGS-B for efficient handling of bounds and smooth gradients
        try:
            res1 = minimize(
                objective,
                c_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_curr = res1.x.reshape(N, 3)
            r_curr = compute_radius(c_curr)
            
            if r_curr > best_r:
                best_r = r_curr
                best_c = c_curr.copy()
                
        except Exception:
            continue
            
        # Phase 2: Fine-tuning with Nelder-Mead
        # Nelder-Mead can be better at navigating non-smooth parts or polishing
        if time.time() < deadline - 10.0:
            try:
                res2 = minimize(
                    objective,
                    c_curr.flatten(),
                    method='Nelder-Mead',
                    bounds=bounds, # Nelder-Mead doesn't strictly support bounds in all versions, 
                                   # but recent SciPy does. If not, we rely on clipping.
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                # Ensure result is within bounds
                c_curr2 = np.clip(res2.x.reshape(N, 3), -0.5, 0.5)
                r_curr2 = compute_radius(c_curr2)
                
                if r_curr2 > best_r:
                    best_r = r_curr2
                    best_c = c_curr2.copy()
                    
            except Exception:
                continue

    # Final Polish: Coordinate Descent or small perturbations
    if best_c is not None and time.time() < deadline - 2.0:
        # Try to improve the best found solution further
        c_final = best_c.copy()
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            # Small random perturbation
            c_pert = c_final + np.random.normal(0, 0.005, size=(N, 3))
            c_pert = np.clip(c_pert, -0.5, 0.5)
            
            try:
                res3 = minimize(
                    objective,
                    c_pert.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                c_new = np.clip(res3.x.reshape(N, 3), -0.5, 0.5)
                r_new = compute_radius(c_new)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                    c_final = c_new.copy()
            except Exception:
                continue

    # Fallback if optimization failed completely
    if best_c is None:
        best_c = np.random.uniform(-0.5, 0.5, size=(N, 3))
        best_r = compute_radius(best_c)
        
    return best_c
# EVOLVE-BLOCK-END
