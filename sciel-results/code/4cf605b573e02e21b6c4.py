"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist


def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)
    
    def objective_smooth(positions):
        """
        Smooth objective: use log-sum-exp to approximate the negative minimum radius.
        This makes the optimization landscape smoother for gradient-based methods.
        """
        positions = positions.reshape(-1, 3)
        
        # Compute all pairwise distances
        dists = pdist(positions, 'euclidean')
        if len(dists) == 0:
            return 0.0
        
        # Wall distances
        walls = 0.5 - np.abs(positions)
        
        # Combine all constraints: we want to maximize the minimum of all these values
        # Use a soft-min approximation: -logsumexp(-x) ≈ min(x) for large beta
        beta = 50.0  # sharpness parameter
        
        # Pairwise constraint values (we want these to be large)
        pair_vals = dists / 2.0
        # Wall constraint values
        wall_vals = walls.flatten()
        
        # Combine all
        all_vals = np.concatenate([pair_vals, wall_vals])
        
        # Soft minimum: -logsumexp(-beta * x) / beta
        neg_beta_x = -beta * all_vals
        max_val = np.max(neg_beta_x)
        log_sum = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min = -log_sum / beta
        
        return -soft_min
    
    def objective_direct(positions):
        """Direct objective: negative of the actual minimum radius."""
        return -compute_min_radius(positions)
    
    def make_initial_grid():
        """Create a reasonable initial grid configuration."""
        m = int(np.ceil(N ** (1.0 / 3.0)))
        # Create m x m x m grid, take first N points
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def make_bcc_like():
        """Create a body-centered cubic inspired arrangement."""
        # Start with a grid and add perturbations
        positions = make_initial_grid()
        # Add small random perturbations to break symmetry
        np.random.seed(123)
        positions += np.random.uniform(-0.02, 0.02, size=positions.shape)
        return positions
    
    def make_random():
        """Create random initial positions."""
        return np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    def local_optimize(positions, method='L-BFGS-B', maxiter=300, ftol=1e-12):
        """Run local optimization on positions."""
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(objective_direct, positions.flatten(), method=method,
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol})
            pos_opt = res.x.reshape(-1, 3)
            # Clip to ensure within bounds
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except:
            return positions
    
    def differential_evolve_optimize():
        """Use differential evolution for global search."""
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def de_objective(x):
            return -compute_min_radius(x.reshape(-1, 3))
        
        try:
            res = differential_evolution(de_objective, bounds, seed=42, maxiter=100,
                                         tol=1e-10, polish=True, popsize=15)
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except:
            return None
    
    best_positions = None
    best_radius = -1.0
    
    np.random.seed(42)
    
    # Strategy 1: Differential evolution for global search
    print("Running differential evolution...")
    de_pos = differential_evolve_optimize()
    if de_pos is not None:
        de_pos = local_optimize(de_pos, maxiter=200)
        r = compute_min_radius(de_pos)
        print(f"DE result: radius = {r:.10f}")
        if r > best_radius:
            best_radius = r
            best_positions = de_pos
    
    # Strategy 2: Multiple local optimizations from different starts
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_like),
        ("rand1", lambda: make_random()),
        ("rand2", lambda: make_random()),
        ("rand3", lambda: make_random()),
    ]
    
    for name, init_func in initializations:
        positions = init_func()
        # First optimize with smooth objective to get into a good basin
        try:
            res_smooth = minimize(objective_smooth, positions.flatten(), method='L-BFGS-B',
                                  bounds=[(-0.5, 0.5)] * (N * 3),
                                  options={'maxiter': 100, 'ftol': 1e-10})
            pos_smooth = res_smooth.x.reshape(-1, 3)
        except:
            pos_smooth = positions
        
        # Then refine with direct objective
        pos_opt = local_optimize(pos_smooth, maxiter=300, ftol=1e-12)
        r = compute_min_radius(pos_opt)
        print(f"{name} result: radius = {r:.10f}")
        
        if r > best_radius:
            best_radius = r
            best_positions = pos_opt
    
    # Strategy 3: Perturb best solution and re-optimize
    if best_positions is not None:
        for perturb_round in range(3):
            np.random.seed(100 + perturb_round)
            perturbed = best_positions + np.random.normal(0, 0.01, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            pos_opt = local_optimize(perturbed, maxiter=200)
            r = compute_min_radius(pos_opt)
            print(f"perturb{perturb_round} result: radius = {r:.10f}")
            if r > best_radius:
                best_radius = r
                best_positions = pos_opt
    
    # Final clip
    best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    print(f"Final best radius: {best_radius:.10f}")
    
    return best_positions
# EVOLVE-BLOCK-END
