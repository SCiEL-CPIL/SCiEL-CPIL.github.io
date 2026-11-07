"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with sparse global lattice initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Sparse global lattice search to find a good starting point
        # Use a grid with stride 2.0 over [-5, 5] -> 6 points per dimension
        # That's 6^5 = 7776 points, too many. Use stride 2.5 -> 5 points per dim -> 3125
        # Still too many. Use stride 3.0 -> 4 points per dim -> 1024
        # Or use a smarter approach: evaluate at center and corners first, then refine
        
        # Let's use a two-phase approach:
        # Phase 1: Evaluate at a coarse grid (stride ~2.5) to find basin
        # Phase 2: Run Nelder-Mead from the best point found
        
        # Coarse grid: 4 points per dimension from -4 to 4 (stride ~2.67)
        # Actually, let's be more strategic. The minimizer is in [-4, 4]^5.
        # Use 3 points per dim: -4, 0, 4 -> 3^5 = 243 evaluations
        # Then refine around the best with Nelder-Mead
        
        # Generate coarse grid
        grid_points = np.array(np.meshgrid(
            np.linspace(-4, 4, 3),
            np.linspace(-4, 4, 3),
            np.linspace(-4, 4, 3),
            np.linspace(-4, 4, 3),
            np.linspace(-4, 4, 3)
        )).T.reshape(-1, dim)
        
        # Shuffle to diversify early progress
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]
        
        # Evaluate coarse grid
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for i in range(min(len(grid_points), budget // 4)):
            x = grid_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead simplex method from the best point
        # Initialize simplex around best_x
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Create initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x
        for i in range(1, dim + 1):
            simplex[i] = best_x.copy()
            simplex[i, i - 1] += 0.5  # step size
        
        # Evaluate simplex
        simplex_f = np.zeros(dim + 1)
        simplex_f[0] = best_f
        calls_made += 1  # already counted best_x
        
        for i in range(1, dim + 1):
            if calls_made >= budget:
                break
            simplex_f[i] = func(simplex[i])
            calls_made += 1
            if simplex_f[i] < best_f:
                best_f = simplex_f[i]
                best_x = simplex[i].copy()
        
        # Main Nelder-Mead loop
        while calls_made < budget:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            # Centroid of all but worst point
            centroid = simplex[:-1].mean(axis=0)
            
            # Reflection
            if calls_made >= budget:
                break
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            
            if f_r < simplex_f[0]:
                # Expansion
                if calls_made >= budget:
                    break
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if calls_made >= budget:
                    break
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                
                if f_c < f_r:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls_made >= budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_made += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
