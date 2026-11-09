"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Nelder-Mead optimizer with robust initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def count_call(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            # Clip to bounds
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # 1. Initialization: Sample a few points to find a good start
        # We reserve some budget for the main optimization.
        # Let's use 10% of the budget for initialization, min 10 calls.
        init_budget = max(10, int(0.1 * self.budget))
        
        # Start with origin
        x0 = np.zeros(self.dim)
        f0 = count_call(x0)
        
        # Sample random points
        for _ in range(init_budget - 1):
            xi = rng.uniform(-5.0, 5.0, self.dim)
            count_call(xi)
            
        # If we ran out of budget during init (unlikely but possible if budget is tiny)
        if calls_made >= self.budget:
            return best_x, best_f

        # 2. Nelder-Mead Simplex Method
        # We will run Nelder-Mead starting from best_x
        # Simplex size: alpha = 0.5 * range? Or adaptive.
        # Standard Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        
        # Create initial simplex
        # Use best_x as the first vertex
        # Other vertices are best_x + delta * e_i
        # Choose delta based on function scale or just a fixed small step
        # Since bounds are [-5, 5], a step of 0.5 is reasonable.
        delta = 0.5
        
        simplex = []
        fvals = []
        
        # Vertex 0
        simplex.append(best_x.copy())
        fvals.append(best_f)
        
        # Vertices 1 to dim
        for i in range(self.dim):
            x_new = best_x.copy()
            x_new[i] += delta
            # Ensure it's within bounds for the initial setup to be valid
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = count_call(x_new)
            simplex.append(x_new)
            fvals.append(f_new)
            
            if calls_made >= self.budget:
                break

        # If we didn't form a full simplex, we can't proceed with standard NM
        # But we have best_x. We can try to continue with whatever we have or stop.
        # For dim=5, we need 6 vertices. Init used ~50 calls. Budget 500. So we have ~440 left.
        
        while calls_made < self.budget:
            # Sort simplex by fitness
            # We need to keep simplex and fvals sorted
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Check convergence: if range of fvals is small, stop
            if fvals[-1] - fvals[0] < 1e-8:
                break
                
            # Centroid of all points except the worst
            centroid = np.sum(simplex[:-1], axis=0) / self.dim
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = count_call(x_r)
            
            if calls_made >= self.budget:
                break
                
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = count_call(x_e)
                
                if calls_made >= self.budget:
                    break
                    
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = count_call(x_c)
                
                if calls_made >= self.budget:
                    break
                    
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = count_call(simplex[i])
                        
                        if calls_made >= self.budget:
                            break
                            
                    if calls_made >= self.budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
