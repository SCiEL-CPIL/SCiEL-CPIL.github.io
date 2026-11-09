"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead Simplex Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        tol = 1e-12
        max_iter = 100

        def nelder_mead(x0, f0, remaining_budget):
            nonlocal calls_made, best_x, best_f
            
            # Initialize simplex
            # x0 is the center
            simplex = [x0]
            f_values = [f0]
            calls_made += 1
            
            # Generate other vertices
            for i in range(dim):
                xi = x0.copy()
                # Step size based on dimension and box size
                step = 0.5 if abs(xi[i]) < 4.5 else 0.1
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                f_values.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Check if we have enough budget for meaningful iterations
            if calls_made >= budget:
                return

            # Sort simplex
            for _ in range(max_iter):
                # Check time and budget
                if time.time() > deadline or calls_made >= budget:
                    break

                # Sort by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Convergence check
                if abs(f_values[0] - f_values[-1]) < tol:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            calls_made += 1
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                        if calls_made >= budget:
                            break

        # Multi-start strategy
        # We allocate budget to multiple restarts.
        # Griewank-Rosenbrock is multimodal. 
        # Let's try 5-10 restarts depending on dim.
        # With dim=5, Nelder-Mead is fast.
        # Each restart uses ~ (dim+1) + max_iter * (dim+1) calls roughly.
        # But we stop early on convergence or budget.
        
        # Initial random points
        num_starts = 10
        # Reserve some budget for each
        # Total budget 500. 10 starts -> 50 calls each.
        # This might be too few for NM to converge from scratch if far from optimum.
        # However, NM is local. 
        # Better: Use a coarse grid or random spread, then refine.
        
        # Strategy: 
        # 1. Evaluate 10 random points.
        # 2. Pick the best 3 as seeds for NM.
        # 3. Run NM on each seed with remaining budget.
        
        seeds = []
        seed_vals = []
        
        # Spread initial points
        for i in range(num_starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            seeds.append(x)
            seed_vals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if not seeds:
            return best_x, best_f

        # Sort seeds by value
        seed_order = np.argsort(seed_vals)
        
        # Take top 3 seeds for detailed optimization
        top_seeds = [seeds[i] for i in seed_order[:3]]
        top_vals = [seed_vals[i] for i in seed_order[:3]]
        
        # Run NM on each top seed
        # Distribute remaining budget
        remaining = budget - calls_made
        if remaining > 0:
            # Give each seed a portion of the budget
            # NM is efficient, so even small budgets help refine
            # Let's just run NM and let it stop when budget/time runs out
            for i in range(len(top_seeds)):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # We pass the current best as a potential reference? No, NM is local.
                # Just run NM from the seed.
                nelder_mead(top_seeds[i], top_vals[i], remaining)
                
        return best_x, best_f

# EVOLVE-BLOCK-END
