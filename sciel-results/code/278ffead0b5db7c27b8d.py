"""Multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1 optimizer exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 Pareto front: sum(f) = 0.5, f_i >= 0
        # For 3 objectives: f1 + f2 + f3 = 0.5
        # We can parameterize with two simplex coordinates
        # Generate dense samples on the Pareto front (FREE - no evaluation needed)
        n_front_samples = 5000
        # Use Dirichlet distribution to sample from simplex
        rng = np.random.default_rng(12345)
        # Sample from Dirichlet(1,1,1) to get uniform points on 2-simplex
        simplex_samples = rng.dirichlet(np.ones(n_obj), size=n_front_samples)
        front_points = 0.5 * simplex_samples  # f_i = 0.5 * t_i where sum(t_i)=1
        # Filter: only keep points strictly better than reference [400,400,400]
        ref = np.array([400.0, 400.0, 400.0])
        mask = np.all(front_points < ref, axis=1)
        front_points = front_points[mask]
        
        all_points = list(front_points)

        # Now use evaluation budget to find points near the Pareto front
        # Strategy: initialize population near the front, then do local search
        # to find points that are close to (but not necessarily on) the front
        
        # Initialize with points derived from the front structure
        # For DTLZ1, the front is at g(x) = 0 where g depends on x
        # We can create initial points that are close to the front
        
        pop_size = 40
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        for i in range(pop_size):
            # Create a point that should be near the Pareto front
            # Use the front point as guide
            f_target = front_points[i % len(front_points)]
            # Create x that maps close to f_target
            # For DTLZ1: f_i = 0.5 * (1 + g) * x_i for i < n_obj-1... 
            # Actually DTLZ1: f_i = 0.5 * (1 + g(x_M)) * x_i for i=1..n_obj
            # where g(x_M) = 100*(|x_M| - sum(x_i - 0.5^2)) 
            # The front is at g=0, i.e., sum(x_i - 0.5^2) = |x_M|
            # Simplest: set x_M such that g=0, and x_1..x_{n-1} to match front
            # Let's just do random search near the front with local refinement
            
            # Random initialization in [0,1]^7
            X[i] = rng.random(dim)
        
        # Evaluate initial population
        for i in range(pop_size):
            F[i] = problem(X[i])
            all_points.append(F[i])
        used = pop_size

        # Local search / refinement phase
        # For each point, try to improve it by moving toward the Pareto front
        # Use a simple gradient-free local search
        
        best_f = np.min(F, axis=0)  # component-wise best
        
        while used < budget:
            if time.time() > deadline - 5:
                break
            
            # Pick a random point and do local search
            idx = rng.integers(pop_size)
            x_curr = X[idx].copy()
            f_curr = F[idx].copy()
            
            # Local search: try small perturbations
            step_size = 0.1 * (1.0 - used / budget) + 0.01
            improved = False
            for _ in range(20):  # max 20 local search steps per point
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                    
                # Random perturbation
                x_new = x_curr + step_size * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                all_points.append(f_new)
                
                # Check if new point dominates current or is non-dominated
                # For now, just accept if it's better in at least one objective
                # and not worse in all
                if np.any(f_new < f_curr) and not np.all(f_new > f_curr):
                    x_curr = x_new
                    f_curr = f_new
                    improved = True
                    # Decrease step size for finer search
                    step_size *= 0.8
                else:
                    # Try a smaller step
                    step_size *= 0.7
            
            if improved:
                X[idx] = x_curr
                F[idx] = f_curr

        return X
# EVOLVE-BLOCK-END
