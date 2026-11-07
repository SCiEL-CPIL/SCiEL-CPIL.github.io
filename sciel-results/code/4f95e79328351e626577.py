"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using weighted-sum scalarization with multi-start local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Use a set of fixed weight vectors to cover the objective space.
        # For each weight vector, perform a local search (Nelder-Mead style or simple gradient-free)
        # to find a point that minimizes the weighted sum.
        # This is a structured approach to generating diverse Pareto-optimal candidates.
        
        # Generate weight vectors. We want to cover the simplex of weights summing to 1.
        # For 3 objectives, we can use a grid on the 2D simplex.
        # Let n_grid be the resolution. Total weights = n_grid*(n_grid+1)/2.
        # We have 200 evaluations. Let's allocate 200 / n_weights evaluations per weight.
        # If n_weights = 20, we have 10 evals per local search. That's tight but doable for a simple local search.
        # If n_weights = 10, we have 20 evals per local search.
        
        # Let's try to generate weights that are well-distributed.
        # We'll use a set of random weights or a structured grid.
        # Given the small budget, a structured grid might be more reliable than random.
        
        # Let's define a function to generate weights on the simplex.
        def generate_weights(n_obj, n_samples):
            # Use Dirichlet distribution for random weights
            rng = np.random.default_rng(42)
            weights = rng.dirichlet(np.ones(n_obj), size=n_samples)
            return weights

        # How many weights to use?
        # If we use 10 weights, we have 20 evals each.
        # If we use 20 weights, we have 10 evals each.
        # A simple local search with 10-20 evals can make some progress.
        
        n_weights = 20
        weights = generate_weights(self.n_obj, n_weights)
        
        # For each weight, we will run a local search.
        # The local search will minimize the scalar function: sum(w_i * f_i(x))
        # We will use a simple coordinate descent or random perturbation method.
        # Since we don't have gradients, we'll use a random walk that accepts moves if the scalar objective improves.
        
        evals_per_weight = self.budget // n_weights
        
        # Pre-allocate results
        X = np.zeros((self.budget, self.dim))
        F = np.zeros((self.budget, self.n_obj))
        count = 0
        
        for w in weights:
            if count >= self.budget:
                break
            if time.time() - start_time > max_time:
                break
                
            # Initialize the local search with a random point
            rng_local = np.random.default_rng(hash(tuple(w)) % 2**32)
            x = lo + (hi - lo) * rng_local.random(self.dim)
            
            # Evaluate initial point
            f = problem(x)
            X[count] = x
            F[count] = f
            count += 1
            
            if count >= self.budget:
                break
            
            current_scalar = np.dot(w, f)
            
            # Local search loop
            for _ in range(evals_per_weight - 1):
                if count >= self.budget:
                    break
                if time.time() - start_time > max_time:
                    break
                    
                # Random perturbation
                step_size = 0.1 * (hi - lo)
                perturbation = rng_local.standard_normal(self.dim) * step_size
                x_new = np.clip(x + perturbation, lo, hi)
                
                # Evaluate
                f_new = problem(x_new)
                X[count] = x_new
                F[count] = f_new
                count += 1
                
                # Check if it improves the scalar objective
                scalar_new = np.dot(w, f_new)
                if scalar_new < current_scalar:
                    x = x_new
                    f = f_new
                    current_scalar = scalar_new
                # Else, keep the previous x but we've already recorded the new point.
                # This is a "record all" strategy, so even rejected points are kept.
                
        return X

# EVOLVE-BLOCK-END
