"""Optimized multi-objective optimizer for WFG7 using MOEA/D with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        span = hi - lo
        
        # Number of sub-problems (population size)
        # 30 is a good balance for 200 budget
        n_sub = 30
        
        # Generate weight vectors uniformly on the simplex
        # Using Dirichlet with alpha=1 gives uniform distribution
        # Use global np.random to respect evaluator's seed
        rng = np.random
        weights = rng.dirichlet(np.ones(n_obj), size=n_sub)
        
        # Initialize population
        # Start with random points to ensure diversity
        pop_x = lo + span * rng.random((n_sub, dim))
        pop_f = np.zeros((n_sub, n_obj))
        
        eval_count = 0
        all_x = []
        all_f = []
        
        # Ideal point (min) and Nadir point (max) for normalization
        ideal = np.full(n_obj, np.inf)
        nadir = np.full(n_obj, 0.0)
        
        def update_bounds(f):
            nonlocal ideal, nadir
            ideal = np.minimum(ideal, f)
            nadir = np.maximum(nadir, f)
            
        def evaluate(x):
            nonlocal eval_count, ideal, nadir
            if eval_count >= budget:
                return None
            f = problem(x)
            f = np.array(f)
            eval_count += 1
            update_bounds(f)
            all_x.append(x.copy())
            all_f.append(f)
            return f

        # Initial evaluations
        for i in range(n_sub):
            f = evaluate(pop_x[i])
            if f is None:
                break
            pop_f[i] = f
            
        if eval_count >= budget:
            return np.array(all_x)

        # Tchebycheff scalarization function
        # g(x, w) = max_i { w_i * |f_i(x) - z_i^*| }
        # We use z^* as the current ideal point.
        # To avoid division by zero if ideal == nadir, we add a small epsilon.
        
        def tchebycheff(f, w):
            # Normalize by range (nadir - ideal) to make weights meaningful
            # If range is 0, the objective is constant, so contribution is 0
            range_ = nadir - ideal
            range_ = np.where(range_ > 1e-10, range_, 1.0)
            dist = np.abs(f - ideal) / range_
            return np.max(w * dist)

        # Main Loop: Evolution
        # We will use a simple mutation strategy:
        # Pick a random individual i, mutate it using a random direction or neighbor j.
        # If the mutant is better in Tchebycheff sense for weight w_i, replace.
        
        while eval_count < budget:
            i = rng.randint(0, n_sub)
            w = weights[i]
            
            # Mutation: Combine current with a random neighbor or random jump
            # 50% chance: local perturbation
            # 50% chance: crossover with a random neighbor
            if rng.random() < 0.5:
                # Local perturbation
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction) + 1e-10
                # Adaptive step size: start small, maybe larger if stuck? 
                # For simplicity, fixed small step relative to span
                step = 0.1 * span
                x_new = pop_x[i] + direction * step
            else:
                # Crossover with random neighbor
                j = rng.randint(0, n_sub)
                # Blend
                alpha = rng.random()
                x_new = alpha * pop_x[i] + (1 - alpha) * pop_x[j]
                # Add some noise
                x_new += rng.standard_normal(dim) * 0.05 * span

            x_new = np.clip(x_new, lo, hi)
            
            f_new = evaluate(x_new)
            if f_new is None:
                break
            
            # Check improvement using Tchebycheff scalarization for weight w
            # We compare against the current individual i
            # Note: We should ideally check against neighbors too, but for speed, 
            # checking against self is a valid "local search" step.
            # To be more robust, we can check if it dominates the current point 
            # OR has a better Tchebycheff value.
            
            current_f = pop_f[i]
            
            # Calculate Tchebycheff values
            # Note: ideal/nadir are updated with f_new already
            # We need to be careful: if f_new improved ideal, the scalarization 
            # for the old point changes. 
            # Standard MOEA/D updates the archive and then checks.
            # Here, we just check if f_new is "better" than current_f 
            # in the direction of w.
            
            # Re-calculate scalarization for both using the *current* global ideal/nadir
            # This ensures consistency.
            scalar_new = tchebycheff(f_new, w)
            scalar_old = tchebycheff(current_f, w)
            
            # Also check dominance: if f_new dominates current_f, it's always better
            dominates = np.all(f_new <= current_f) and np.any(f_new < current_f)
            
            if dominates or scalar_new < scalar_old:
                pop_x[i] = x_new
                pop_f[i] = f_new

        # Final Processing: Extract Non-Dominated Set
        if not all_f:
            return np.array([mid])
            
        all_f = np.array(all_f)
        all_x = np.array(all_x)
        
        n_points = len(all_f)
        if n_points == 0:
            return np.array([])
            
        # Vectorized Non-Dominated Sorting
        # A point i is dominated if there exists j such that f_j <= f_i and f_j != f_i
        # We can compute this matrix efficiently
        
        # Create a matrix of comparisons
        # D[i, j] is True if j dominates i
        # f_j <= f_i for all objectives AND f_j < f_i for at least one objective
        
        # f_j <= f_i  ->  (all_f[j] <= all_f[i])
        # We can broadcast:
        # less_eq = (all_f[:, np.newaxis, :] <= all_f[np.newaxis, :, :]) # Shape (N, N, M)
        # This is O(N^2 * M). N=200, M=3. 200*200*3 = 120,000 ops. Very fast.
        
        less_eq = (all_f[np.newaxis, :, :] <= all_f[:, np.newaxis, :])
        # less_eq[i, j] is True if all_f[j] <= all_f[i] element-wise
        
        # strict_less = (all_f[np.newaxis, :, :] < all_f[:, np.newaxis, :])
        # any_strict = np.any(strict_less, axis=2)
        
        # Dominated if: (j <= i) AND (j < i in at least one dim)
        # We can compute:
        # all_leq = np.all(less_eq, axis=2) # Shape (N, N)
        # all_leq[i, j] is True if f_j <= f_i
        
        # We need j to be strictly better in at least one objective.
        # strict = (all_f[np.newaxis, :, :] < all_f[:, np.newaxis, :])
        # any_strict = np.any(strict, axis=2)
        
        # dominated[i] = exists j != i such that all_leq[i, j] and any_strict[i, j]
        
        # Let's do it in two steps to save memory if N is large, but N=200 is small.
        all_leq = np.all(less_eq, axis=2)
        
        # Exclude self-comparison (i == j)
        # If i == j, all_leq[i, i] is True, but strict is False. So it's safe.
        
        # Check for strict inequality
        # We can reuse less_eq logic or just compute strict directly
        strict = (all_f[np.newaxis, :, :] < all_f[:, np.newaxis, :])
        any_strict = np.any(strict, axis=2)
        
        # Dominated condition:
        # dominated[i] = any(all_leq[i, :] & any_strict[i, :])
        dominated = np.any(all_leq & any_strict, axis=1)
        
        non_dominated_idx = np.where(~dominated)[0]
        
        return all_x[non_dominated_idx]

# EVOLVE-BLOCK-END
