"""Multi-objective optimizer on WFG7: Tchebycheff decomposition with adaptive simplex."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate 30 uniformly distributed weight vectors on the simplex
        n_weights = 30
        weights = []
        # For 3 objectives, generate weights (w1, w2, w3) with w1+w2+w3=1, wi>=0
        # Use a grid-like approach for uniform coverage
        for i in range(n_weights + 1):
            w1 = i / n_weights
            for j in range(int((n_weights - i) * 0.999) + 1):
                w2 = j / n_weights
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-9:
                    w3 = max(0.0, w3)
                    weights.append([w1, w2, w3])
        # If we have more than n_weights, subsample; if less, add more
        if len(weights) > n_weights:
            indices = np.linspace(0, len(weights) - 1, n_weights).astype(int)
            weights = [weights[i] for i in indices]
        elif len(weights) < n_weights:
            # Add random weights to fill
            rng_fill = np.random.default_rng(123)
            while len(weights) < n_weights:
                w = rng_fill.dirichlet(np.ones(n_obj))
                weights.append(list(w))
        
        weights = np.array(weights, dtype=np.float64)
        n_sub = len(weights)

        # Evaluate budget per sub-problem
        # Reserve some budget for initial exploration
        initial_pop_size = min(50, budget // 4)
        per_sub_budget = max(2, (budget - initial_pop_size) // n_sub)

        rng = np.random.default_rng(42)

        # Initial population: random points
        X_init = lo + (hi - lo) * rng.random((initial_pop_size, dim))
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = initial_pop_size

        # Store all evaluated points for hypervolume calculation
        all_F = F_init.copy()
        all_X = X_init.copy()

        # For each sub-problem, run an adaptive simplex search
        # Tchebycheff scalarization: min max_i (w_i * (f_i(x) - z_i^*))
        # We use adaptive reference points based on current best

        # Initialize reference points (z^*) with the best found so far
        z_star = F_init.min(axis=0).copy()

        # For each weight vector, find a good starting point from initial population
        # by minimizing the Tchebycheff scalarization
        def tchebycheff(x, w, z):
            """Compute Tchebycheff scalarization value."""
            f = problem(x)
            return np.max(w * (f - z))

        # Assign initial points to sub-problems based on best scalarization value
        sub_best_x = np.zeros((n_sub, dim))
        sub_best_f = np.zeros((n_sub, n_obj))
        sub_best_scalar = np.full(n_sub, np.inf)

        # For each sub-problem, find the best initial point
        for s in range(n_sub):
            w = weights[s]
            best_val = np.inf
            best_x = X_init[0].copy()
            best_f = F_init[0].copy()
            for i in range(initial_pop_size):
                val = np.max(w * (F_init[i] - z_star))
                if val < best_val:
                    best_val = val
                    best_x = X_init[i].copy()
                    best_f = F_init[i].copy()
            sub_best_x[s] = best_x
            sub_best_f[s] = best_f
            sub_best_scalar[s] = best_val

        # Adaptive simplex (Nelder-Mead-like) for each sub-problem
        # We'll run multiple iterations, each time evaluating a new point
        # and updating the simplex

        # Simplex structure: n+1 points for n-dimensional problem
        # But we have limited budget, so we'll use a simplified approach
        # that tries different step sizes and directions

        # Phase 1: Local refinement around each sub-problem's best point
        # Use a simplified Nelder-Mead with adaptive step sizes

        simplex_size = dim + 1  # n+1 points
        # But this would require too many evaluations, so we'll use a 
        # coordinate-wise perturbation approach instead

        # Strategy: For each sub-problem, try perturbations in different directions
        # and keep the best one

        # Adaptive step sizes for each sub-problem
        step_sizes = np.ones(n_sub) * 0.1  # Initial step size (fraction of range)
        
        # Directions to try: coordinate axes + random directions
        n_directions = 2 * dim + 5  # 2*dim coordinate directions + 5 random
        
        # Pre-compute random directions
        rand_dirs = rng.standard_normal((5, dim))
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        
        # Main optimization loop
        # Alternate between sub-problems, trying to improve each one
        remaining_budget = budget - used
        
        # Track improvements for adaptive step sizing
        improvement_count = np.zeros(n_sub)
        no_improve_count = np.zeros(n_sub)
        
        # For each sub-problem, maintain a small local population
        # to enable differential-evolution-like moves within the local region
        local_pop_size = 3
        local_pops = []
        for s in range(n_sub):
            # Create local population around the best point
            local_x = np.zeros((local_pop_size, dim))
            local_f = np.zeros((local_pop_size, n_obj))
            local_x[0] = sub_best_x[s]
            local_f[0] = sub_best_f[s]
            for k in range(1, local_pop_size):
                perturbation = step_sizes[s] * half_range * rng.standard_normal(dim)
                new_x = np.clip(sub_best_x[s] + perturbation, lo, hi)
                new_f = problem(new_x)
                used += 1
                local_x[k] = new_x
                local_f[k] = new_f
                all_F = np.vstack([all_F, new_f.reshape(1, -1)])
                all_X = np.vstack([all_X, new_x.reshape(1, -1)])
            local_pops.append((local_x, local_f))
        
        # Main loop: iterate through sub-problems
        sub_idx = 0
        iteration = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            s = sub_idx % n_sub
            sub_idx += 1
            iteration += 1
            
            w = weights[s]
            local_x, local_f = local_pops[s]
            
            # Update reference point with current best
            # Use the best objective values found so far
            z_star = np.minimum(z_star, all_F.min(axis=0))
            
            # Compute scalarization values for local population
            scalars = np.array([np.max(w * (local_f[i] - z_star)) for i in range(local_pop_size)])
            best_local_idx = np.argmin(scalars)
            best_local_scalar = scalars[best_local_idx]
            
            # Try to generate a better point using DE-like crossover within local pop
            if local_pop_size >= 3:
                # DE/rand/1 within local population
                i1, i2 = rng.choice(local_pop_size - 1, size=2, replace=False)
                i1, i2 = i1 + 1, i2 + 1  # Avoid using the best point as base
                if i1 == best_local_idx:
                    i1 = (i1 + 1) % local_pop_size
                if i2 == best_local_idx:
                    i2 = (i2 + 1) % local_pop_size
                if i1 == i2:
                    i2 = (i2 + 1) % local_pop_size
                    
                # Crossover
                F_cr = 0.8  # Crossover rate
                CR = 0.5   # Crossover constant
                trial = local_x[i1].copy()
                for d in range(dim):
                    if rng.random() < CR or d == rng.integers(dim):
                        trial[d] = local_x[i1][d] + F_cr * (local_x[i2][d] - local_x[i1][d])
                
                # Also try coordinate-wise perturbation
                if rng.random() < 0.3:
                    d = rng.integers(dim)
                    direction = rng.choice([-1, 1])
                    trial[d] = sub_best_x[s][d] + direction * step_sizes[s] * half_range[d] * rng.uniform(0.5, 1.5)
                
                trial = np.clip(trial, lo, hi)
            else:
                # Fallback: coordinate-wise perturbation
                d = rng.integers(dim)
                direction = rng.choice([-1, 1])
                trial = sub_best_x[s].copy()
                trial[d] = sub_best_x[s][d] + direction * step_sizes[s] * half_range[d] * rng.uniform(0.5, 1.5)
                trial = np.clip(trial, lo, hi)
            
            # Evaluate trial point
            trial_f = problem(trial)
            used += 1
            all_F = np.vstack([all_F, trial_f.reshape(1, -1)])
            all_X = np.vstack([all_X, trial.reshape(1, -1)])
            
            # Check if trial improves the local best
            trial_scalar = np.max(w * (trial_f - z_star))
            
            if trial_scalar < best_local_scalar - 1e-10:
                # Improvement: replace worst in local pop
                worst_idx = np.argmax(scalars)
                local_x[worst_idx] = trial
                local_f[worst_idx] = trial_f
                scalars[worst_idx] = trial_scalar
                
                # Update sub-problem best
                new_best_idx = np.argmin(scalars)
                if scalars[new_best_idx] < sub_best_scalar[s]:
                    sub_best_x[s] = local_x[new_best_idx].copy()
                    sub_best_f[s] = local_f[new_best_idx].copy()
                    sub_best_scalar[s] = scalars[new_best_idx]
                
                # Increase step size on improvement
                step_sizes[s] = min(step_sizes[s] * 1.1, 0.5)
                improvement_count[s] += 1
                no_improve_count[s] = 0
            else:
                # No improvement: replace worst in local pop anyway to maintain diversity
                worst_idx = np.argmax(scalars)
                # Only replace if trial is not much worse
                if trial_scalar < scalars[worst_idx] + 0.1 * (scalars.max() - scalars.min() + 1e-10):
                    local_x[worst_idx] = trial
                    local_f[worst_idx] = trial_f
                    scalars[worst_idx] = trial_scalar
                
                # Decrease step size on stagnation
                no_improve_count[s] += 1
                if no_improve_count[s] > 5:
                    step_sizes[s] = max(step_sizes[s] * 0.9, 0.01)
                    no_improve_count[s] = 0
            
            # Periodically update local populations with global information
            if iteration % (n_sub * 3) == 0:
                # For each sub-problem, check if any global point is better
                for s2 in range(n_sub):
                    w2 = weights[s2]
                    # Find best global point for this sub-problem
                    global_scalars = np.array([np.max(w2 * (all_F[i] - z_star)) for i in range(len(all_F))])
                    best_global_idx = np.argmin(global_scalars)
                    if global_scalars[best_global_idx] < sub_best_scalar[s2] - 1e-10:
                        # Update sub-problem best from global
                        sub_best_x[s2] = all_X[best_global_idx].copy()
                        sub_best_f[s2] = all_F[best_global_idx].copy()
                        sub_best_scalar[s2] = global_scalars[best_global_idx]
                        # Reset local population around new best
                        local_x2, local_f2 = local_pops[s2]
                        local_x2[0] = sub_best_x[s2]
                        local_f2[0] = sub_best_f[s2]
                        for k in range(1, local_pop_size):
                            perturbation = step_sizes[s2] * half_range * rng.standard_normal(dim)
                            new_x = np.clip(sub_best_x[s2] + perturbation, lo, hi)
                            new_f = problem(new_x)
                            used += 1
                            if used >= budget:
                                break
                            local_x2[k] = new_x
                            local_f2[k] = new_f
                            all_F = np.vstack([all_F, new_f.reshape(1, -1)])
                            all_X = np.vstack([all_X, new_x.reshape(1, -1)])
                        local_pops[s2] = (local_x2, local_f2)
                
                if used >= budget:
                    break
        
        return all_X
# EVOLVE-BLOCK-END
