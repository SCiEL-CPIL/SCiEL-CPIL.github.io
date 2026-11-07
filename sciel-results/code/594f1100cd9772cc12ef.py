"""Optimizer for BBOB-style f17 (Schaffers F7) using Multi-Fidelity SMC."""
import numpy as np
import time
import math

class Optimizer:
    """
    Multi-fidelity Sequential Monte Carlo (SMC) optimizer.
    
    This approach frames the minimization as a Bayesian inference problem.
    We maintain a population of particles (candidate solutions) that represent
    the posterior distribution of the optimum. In each iteration, we:
    1. Evaluate a subset of particles (the 'high fidelity' evaluations).
    2. Compute weights based on the objective values.
    3. Resample to concentrate particles in good regions.
    4. Perturb the resampled particles to explore.
    
    This differs from GP-BO by not relying on a single surrogate model for
    prediction, but rather on the empirical distribution of the population.
    It is robust to the hidden shift/rotation because the population adapts
    its covariance implicitly through the selection and perturbation steps.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # SMC Parameters
        n_particles = 50  # Population size
        # We will evaluate a subset of particles each step to save budget
        # or evaluate all if budget allows. 
        # With 500 budget and 50 particles, we can do ~10 iterations of full evaluation.
        # Or we can do many iterations with partial evaluation.
        # Let's start with full evaluation of the population in each SMC step.
        # 500 / 50 = 10 steps. This might be too few for convergence.
        # Let's use a smaller population or partial updates.
        
        # Alternative: Smaller population, more steps.
        # Let's try n_particles = 20. 500/20 = 25 steps.
        # Or n_particles = 10. 500/10 = 50 steps.
        # Let's stick with a moderate population size for stability.
        n_particles = 30
        
        # Initial particles: Uniform random in the box
        # To improve early AOCC, let's use a stratified initialization (Halton/Latin Hypercube)
        # for the first batch, then SMC takes over.
        
        # Generate initial particles using Latin Hypercube Sampling (LHS) for better spread
        def latin_hypercube_sample(n, d):
            samples = np.zeros((n, d))
            for i in range(d):
                # Random permutation of intervals
                perm = np.random.permutation(n)
                samples[:, i] = (perm + np.random.rand(n)) / n
            # Scale to [-5, 5]
            samples = samples * 10.0 - 5.0
            return samples

        X = latin_hypercube_sample(n_particles, dim)
        
        # We need to track the best value found so far for AOCC calculation
        # and for the SMC weighting.
        best_x = X[0].copy()
        best_f = float('inf')
        y = np.full(n_particles, np.inf)
        
        count = 0
        
        # Evaluate initial population
        for i in range(n_particles):
            if time.time() > deadline:
                break
            try:
                f = func(X[i])
                count += 1
                y[i] = f
                if f < best_f:
                    best_f = f
                    best_x = X[i].copy()
            except Exception:
                # If evaluation fails, fill remaining with inf and break
                y[i:] = np.inf
                break

        if count >= budget:
            return best_x, best_f

        # SMC Loop
        # We will run SMC iterations until budget or time is exhausted.
        # In each iteration:
        # 1. Weights are proportional to exp(-beta * y). We use a soft-max with temperature.
        # 2. Resample particles according to weights.
        # 3. Perturb resampled particles with adaptive noise.
        # 4. Evaluate the new population.
        
        # Temperature (beta) for weighting. Start low (exploratory), increase later?
        # Or keep it constant? 
        # Let's use a constant beta that is tuned to the scale of the function.
        # Since f_opt is in [-100, 100], differences can be large.
        # Let's use a robust scaling: beta = 1 / (std(y) + eps)
        
        # Perturbation scale
        # Start with a large scale, reduce it as we converge?
        # Or use the covariance of the top-k particles?
        # Let's use an adaptive scale based on the spread of the current population.
        
        # To avoid getting stuck in local minima (Schaffers has many),
        # we need a balance between exploitation and exploration.
        # SMC with resampling can suffer from particle degeneracy.
        # We add a "mutation" step to maintain diversity.
        
        n_smc_steps = 0
        max_smc_steps = (budget - count) // n_particles + 5 # Safety cap
        
        while count < budget and n_smc_steps < max_smc_steps:
            if time.time() > deadline:
                break
            
            # 1. Compute weights
            # Use a softmax with temperature. 
            # If all y are inf (shouldn't happen), fallback to uniform.
            valid_y = y[np.isfinite(y)]
            if len(valid_y) == 0:
                break
            
            # Scale y to [0, 1] or similar for stable exp
            y_min = np.min(valid_y)
            y_max = np.max(valid_y)
            y_range = y_max - y_min
            if y_range < 1e-8:
                # All values are similar, use uniform weights
                weights = np.ones(n_particles) / n_particles
            else:
                # Softmax: exp(-(y - y_min) / (y_range * T))
                # T is temperature. T=1 means range is the scale.
                # We want to favor the best but not ignore others completely.
                T = 0.5 # Lower T = more exploitative
                scaled_y = (y - y_min) / (y_range * T)
                # Clip to avoid overflow/underflow
                scaled_y = np.clip(scaled_y, 0, 100)
                weights = np.exp(-scaled_y)
                weights /= np.sum(weights)
            
            # 2. Resample
            # Systematic resampling
            u0 = np.random.rand() / n_particles
            u = u0 + np.arange(n_particles) / n_particles
            cdf = np.cumsum(weights)
            cdf = cdf / cdf[-1] # Ensure sums to 1
            indices = np.searchsorted(cdf, u)
            indices = np.clip(indices, 0, n_particles - 1)
            
            X_new = X[indices].copy()
            
            # 3. Perturb
            # Adaptive noise: use the standard deviation of the top 20% particles
            # to estimate the local landscape scale.
            k_top = max(1, int(0.2 * n_particles))
            top_indices = np.argsort(y)[:k_top]
            if len(top_indices) > 1:
                # Estimate covariance of top particles
                # For simplicity, use diagonal covariance (independent dimensions)
                stds = np.std(X[top_indices], axis=0)
                # Floor the standard deviation to avoid collapse
                stds = np.maximum(stds, 0.1)
            else:
                stds = np.ones(dim) * 0.5
            
            # Add noise
            # Scale the noise by a factor that decreases over time?
            # Let's keep it constant relative to the local scale.
            noise = np.random.normal(0, 1, size=(n_particles, dim)) * stds[np.newaxis, :]
            X_new += noise
            X_new = np.clip(X_new, -5, 5)
            
            # 4. Evaluate
            for i in range(n_particles):
                if count >= budget or time.time() > deadline:
                    break
                try:
                    f = func(X_new[i])
                    count += 1
                    y[i] = f
                    X[i] = X_new[i]
                    if f < best_f:
                        best_f = f
                        best_x = X_new[i].copy()
                except Exception:
                    # If evaluation fails, stop
                    y[i:] = np.inf
                    break
            
            n_smc_steps += 1
        
        return best_x, best_f
