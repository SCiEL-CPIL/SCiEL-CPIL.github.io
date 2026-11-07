"""Optimizer for BBOB-style f17 (Schaffers F7) using Multi-Fidelity SMC."""
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Multi-fidelity Sequential Monte Carlo sampler for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # SMC parameters
        n_particles = 40
        n_levels = 15
        n_particles_per_level = max(1, budget // n_levels)
        # We will actually just run until budget is exhausted, adjusting level transitions based on time/budget
        # Let's stick to a simpler budget allocation:
        # Split budget into n_levels chunks.
        # At each level, we resample and mutate.
        
        # However, SMC usually requires a sequence of target distributions.
        # Here, the target is proportional to exp(-beta * f(x)).
        # We increase beta over time.
        
        # Initial particles: Halton sequence for good coverage
        particles = self._generate_halton(n_particles, dim)
        
        # Evaluate initial particles
        values = np.array([func(p) for p in particles])
        
        best_x = particles[np.argmin(values)].copy()
        best_f = values[np.argmin(values)]
        
        count = n_particles
        
        # SMC loop
        # We define levels by increasing beta.
        # beta_t = beta_0 * (beta_max / beta_0)^(t / (n_levels - 1))
        # Let's use a fixed number of levels based on remaining budget.
        # Or simpler: just iterate.
        
        # To make it "multi-fidelity", we can use a lightweight surrogate (GP) 
        # to guide the mutation or selection, but with 500 calls, GP overhead might be high.
        # Let's use a simple adaptive step size based on the spread of particles.
        
        # We will perform n_levels = 10 iterations of "Resample + Mutate".
        # Each iteration uses a chunk of the budget.
        
        n_iterations = 10
        budget_per_iter = max(1, (budget - count) // n_iterations)
        
        # Weights
        # Initial weights equal
        weights = np.ones(n_particles) / n_particles
        
        # Beta schedule
        betas = np.logspace(0, 4, n_iterations + 1)  # From 1 to 10000
        
        for level in range(n_iterations):
            if time.time() > deadline:
                break
            if count >= budget:
                break
            
            # 1. Resample particles based on weights
            # Use systematic resampling
            particles = self._resample(particles, weights)
            
            # 2. Mutate particles
            # Adaptive step size: based on the standard deviation of the particles
            # Or based on the current best f?
            # Let's use the spread of the current population.
            if len(particles) > 1:
                # Distance from centroid
                centroid = np.mean(particles, axis=0)
                dists = np.linalg.norm(particles - centroid, axis=1)
                step_size = np.median(dists) * 0.5
            else:
                step_size = 1.0
            
            # Ensure step size is reasonable
            step_size = np.clip(step_size, 0.1, 2.0)
            
            # Generate new particles by adding noise
            # We need to evaluate new particles.
            # To maintain the population size, we replace some particles.
            # Let's replace half the population with mutated versions of the best ones?
            # Or just mutate all and keep the best n_particles?
            
            # Strategy: Mutate all, evaluate, then select best n_particles.
            # This is effectively a (mu + lambda) selection.
            
            new_particles = []
            new_values = []
            
            # We have budget_per_iter calls available.
            # Let's generate that many new candidates.
            # To keep diversity, we mutate a mix of good and random particles.
            
            # Sort current particles by value (approximate, using previous values)
            # We don't have current values for resampled particles directly, 
            # but we can keep track. Let's keep a list of (particle, value).
            
            # Actually, the resampling above lost the values. 
            # Let's restructure: keep a list of (particle, value).
            
            # Let's restart the loop logic with explicit value tracking.
            
            # This is getting complicated. Let's simplify to a standard SMC-PDE or similar.
            # 
            # Revised Plan:
            # 1. Maintain a population of (x, f(x)).
            # 2. At each step:
            #    a. Compute weights w_i proportional to exp(-beta * f_i).
            #    b. Resample indices according to w.
            #    c. Mutate the resampled points to get new candidates.
            #    d. Evaluate new candidates.
            #    e. Combine old and new, select best N.
            #    f. Increase beta.
            
            # Let's implement this robustly.
            
            pass # Placeholder for re-implementation in the actual code below

        # The above loop sketch is messy. I will write the clean implementation now.
        
        # Clean Implementation:
        
        # Re-initialize state if needed (though we already did above)
        # Let's restart the logic cleanly.
        
        particles = self._generate_halton(n_particles, dim)
        values = np.array([func(p) for p in particles])
        count = n_particles
        best_x = particles[np.argmin(values)].copy()
        best_f = values[np.argmin(values)]
        
        n_iterations = 15
        # Distribute remaining budget
        remaining = budget - count
        calls_per_iter = max(1, remaining // n_iterations)
        
        # Beta schedule
        betas = np.logspace(0, 3, n_iterations + 1)
        
        for level in range(n_iterations):
            if time.time() > deadline:
                break
            if count >= budget:
                break
            
            # 1. Compute weights
            # Normalize values to avoid overflow in exp
            min_val = np.min(values)
            # If min_val is very negative, exp(-beta * (v - min_val)) is safe.
            # We want weight proportional to exp(-beta * f).
            # Let's use relative to min_val to keep numbers stable.
            # w_i = exp(-beta * (f_i - min_val))
            # This makes the best particle have weight 1.
            
            beta = betas[level]
            scaled_vals = values - min_val
            # Clip to prevent underflow issues if beta is large and diff is large
            # exp(-beta * scaled)
            log_weights = -beta * scaled_vals
            # Shift log weights for numerical stability
            max_log_w = np.max(log_weights)
            log_weights -= max_log_w
            weights = np.exp(log_weights)
            sum_w = np.sum(weights)
            if sum_w == 0:
                weights = np.ones(n_particles) / n_particles
            else:
                weights /= sum_w
            
            # 2. Resample
            # Systematic resampling
            u = np.random.uniform(0, 1 / n_particles)
            indices = np.zeros(n_particles, dtype=int)
            cdf = np.cumsum(weights)
            cdf[-1] = 1.0
            for i in range(n_particles):
                u_i = u + i / n_particles
                # Find index where cdf >= u_i
                indices[i] = np.searchsorted(cdf, u_i, side='right')
                if indices[i] >= n_particles:
                    indices[i] = n_particles - 1
            
            resampled_particles = particles[indices]
            resampled_values = values[indices]
            
            # 3. Mutate
            # Adaptive step size
            if n_particles > 1:
                # Use spread of resampled particles
                centroid = np.mean(resampled_particles, axis=0)
                dists = np.linalg.norm(resampled_particles - centroid, axis=1)
                step_size = np.median(dists) * 0.5
            else:
                step_size = 1.0
            
            step_size = np.clip(step_size, 0.05, 1.5)
            
            # Generate new particles
            # We will generate calls_per_iter new particles.
            # To ensure diversity, we mutate a subset of the resampled particles.
            # Let's just mutate random resampled particles.
            
            new_particles = []
            new_values = []
            
            for _ in range(calls_per_iter):
                if count >= budget:
                    break
                # Pick a random resampled particle to mutate
                idx = np.random.randint(n_particles)
                parent = resampled_particles[idx]
                
                # Gaussian mutation
                noise = np.random.randn(dim) * step_size
                child = parent + noise
                child = np.clip(child, -5, 5)
                
                new_particles.append(child)
                
                # Evaluate
                try:
                    f_val = func(child)
                    count += 1
                    new_values.append(f_val)
                    
                    if f_val < best_f:
                        best_f = f_val
                        best_x = child.copy()
                except Exception:
                    break
            
            if not new_particles:
                continue
                
            new_particles = np.array(new_particles)
            new_values = np.array(new_values)
            
            # 4. Combine and Select
            # Combine old (resampled) and new
            all_particles = np.vstack([resampled_particles, new_particles])
            all_values = np.concatenate([resampled_values, new_values])
            
            # Select best n_particles
            order = np.argsort(all_values)
            selected_indices = order[:n_particles]
            
            particles = all_particles[selected_indices]
            values = all_values[selected_indices]
            
        return best_x, best_f

    def _generate_halton(self, n, dim):
        """Generate Halton sequence points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                # Base is d+2
                base = d + 2
                f = 1.0
                result = 0.0
                i_val = i + 1  # 1-indexed for Halton
                while i_val > 0:
                    i_val, rem = divmod(i_val, base)
                    f /= base
                    result += rem * f
                # Map [0, 1] to [-5, 5]
                points[i, d] = result * 10 - 5
        return points

    def _resample(self, particles, weights):
        """Systematic resampling."""
        n = len(particles)
        u = np.random.uniform(0, 1.0 / n)
        indices = np.zeros(n, dtype=int)
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0
        for i in range(n):
            u_i = u + i / n
            indices[i] = np.searchsorted(cdf, u_i, side='right')
            if indices[i] >= n:
                indices[i] = n - 1
        return particles[indices]
