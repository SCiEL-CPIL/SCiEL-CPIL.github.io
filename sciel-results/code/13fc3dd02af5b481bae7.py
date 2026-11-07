"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(12345)
        
        # Initial population of 5 points for better exploration
        pop_size = 5
        x = np.zeros(self.dim)
        best_x = x.copy()
        best_f = np.inf
        calls = 0

        # Evaluate initial points
        for i in range(pop_size):
            if time.time() > deadline:
                break
            xi = rng.uniform(-5, 5, self.dim)
            fi = func(xi)
            calls += 1
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
            if i == 0:
                x = xi.copy()
                fx = fi
            else:
                # Keep a small population implicitly by tracking best
                pass

        # Main optimization loop using a simple (1+1) with restarts
        # But to be stronger, let's use a more robust approach:
        # CMA-ES is ideal but complex. Let's try a robust (1+1)-ES with
        # adaptive sigma and occasional large jumps (restarts).
        
        # Actually, for Rosenbrock, a key issue is the narrow valley.
        # A pure (1+1) can get stuck. Let's use a population-based approach
        # with a small population (e.g., 5) and selection.
        
        # Let's implement a simple evolution strategy with population size 5
        # and rank selection.
        
        pop_size = 5
        pop = np.array([best_x] + [rng.uniform(-5, 5, self.dim) for _ in range(pop_size - 1)])
        fvals = np.array([best_f] + [np.inf] * (pop_size - 1))
        
        # Evaluate the rest of the initial population
        for i in range(1, pop_size):
            if time.time() > deadline:
                break
            fvals[i] = func(pop[i])
            calls += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = pop[i].copy()

        sigma = 1.0
        success_count = 0
        total_steps = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Sample from current best with some noise
                # Or sample from a Gaussian centered at the current mean?
                # For simplicity, let's use the current best as the center
                # and generate variations.
                # A better approach: use the mean of the top half of the population.
                
                # Let's use a simple (lambda+mu)-ES strategy
                # Center: best_x
                # Step: sigma
                offspring[i] = best_x + sigma * rng.standard_normal(self.dim)
                offspring[i] = np.clip(offspring[i], -5, 5)
            
            # Evaluate offspring
            off_fvals = np.zeros(pop_size)
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                off_fvals[i] = func(offspring[i])
                calls += 1
                if off_fvals[i] < best_f:
                    best_f = off_fvals[i]
                    best_x = offspring[i].copy()
            
            # Selection: keep the best `pop_size` from current pop + offspring
            # Combine
            combined_x = np.vstack([pop, offspring])
            combined_f = np.concatenate([fvals, off_fvals])
            
            # Sort by fitness
            idx = np.argsort(combined_f)
            pop = combined_x[idx[:pop_size]]
            fvals = combined_f[idx[:pop_size]]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
            
            # Adapt sigma based on success
            # If the best offspring was better than the previous best, increase sigma
            # Otherwise, decrease it.
            # Simple rule: if best_f improved in this step, sigma *= 1.2, else sigma *= 0.9
            # But we need to track if the *best* improved.
            
            # Let's use a simpler adaptation:
            # If the best value in the new population is better than the previous best,
            # we can slightly increase sigma to explore more.
            # Otherwise, decrease it to exploit.
            
            # For Rosenbrock, we might want to be cautious with large steps.
            
            # Let's use a standard CMA-ES-like adaptation if possible, but that's complex.
            # Let's stick to a simple (1+1)-ES on the best point, but with a population
            # to escape local minima? No, that's not standard.
            
            # Let's revert to a more robust (1+1)-ES with adaptive sigma and restarts.
            # The population approach above is a bit ad-hoc.
            
            # Let's try a different strategy:
            # 1. Start with a random point.
            # 2. Use (1+1)-ES with adaptive sigma.
            # 3. If no improvement for N steps, do a large jump (restart).
            
            # This is simpler and often works well for BBOB.
            
            break # Exit this loop to use the simpler (1+1)-ES with restarts below

        # --- Simpler and more robust approach: (1+1)-ES with adaptive sigma and restarts ---
        
        # Reset state for the new strategy
        calls = 0
        best_x = rng.uniform(-5, 5, self.dim)
        best_f = func(best_x)
        calls += 1
        
        sigma = 1.0
        stagnation = 0
        max_stagnation = 50
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = best_x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < best_f:
                best_f = fy
                best_x = y
                stagnation = 0
                # Increase step size on success
                sigma *= 1.1
                # Cap sigma to avoid huge jumps
                if sigma > 5.0:
                    sigma = 5.0
            else:
                stagnation += 1
                # Decrease step size on failure
                sigma *= 0.95
                # Floor sigma to avoid getting stuck in a tiny region
                if sigma < 0.01:
                    sigma = 0.01
                
            # Restart if stuck for too long
            if stagnation > max_stagnation:
                # Try a few random restarts
                for _ in range(3):
                    if calls >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x_new = rng.uniform(-5, 5, self.dim)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new
                        sigma = 1.0
                        stagnation = 0
                        break
                # If no restart worked, reduce sigma further or just continue
                # Let's reset stagnation to avoid infinite restarts
                stagnation = 0
                sigma = max(sigma * 0.5, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
