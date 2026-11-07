"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        
        # PSO Parameters
        # Population size: small to conserve budget, but enough for diversity
        # Budget 500, Dim 5.
        # If pop size is 10, we can do 50 generations.
        # If pop size is 5, we can do 100 generations.
        # Let's try a moderate pop size to balance exploration and exploitation.
        pop_size = 10
        
        # Check if budget allows for at least one generation
        if total_budget < pop_size:
            # Just do random search
            best_x = None
            best_f = np.inf
            for _ in range(total_budget):
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # Initialize particles
        # Positions: random in [-5, 5]
        positions = rng.uniform(-5, 5, (pop_size, n))
        # Velocities: small random initial velocities
        # Scale velocity to cover the domain in ~10 steps
        # Domain size is 10, so step size ~1
        velocities = rng.uniform(-0.5, 0.5, (pop_size, n))
        
        # Evaluate initial population
        fitnesses = np.array([func(p) for p in positions])
        calls_made = pop_size
        
        # Initialize personal bests
        pbests = positions.copy()
        pbests_f = fitnesses.copy()
        
        # Initialize global best
        global_idx = np.argmin(fitnesses)
        gbest = positions[global_idx].copy()
        gbest_f = fitnesses[global_idx]
        
        # PSO Coefficients
        w = 0.7       # Inertia weight
        c1 = 1.5      # Cognitive coefficient
        c2 = 1.5      # Social coefficient
        
        # Loop for remaining budget
        while total_budget - calls_made >= pop_size:
            # Update velocities
            # v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
            r1 = rng.uniform(0, 1, (pop_size, n))
            r2 = rng.uniform(0, 1, (pop_size, n))
            
            velocities = (w * velocities + 
                          c1 * r1 * (pbests - positions) + 
                          c2 * r2 * (gbest - positions))
            
            # Update positions
            new_positions = positions + velocities
            new_positions = np.clip(new_positions, -5, 5)
            
            # Evaluate new positions
            new_fitnesses = np.array([func(p) for p in new_positions])
            calls_made += pop_size
            
            # Update personal bests
            improved_mask = new_fitnesses < pbests_f
            pbests[improved_mask] = new_positions[improved_mask]
            pbests_f[improved_mask] = new_fitnesses[improved_mask]
            
            # Update global best
            new_global_idx = np.argmin(new_fitnesses)
            if new_fitnesses[new_global_idx] < gbest_f:
                gbest_f = new_fitnesses[new_global_idx]
                gbest = new_positions[new_global_idx].copy()
            
            # Update current positions for next iteration
            positions = new_positions
            
            # Optional: Diversity maintenance
            # If the swarm is stuck (all positions very close), add noise
            # Calculate spread
            spread = np.std(positions, axis=0)
            if np.mean(spread) < 0.1:
                # Inject diversity into worst particles
                worst_idx = np.argsort(new_fitnesses)[-pop_size//2:]
                positions[worst_idx] = rng.uniform(-5, 5, (len(worst_idx), n))
                # We don't evaluate these immediately to save budget, 
                # but their velocities will be reset implicitly or we can reset them
                velocities[worst_idx] = rng.uniform(-0.5, 0.5, (len(worst_idx), n))
                # Note: The next iteration's evaluation will handle these new positions.
                # However, pbests for these particles should probably be reset or kept?
                # Keeping pbests is safer to not lose the best found so far.
                # But their current position is now random.
        
        # If we have some remaining budget, use it for local refinement around gbest
        remaining = total_budget - calls_made
        if remaining > 0:
            # Simple local search: try small perturbations around gbest
            step_size = 0.1
            for _ in range(remaining):
                # Random direction
                direction = rng.normal(0, 1, n)
                direction /= (np.linalg.norm(direction) + 1e-8)
                x_new = gbest + step_size * direction
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                if fx_new < gbest_f:
                    gbest_f = fx_new
                    gbest = x_new.copy()
                    # Reduce step size as we get closer
                    step_size *= 0.9

        return gbest, gbest_f
# EVOLVE-BLOCK-END
