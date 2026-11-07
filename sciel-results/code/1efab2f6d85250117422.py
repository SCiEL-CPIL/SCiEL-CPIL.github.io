"""Multi-restart population-based optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart population-based search with adaptive step sizes.
    Strategy:
    1. Initial broad sampling to identify promising regions.
    2. Multiple restarts of a population-based local search (similar to (1+λ)-ES).
    3. Each restart focuses on a different region, balancing exploration and exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        budget = self.budget
        dim = self.dim
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(12345)

        # Phase 1: Initial exploration - sample diverse points
        n_initial = min(30, max(10, budget // 20))
        
        # Start at origin
        x0 = np.zeros(dim)
        evaluate(x0)
        
        # Sample around origin and random locations
        for _ in range(n_initial - 1):
            if check_deadline():
                break
            # Mix of local and global samples
            if rng.random() < 0.5:
                x = rng.uniform(-2, 2, dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if best_x is None:
            x = np.zeros(dim)
            evaluate(x)
            return best_x, best_f

        # Phase 2: Multi-restart population-based local search
        # We'll do several restarts, each with a small population
        n_restarts = 8
        pop_size = 6
        remaining = budget - calls_made
        
        if remaining <= 0:
            return best_x, best_f

        # Allocate budget across restarts
        # First restart gets more budget, subsequent ones get less
        restart_budgets = []
        for i in range(n_restarts):
            frac = 1.0 / (i + 2)  # Decaying allocation
            b = int(remaining * frac / sum(1.0 / (j + 2) for j in range(n_restarts)))
            restart_budgets.append(max(5, b))
        
        # Normalize to fit within remaining
        total_alloc = sum(restart_budgets)
        if total_alloc > remaining:
            scale = remaining / total_alloc
            restart_budgets = [int(b * scale) for b in restart_budgets]
        
        # Ensure we don't exceed budget
        while sum(restart_budgets) > remaining and any(b > 1 for b in restart_budgets):
            idx = np.argmax(restart_budgets)
            restart_budgets[idx] -= 1

        for r in range(n_restarts):
            if check_deadline():
                break
            
            rb = restart_budgets[r]
            if rb < 5:
                continue
            
            # Choose starting point for this restart
            if r == 0:
                # First restart: start from current best
                x_center = best_x.copy()
            else:
                # Subsequent restarts: perturb the best or sample fresh
                if rng.random() < 0.6:
                    # Perturb best with moderate noise
                    noise_scale = 1.0 + 0.5 * r
                    x_center = np.clip(best_x + noise_scale * rng.standard_normal(dim), -5, 5)
                else:
                    # Fresh random sample
                    x_center = rng.uniform(-4, 4, dim)
            
            # Initialize population around x_center
            sigma = 0.5
            pop = []
            for i in range(pop_size):
                if i == 0:
                    x = x_center
                else:
                    x = np.clip(x_center + sigma * rng.standard_normal(dim), -5, 5)
                f = evaluate(x)
                pop.append((f, x.copy()))
            
            # Sort population
            pop.sort(key=lambda t: t[0])
            
            # Local search loop
            for step in range(rb):
                if check_deadline():
                    break
                
                # Generate offspring from the best individuals
                offspring = []
                n_offspring = pop_size
                
                for _ in range(n_offspring):
                    # Pick a parent from the top half
                    parent_idx = int(rng.random() * (pop_size // 2 + 1))
                    if parent_idx >= pop_size:
                        parent_idx = 0
                    parent_x = pop[parent_idx][1]
                    
                    # Adapt sigma based on success rate
                    # Simple adaptation: keep sigma if we're making progress
                    y = np.clip(parent_x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    offspring.append((fy, y))
                
                # Combine population and offspring, keep best pop_size
                combined = pop + offspring
                combined.sort(key=lambda t: t[0])
                pop = combined[:pop_size]
                
                # Update global best if needed (already done in evaluate)
                
                # Adapt sigma
                # If the best offspring is better than the worst parent, increase sigma slightly
                # Otherwise, decrease
                best_offspring_f = min(f for f, _ in offspring)
                worst_parent_f = max(f for f, _ in pop[:pop_size//2])
                
                if best_offspring_f < worst_parent_f:
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                
                sigma = np.clip(sigma, 0.01, 3.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
