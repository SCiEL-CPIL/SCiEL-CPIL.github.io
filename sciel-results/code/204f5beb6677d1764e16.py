"""Population-based CMA-ES-like optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based optimizer using CMA-ES-like strategy:
    - Maintains a population of candidate points
    - Adapts step size based on selection success rate
    - Uses mean-shift toward better solutions
    - Multi-restart with adaptive population size
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # Initial exploration: sample diverse points
        n_initial = min(20, max_calls // 10)
        for i in range(n_initial):
            if check_deadline():
                break
            if i == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4.5, 4.5, self.dim)
            evaluate(x)

        if best_x is None:
            x = np.zeros(self.dim)
            evaluate(x)
            return best_x, best_f

        # CMA-ES-like population search
        def cma_es_search(x_mean, f_mean, budget_left, pop_size=None, sigma_init=1.0):
            nonlocal calls_made, best_x, best_f
            if pop_size is None:
                pop_size = max(4, min(10, budget_left // 10))
            
            x = x_mean.copy()
            sigma = sigma_init
            lam = pop_size  # population size
            mu = lam // 2   # number of parents
            c_sigma = 1.0 / (5 + np.log(lam))  # adaptation rate
            
            # Track success rate for sigma adaptation
            success_count = 0
            window = 10
            
            for gen in range(budget_left // lam + 1):
                if check_deadline():
                    break
                
                # Generate offspring
                offspring = []
                for i in range(lam):
                    if check_deadline():
                        break
                    step = sigma * rng.standard_normal(self.dim)
                    y = x + step
                    y, fy = evaluate(y)
                    offspring.append((fy, y))
                
                if len(offspring) == 0:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update mean: weighted average of top mu
                weights = np.exp(-np.arange(mu) / (mu / 2))
                weights /= weights.sum()
                new_x = np.zeros(self.dim)
                for i in range(min(mu, len(offspring))):
                    new_x += weights[i] * offspring[i][1]
                
                # Adapt sigma based on success
                success_rate = success_count / window if window > 0 else 0.5
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.3:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.01, 5.0)
                
                # Update success tracking
                if offspring[0][0] < f_mean:
                    success_count += 1
                else:
                    success_count -= 0.5
                success_count = max(0, min(window, success_count))
                
                f_mean = offspring[0][0]
                x = new_x

        # Main loop: run CMA-ES with restarts
        remaining = max_calls - calls_made
        if remaining <= 0:
            return best_x, best_f

        # First CMA-ES run from best point
        cma_es_search(best_x, best_f, remaining, sigma_init=1.0)
        remaining = max_calls - calls_made

        # If we have budget left, try restarts from perturbed best
        restarts = 3
        for r in range(restarts):
            if remaining <= 20 or check_deadline():
                break
            sigma_perturb = 1.5 * (r + 1)
            x_start = np.clip(best_x + sigma_perturb * rng.standard_normal(self.dim), -5, 5)
            _, f_start = evaluate(x_start)
            remaining = max_calls - calls_made
            if remaining <= 10:
                break
            cma_es_search(x_start, f_start, remaining, sigma_init=0.8)
            remaining = max_calls - calls_made

        # Final polish with small sigma
        if remaining > 5 and not check_deadline():
            cma_es_search(best_x, best_f, remaining, sigma_init=0.1)

        return best_x, best_f

# EVOLVE-BLOCK-END
