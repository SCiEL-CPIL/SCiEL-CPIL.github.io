"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Single-trajectory SA with adaptive covariance for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # SA Parameters
        # Initial temperature high enough to jump over Rastrigin barriers
        T = 10.0
        T_min = 1e-6
        cooling_rate = 0.995
        
        # Adaptive Covariance Parameters
        # We maintain a diagonal covariance matrix initially, but can adapt it.
        # For simplicity and robustness, we use a diagonal adaptation based on recent steps.
        # This mimics the anisotropic step distribution requested.
        sigma = np.ones(dim) * 2.0  # Initial standard deviation per dimension
        sigma_min = 1e-5
        sigma_max = 5.0
        
        # History for adaptation
        recent_successes = []
        history_len = 20

        # Multi-start logic
        max_restarts = 5
        restarts = 0
        stagnation_counter = 0
        stagnation_limit = 100 # Calls without improvement

        # Initialize current state
        x = rng.uniform(-4.5, 4.5, dim)
        f = func(x)
        calls += 1
        best_f = f
        best_x = x.copy()

        while calls < budget and time.time() - start_time < max_time:
            # Generate candidate step
            # Use the current sigma vector for anisotropic sampling
            step = rng.standard_normal(dim) * sigma
            x_new = x + step
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1

            # Simulated Annealing Acceptance
            if f_new < f:
                # Always accept improvement
                x = x_new
                f = f_new
                recent_successes.append(1)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                # Accept with probability exp(-delta_E / T)
                delta_E = f_new - f
                if delta_E < 100: # Avoid overflow
                    prob = np.exp(-delta_E / T)
                    if rng.random() < prob:
                        x = x_new
                        f = f_new
                        recent_successes.append(0)
                        stagnation_counter += 1
                    else:
                        recent_successes.append(0)
                        stagnation_counter += 1
                else:
                    recent_successes.append(0)
                    stagnation_counter += 1

            # Keep history bounded
            if len(recent_successes) > history_len:
                recent_successes.pop(0)

            # Adapt sigma based on success rate in recent history
            if len(recent_successes) >= 10:
                success_rate = np.mean(recent_successes)
                # If success rate is high, we are moving in good directions, maybe expand?
                # If low, we are stuck or moving too much, shrink?
                # Standard 1/5 rule: if success ~ 0.2, keep size.
                # If success > 0.5, expand. If < 0.1, shrink.
                for i in range(dim):
                    # Per-dimension adaptation is complex without CMA, so we do global scaling
                    # but applied to the vector.
                    pass 
                
                # Global scaling based on average success
                if success_rate > 0.3:
                    sigma *= 1.05
                elif success_rate < 0.1:
                    sigma *= 0.95
                
                sigma = np.clip(sigma, sigma_min, sigma_max)

            # Cooling
            T = max(T * cooling_rate, T_min)

            # Restart Logic
            # If we are stagnant for a long time, or if T is very low and we are stuck,
            # try a new start or a large jump.
            if stagnation_counter > stagnation_limit:
                restarts += 1
                if restarts > max_restarts:
                    # If we've restarted too many times, just do local polish
                    break
                
                # Reset temperature
                T = 5.0
                # Reset sigma to moderate
                sigma = np.ones(dim) * 2.0
                recent_successes = []
                stagnation_counter = 0
                
                # Jump to a new random point or perturb best
                if rng.random() < 0.5:
                    x = rng.uniform(-4.5, 4.5, dim)
                else:
                    x = best_x + rng.normal(0, 2.0, dim)
                    x = np.clip(x, -5, 5)
                
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # Final polish with local search if budget remains
        if calls < budget and time.time() - start_time < max_time:
            # Simple Nelder-Mead or (1+1) ES polish
            # We'll use a simple (1+1) ES with small sigma
            x = best_x.copy()
            f = best_f
            sigma_local = 0.5
            for _ in range(100):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                step = rng.standard_normal(dim) * sigma_local
                x_new = x + step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < f:
                    x = x_new
                    f = f_new
                    sigma_local = min(sigma_local * 1.1, 2.0)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma_local = max(sigma_local * 0.9, 1e-4)

        return best_x, best_f
# EVOLVE-BLOCK-END
