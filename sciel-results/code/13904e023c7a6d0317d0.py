"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Initialize at center
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial step size
        sigma = 2.0

        # 1/5 success rule parameters
        # rho = 1/5 success rate
        # We track a counter for successes in the last N steps
        # N = 15 is common for 1/5 rule
        N = 15
        successes = 0

        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            z = rng.standard_normal(n)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)

            # Evaluate
            fx_trial = func(x_trial)
            calls += 1

            # Update best
            if fx_trial < best_f:
                best_f = fx_trial
                best_x = x_trial.copy()

            # Update step size based on 1/5 rule
            if fx_trial < fx:
                successes += 1
                x = x_trial
                fx = fx_trial

            # Adapt sigma every N steps
            if (calls - 1) % N == 0 and calls > 1:
                # If success rate is high, increase sigma; else decrease
                # Standard 1/5 rule: sigma = sigma * (1 + 0.33 * (successes/N - 0.2))
                # Or simpler: if successes > N/5, increase, else decrease
                # Let's use the multiplicative update:
                # sigma = sigma * (1 + 0.33 * (successes / N - 0.2))
                # But we need to reset successes counter
                if successes > 0:
                    # Increase sigma
                    sigma *= (1.0 + 0.33 * (successes / N - 0.2))
                else:
                    # Decrease sigma
                    sigma *= (1.0 + 0.33 * (0.0 / N - 0.2))
                
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Reset counter
                successes = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
