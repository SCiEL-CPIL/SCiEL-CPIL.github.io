"""(1+1)-ES with 1/5th success rule for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Pure (1+1)-Evolution Strategy with 1/5th success rule.
    This approach maintains an isotropic search distribution that implicitly
    adapts to the local curvature of the ill-conditioned Discus landscape
    without the overhead or stagnation risks of population-based methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize state
        # Start at the center of the box, which is a reasonable guess for the minimum
        # given the shift is in [-4, 4]^5.
        x = np.zeros(dim)
        f = func(x)
        evals = 1

        # Initial step size
        # Heuristic: start with a step size that covers a reasonable portion of the box
        sigma = 2.0

        # 1/5th success rule parameters
        # Target success rate is 1/5 = 0.2
        # Update rule: sigma_new = sigma * (1 + c * (success_rate - 0.2))
        # c controls the adaptation speed. c=1 is standard.
        c_sigma = 1.0

        # Window size for calculating success rate
        window_size = 10

        # Track successes in the current window
        successes = []

        best_x = x.copy()
        best_f = f

        while evals < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Sample offspring
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            fy = func(y)
            evals += 1

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Determine success
            success = fy < f
            if success:
                successes.append(1)
            else:
                successes.append(0)

            # Update step size if window is full
            if len(successes) >= window_size:
                success_rate = np.mean(successes)
                # 1/5th rule adaptation
                sigma *= (1.0 + c_sigma * (success_rate - 0.2))
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)
                successes = []

            # Move to offspring if successful
            if success:
                x = y
                f = fy
            else:
                # If not successful, we stay at x.
                # The 1/5th rule handles the step size reduction.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
