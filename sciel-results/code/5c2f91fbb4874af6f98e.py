"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with step-size adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Count calls
        call_count = [0]

        def counted_func(x):
            call_count[0] += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Initial point: center of the box is a reasonable start
        # Since minimizer is in [-4, 4]^5, center 0 is good
        x = np.zeros(dim)
        fx = counted_func(x)
        best_x = x.copy()
        best_fx = fx

        # Step size adaptation parameters
        sigma = 3.0  # initial step size
        # 1/5 success rule: target success rate ~ 1/5
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        c_sigma = 0.5
        d_sigma = 1 + 0.5*c_sigma + 2/max(1e-10, (dim+2)/2 - 1)

        # Path accumulators for CSA (Cumulative Step-size Adaptation)
        p_sigma = np.zeros(dim)
        h_sigma = 0.0

        # Success/failure tracking for 1/5 rule
        successes = []

        for iteration in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            z = np.random.standard_normal(dim)
            y = x + sigma * z

            # Clip to box
            y = np.clip(y, -5, 5)
            fy = counted_func(y)

            # Check deadline again after function call
            if time.time() > deadline:
                break

            # Update path for CSA
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (y - x) / (sigma + 1e-30)
            norm_p_sigma = np.linalg.norm(p_sigma)

            # Success?
            success = fy < fx
            if success:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = y.copy(), fy
                successes.append(1)
            else:
                successes.append(0)

            # Keep only last 10 successes for 1/5 rule
            if len(successes) > 10:
                successes.pop(0)

            # Adapt sigma using 1/5 success rule
            if len(successes) >= 5:
                success_rate = sum(successes) / len(successes)
                target = 1.0 / 5.0
                sigma *= 1.0 + 0.5 * (success_rate - target)
                sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
