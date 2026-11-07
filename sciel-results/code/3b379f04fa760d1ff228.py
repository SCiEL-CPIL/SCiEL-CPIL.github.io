"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: estimate gradient, march to best corner, then refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        def time_ok():
            return time.time() < deadline

        # Phase 1: Estimate gradient using finite differences
        # Start at origin (center of box)
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # Estimate gradient using central differences with small step
        step = 0.1
        grad = np.zeros(dim)
        for i in range(dim):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += step
            xm[i] -= step
            fp = func(xp)
            fm = func(xm)
            grad[i] = (fp - fm) / (2 * step)
        
        # Check if we're out of budget or time
        if not time_ok():
            return x0, f0

        # Phase 2: For a linear function, the minimum is at a corner
        # The sign of the gradient tells us which direction to go
        # If grad[i] > 0, we want to go to -5 in dimension i
        # If grad[i] < 0, we want to go to +5 in dimension i
        # If grad[i] == 0, the dimension doesn't matter
        
        # Evaluate the predicted optimal corner
        x_corner = np.where(grad > 0, -5.0, 5.0)
        f_corner = func(x_corner)
        
        best_x = x_corner
        best_f = f_corner
        
        if not time_ok():
            return best_x, best_f

        # Phase 3: Refine around the corner using local search
        # The function might not be perfectly linear, so let's do some local optimization
        # Use a simple coordinate descent / local search
        
        # First, let's try a few nearby points to see if we can improve
        sigma = 0.5
        for _ in range(100):
            if not time_ok():
                break
            # Try moving slightly inward from the corner
            for i in range(dim):
                if not time_ok():
                    break
                # Move inward from the boundary
                direction = 1.0 if x_corner[i] == 5.0 else -1.0
                x_try = best_x.copy()
                x_try[i] -= direction * sigma
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                if f_try < best_f:
                    best_x = x_try
                    best_f = f_try
            sigma *= 0.7
            if sigma < 1e-4:
                break

        # Phase 4: Use remaining budget for final polishing with small steps
        if time_ok():
            sigma = 0.01
            for _ in range(50):
                if not time_ok():
                    break
                for i in range(dim):
                    if not time_ok():
                        break
                    # Try both directions
                    for direction in [1, -1]:
                        x_try = best_x.copy()
                        x_try[i] += direction * sigma
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        if f_try < best_f:
                            best_x = x_try
                            best_f = f_try
                sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
