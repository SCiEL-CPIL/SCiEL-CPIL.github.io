"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic multi-scale CMA-ES variant with fixed shrinking step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Fixed geometric sequence of step sizes from 4.0 to ~1e-4
        # Number of steps: let's use about 20 distinct scales
        # 4.0 * (decay)^k = 1e-4 => decay = (1e-4/4.0)^(1/19)
        num_scales = 20
        sigma_start = 4.0
        sigma_end = 1e-4
        decay = (sigma_end / sigma_start) ** (1.0 / (num_scales - 1))
        sigmas = [sigma_start * (decay ** k) for k in range(num_scales)]

        calls_made = 1

        # For each scale, perform a deterministic local search using
        # a fixed set of directions (coordinate axes + diagonals)
        # This gives a structured, non-random exploration

        # Generate a fixed set of directions: coordinate axes and their negatives,
        # plus some diagonal directions
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            d[i] = -1.0
            directions.append(d)
        # Add diagonal directions
        for i in range(dim - 1):
            d = np.zeros(dim)
            d[i] = 1.0
            d[i + 1] = 1.0
            directions.append(d)
            d[i] = 1.0
            d[i + 1] = -1.0
            directions.append(d)
            d[i] = -1.0
            d[i + 1] = 1.0
            directions.append(d)
            d[i] = -1.0
            d[i + 1] = -1.0
            directions.append(d)

        # Normalize directions
        directions = np.array([d / np.linalg.norm(d) for d in directions])

        # Multi-scale descent
        for sigma in sigmas:
            if calls_made >= budget:
                break

            # Try each direction with positive and negative steps
            for d in directions:
                if calls_made >= budget:
                    break

                # Try positive step
                y = np.clip(x + sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

                # Try negative step
                if calls_made >= budget:
                    break
                y = np.clip(x - sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

            # Check for convergence to a plateau: if no improvement in this scale
            # and we're not at the first scale, consider a restart
            # Actually, let's detect if we made zero improvements in this scale
            # We need to track improvements per scale

        # If we still have budget left, do a final fine-grained coordinate search
        remaining = budget - calls_made
        if remaining > 0:
            # Fine coordinate search
            fine_sigma = 1e-4
            for _ in range(remaining):
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Positive
                    y = x.copy()
                    y[i] = np.clip(y[i] + fine_sigma, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        improved = True
                        break
                if improved:
                    continue
                # Negative
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    y = x.copy()
                    y[i] = np.clip(y[i] - fine_sigma, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        improved = True
                        break
                if not improved:
                    fine_sigma *= 0.5
                    if fine_sigma < 1e-7:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
