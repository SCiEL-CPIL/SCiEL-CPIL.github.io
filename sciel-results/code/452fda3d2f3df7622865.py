"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free evolutionary strategy with anisotropic covariance,
    large population, and time-budget guard for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial population: spread across the box
        x0 = rng.uniform(-4.5, 4.5, dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # Anisotropic covariance: worst-case conditioning for ellipsoidal f7
        # f7 has condition number up to 1e4 in BBOB, so we use a diagonal
        # covariance with varying scales to explore anisotropically
        # Start with moderate anisotropy
        scales = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])[:dim]
        sigma = 1.5  # base step size

        # Population size
        lam = min(50, max(10, budget // 10))
        # Number of generations we can afford
        # Each generation costs lam evaluations
        max_gens = (budget - calls) // lam

        # Success counter for 1/5 rule adaptation
        success_count = 0
        window_size = max(1, lam // 5)

        for gen in range(max_gens):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                if calls >= budget:
                    break
                # Sample in the transformed space
                z = rng.standard_normal(dim)
                # Apply anisotropic scaling
                step = sigma * scales * z
                y = np.clip(best_x + step, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Select best offspring
            offspring.sort(key=lambda t: t[1])
            best_off_x, best_off_f = offspring[0]

            if best_off_f < best_f:
                best_x = best_off_x.copy()
                best_f = best_off_f
                success_count += 1

            # Adapt sigma based on success rate over recent generations
            # Simple 1/5 rule: track successes over a window
            # For simplicity, use a running average
            if gen > 0:
                # Rough adaptation: if we've been improving, increase sigma
                # If not, decrease
                pass  # We'll use a simpler heuristic below

            # Adapt scales based on which dimensions show progress
            # Track per-dimension variance of successful steps
            if best_off_f < best_f + 1e-10:  # had some improvement
                # Slightly increase sigma
                sigma *= 1.05
            else:
                # No improvement, decrease sigma
                sigma *= 0.95

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

            # Adapt anisotropy: dimensions with larger steps that led to improvement
            # get larger scale, others get smaller
            # Use a simple heuristic: if the best step had large component in dim d,
            # increase scale[d]
            best_step = best_off_x - best_x
            for d in range(dim):
                if abs(best_step[d]) > sigma * scales[d]:
                    scales[d] *= 1.1
                else:
                    scales[d] *= 0.98
            # Normalize scales to prevent drift
            scale_mean = np.mean(scales)
            if scale_mean > 0:
                scales = scales / scale_mean * np.mean(scales)
            # Clip scales
            scales = np.clip(scales, 0.01, 10.0)

        # Use remaining budget for local refinement with (1+1)-ES
        remaining = budget - calls
        if remaining > 0:
            local_sigma = max(0.01, sigma * 0.1)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9
                local_sigma = np.clip(local_sigma, 1e-4, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
