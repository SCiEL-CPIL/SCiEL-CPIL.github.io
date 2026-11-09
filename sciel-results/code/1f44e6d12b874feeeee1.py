"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: (1+1)-ES with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initial point: start near origin but with some spread
        x = rng.uniform(-1, 1, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Start with larger sigma to explore the domain faster
        sigma = 2.0
        no_improve = 0
        max_no_improve = 40

        calls_made = 1

        # Track progress for adaptive sigma
        prev_fx = fx

        while calls_made < self.budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Adaptive sigma: increase when making progress, decrease when stuck
            # Rosenbrock has a narrow valley, so we need careful step sizing
            if fx < prev_fx:
                # Making progress - allow slightly larger steps
                sigma = min(sigma * 1.15, 5.0)
            else:
                # Not making progress - shrink
                sigma *= 0.85
                sigma = max(sigma, 0.01)

            prev_fx = fx

            # Generate candidate with anisotropic perturbation
            # Rosenbrock valley runs roughly along the diagonal in rotated space
            # Use a mix of isotropic and slightly elongated steps
            noise = rng.standard_normal(self.dim)
            
            # Occasionally add a larger perturbation to escape local minima
            if rng.random() < 0.1:
                noise = noise * 1.5
            
            y = x + sigma * noise
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                no_improve = 0

                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Reject
                no_improve += 1

                # Restart if stuck
                if no_improve >= max_no_improve:
                    # Try multiple restart strategies
                    restart_success = False
                    
                    # Strategy 1: Larger perturbation from best point
                    for _ in range(8):
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        candidate = best_x + 3.0 * rng.standard_normal(self.dim)
                        candidate = np.clip(candidate, -5, 5)
                        cf = func(candidate)
                        calls_made += 1
                        if cf < best_fx:
                            best_fx = cf
                            best_x = candidate.copy()
                            x = best_x.copy()
                            fx = best_fx
                            no_improve = 0
                            restart_success = True
                            break

                    # Strategy 2: If still stuck, try a few random points
                    if not restart_success:
                        for _ in range(3):
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            x = rng.uniform(-4, 4, self.dim)
                            fx = func(x)
                            calls_made += 1
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                                no_improve = 0
                                restart_success = True
                                break

                    # If all else fails, do a full random restart
                    if not restart_success:
                        x = rng.uniform(-4, 4, self.dim)
                        fx = func(x)
                        calls_made += 1
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        no_improve = 0
                        sigma = 2.0

        # Local refinement using Nelder-Mead if we have time and budget
        if time.time() < deadline - 20 and calls_made < self.budget - 10:
            try:
                from scipy.optimize import minimize

                remaining = self.budget - calls_made

                def wrapper(z):
                    nonlocal calls_made
                    if calls_made >= self.budget or time.time() > deadline:
                        return best_fx
                    calls_made += 1
                    return func(np.clip(z, -5, 5))

                # First pass: moderate tolerance
                result = minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining // 3, 200),
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )

                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5)

                # Second pass: tighter tolerance for final polish
                if time.time() < deadline - 5 and calls_made < self.budget - 5:
                    remaining2 = self.budget - calls_made
                    if remaining2 > 10:
                        result2 = minimize(
                            wrapper,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': min(remaining2 // 2, 100),
                                'xatol': 1e-8,
                                'fatol': 1e-10,
                                'adaptive': True
                            }
                        )
                        if result2.fun < best_fx:
                            best_fx = result2.fun
                            best_x = np.clip(result2.x, -5, 5)

            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
