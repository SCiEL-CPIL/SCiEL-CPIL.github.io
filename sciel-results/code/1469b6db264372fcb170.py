"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with adaptive sigma and periodic local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        calls_made = 1

        sigma = 2.0
        success_count = 0
        total_count = 0

        # Track best
        best_x = x.copy()
        best_f = fx

        # Local refinement via coordinate descent / Nelder-Mead style
        def local_refine(x_start, f_start, max_calls=20):
            nonlocal calls_made, best_x, best_f
            x_local = x_start.copy()
            f_local = f_start
            step = 0.5

            for _ in range(max_calls):
                if not time_ok() or calls_made >= budget:
                    break
                improved = False
                for i in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    # Try positive step
                    x_try = x_local.copy()
                    x_try[i] = np.clip(x_try[i] + step, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < f_local:
                        x_local = x_try
                        f_local = f_try
                        improved = True
                        if f_local < best_f:
                            best_f = f_local
                            best_x = x_local.copy()
                    else:
                        # Try negative step
                        x_try = x_local.copy()
                        x_try[i] = np.clip(x_try[i] - step, -5, 5)
                        f_try = func(x_try)
                        calls_made += 1
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                            if f_local < best_f:
                                best_f = f_local
                                best_x = x_local.copy()
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
                else:
                    step = min(step * 1.2, 2.0)

            return x_local, f_local

        # Main loop: (1+1)-ES with periodic local refinement
        refine_interval = 50
        last_refine = 0

        for step in range(1, budget):
            if not time_ok() or calls_made >= budget:
                break

            # Periodic local refinement
            if step - last_refine >= refine_interval:
                x, fx = local_refine(x, fx, max_calls=15)
                last_refine = step
                # Reset sigma after refinement
                sigma = max(sigma * 0.5, 0.1)
                success_count = 0
                total_count = 0
                continue

            # (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                pass

            total_count += 1
            if total_count >= 10:
                # 1/5 success rule
                if success_count >= total_count / 5:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0
                total_count = 0

            # Clamp sigma
            sigma = np.clip(sigma, 1e-3, 5.0)

        # Final local refinement if time permits
        if time_ok() and calls_made < budget:
            x, fx = local_refine(x, fx, max_calls=min(30, budget - calls_made))

        return best_x, best_f
# EVOLVE-BLOCK-END
