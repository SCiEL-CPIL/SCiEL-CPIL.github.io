"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 rule step-size adaptation and multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def es_11_search(x0, sigma0, max_calls):
            """
            Runs a (1+1)-ES instance with 1/5 rule step-size adaptation.
            Returns (best_x_local, best_f_local, calls_made_in_this_run)
            """
            nonlocal best_x, best_f, call_count

            n = self.dim
            x = x0.copy()
            sigma = sigma0
            local_best_x = x.copy()
            local_best_f = np.inf
            calls_in_run = 0

            # Evaluate starting point
            fx = make_call(x)
            if fx is None:
                return local_best_x, local_best_f, 1
            calls_in_run += 1
            if np.isfinite(fx) and fx < local_best_f:
                local_best_f = fx
                local_best_x = x.copy()
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # 1/5 rule parameters
            # Target success rate is 1/5 = 0.2
            # We track successes over a window
            success_window = []
            window_size = 15  # Number of steps to consider for adaptation

            stagnation_counter = 0
            last_improvement_calls = 0

            while calls_in_run < max_calls:
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 2.0:
                    break

                # Generate one offspring
                z = rng.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)

                fy = make_call(y)
                if fy is None:
                    break
                calls_in_run += 1

                # Check if offspring is better
                if np.isfinite(fy) and fy < fx:
                    # Success: move to offspring
                    x = y.copy()
                    fx = fy
                    success_window.append(1)

                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = x.copy()
                        if fy < best_f:
                            best_f = fy
                            best_x = x.copy()
                        stagnation_counter = 0
                        last_improvement_calls = calls_in_run
                    else:
                        stagnation_counter += 1
                else:
                    # Failure: stay at current point
                    success_window.append(0)
                    stagnation_counter += 1

                # Keep window bounded
                if len(success_window) > window_size:
                    success_window.pop(0)

                # Adapt step size using 1/5 rule
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window)
                    # If success rate > 1/5, increase sigma; else decrease
                    # Use a multiplicative update
                    if success_rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8

                    # Clamp sigma to reasonable bounds
                    sigma = np.clip(sigma, 1e-8, 10.0)

                # Stagnation check: if no improvement for many steps, restart
                if stagnation_counter > 50:
                    break

            return local_best_x, local_best_f, calls_in_run

        # Strategy:
        # 1. Start from the center (origin) as a baseline.
        # 2. Run (1+1)-ES from there.
        # 3. If stagnation occurs or budget remains, restart from best found with a new random perturbation.

        # Initial start: center
        x_start = np.zeros(self.dim)
        fx_start = make_call(x_start)
        if fx_start is None:
            return best_x, best_f
        if np.isfinite(fx_start) and fx_start < best_f:
            best_f = fx_start
            best_x = x_start.copy()

        # First (1+1)-ES run
        sigma_init = 2.0
        max_calls_first = 200
        x_best, f_best, calls_made = es_11_search(x_start, sigma_init, max_calls_first)

        # If we have budget left, do additional restarts
        # Use the best point found so far as the base for new starts
        restart_count = 0
        max_restarts = 3

        while call_count < self.budget - 10 and restart_count < max_restarts:
            if time.time() > deadline - 5.0:
                break

            # Create a new starting point: best_x + random perturbation
            if best_x is not None:
                perturbation = rng.standard_normal(self.dim) * 0.5
                x_new_start = best_x + perturbation
                x_new_start = np.clip(x_new_start, -5, 5)
            else:
                x_new_start = rng.uniform(-3, 3, self.dim)

            # Use a slightly smaller sigma for exploitation
            sigma_restart = 1.0 if restart_count == 0 else 0.5
            max_calls_restart = min(150, self.budget - call_count - 10)

            if max_calls_restart <= 10:
                break

            x_best_r, f_best_r, calls_made_r = es_11_search(x_new_start, sigma_restart, max_calls_restart)
            restart_count += 1

            # If no improvement from this restart, we could stop early
            # But let's continue if we have budget

        # Final fallback
        if best_x is None:
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
