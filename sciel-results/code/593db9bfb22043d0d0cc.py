"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
import numpy as np
import time


class Optimizer:
    """SA with adaptive cooling, broad scout initialization, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track global best
        best_x = rng.uniform(-5.0, 5.0, dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Broad scout sampling to find a decent initial region
        # For step ellipsoidal, we need to find the boundary of the plateau
        n_scout = max(15, budget // 25)  # ~20 samples for budget=500
        for i in range(n_scout):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of wide and narrow sampling
            if i < n_scout // 3:
                x = rng.uniform(-5.0, 5.0, dim)
            elif i < 2 * n_scout // 3:
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                x = rng.uniform(-1.5, 1.5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Simulated Annealing with adaptive cooling
        # Initialize from the best point found
        x = best_x.copy()
        fx = best_f

        # SA parameters
        # Initial temperature: scale with typical function value range
        # For f7, values can be large on plateau, small near minimum
        T_init = max(1.0, abs(best_f) * 0.1 + 1.0)
        T_min = 1e-6
        # Cooling rate: we want to reach T_min in roughly the remaining budget
        remaining_calls = budget - calls
        # Use geometric cooling: T_k = T_init * alpha^k
        # Want T_init * alpha^(remaining_calls) ~ T_min
        # alpha = (T_min / T_init)^(1/remaining_calls)
        if remaining_calls > 0:
            alpha = (T_min / T_init) ** (1.0 / max(remaining_calls, 1))
        else:
            alpha = 0.95
        # Clamp alpha to reasonable range
        alpha = np.clip(alpha, 0.90, 0.999)

        T = T_init
        step_size = 1.0  # Initial perturbation scale
        min_step = 0.01
        max_step = 5.0

        # Plateau detection: track consecutive non-improvements
        no_improve = 0
        max_no_improve = 30  # Restart SA if stuck for this many steps
        restart_count = 0
        max_restarts = 5

        # Adaptive step size: increase on acceptance, decrease on rejection
        # But with SA, we accept uphill moves, so use a different heuristic
        # Track acceptance rate in a window to adapt step size
        window_size = 20
        accept_count = 0
        total_count = 0

        while calls < budget and time.time() < deadline:
            # Generate perturbation
            # Use step_size to scale the Gaussian perturbation
            y = x + step_size * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1

            # SA acceptance criterion
            if fy < fx:
                # Always accept downhill moves
                x = y
                fx = fy
                accept_count += 1
                total_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
            else:
                # Uphill move: accept with probability exp(-(fy-fx)/T)
                delta_f = fy - fx
                if delta_f > 0 and T > 0:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        x = y
                        fx = fy
                        accept_count += 1
                        total_count += 1
                    else:
                        total_count += 1
                else:
                    total_count += 1
                no_improve += 1

            # Cool down
            T *= alpha
            T = max(T, T_min)

            # Adaptive step size based on acceptance rate
            if total_count >= window_size:
                accept_rate = accept_count / total_count
                # Target acceptance rate ~0.3-0.5 for good exploration
                if accept_rate > 0.6:
                    step_size *= 1.1  # Too many acceptances, increase step
                elif accept_rate < 0.1:
                    step_size *= 0.9  # Too few acceptances, decrease step
                step_size = np.clip(step_size, min_step, max_step)
                accept_count = 0
                total_count = 0

            # Restart if stuck on plateau
            if no_improve >= max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break

                # Restart from a new point
                # Mix: sometimes pure random, sometimes perturbation of best
                if rng.random() < 0.4:
                    x = rng.uniform(-5.0, 5.0, dim)
                else:
                    x = best_x + 2.0 * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Reset SA parameters for new restart
                T = T_init * (0.5 ** restart_count)  # Lower initial T for later restarts
                step_size = 1.0
                no_improve = 0
                accept_count = 0
                total_count = 0
                # Increase patience slightly for later restarts
                max_no_improve = max(20, 30 - restart_count * 3)

        return best_x, best_f
