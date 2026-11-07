"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing + coordinate descent hybrid for rotated Schwefel."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # safety margin before 330s kill
        calls = 0
        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls
            calls += 1

        def safe_func(x):
            """Wrap func to ensure we don't exceed budget."""
            nonlocal calls
            if calls >= self.budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            count_call()
            return fx

        # Phase 1: Simulated Annealing with geometric cooling
        # Large initial step to explore global landscape, probabilistic acceptance
        # of worse states to escape local minima in rotated Schwefel
        sa_budget = 200
        T0 = 50.0  # initial temperature (large for exploration)
        T_end = 0.01  # final temperature
        cooling_rate = (T_end / T0) ** (1.0 / max(1, sa_budget - 1))

        x = np.zeros(self.dim)  # start at center
        fx = safe_func(x)
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        T = T0
        for sa_iter in range(sa_budget):
            if calls >= self.budget or time.time() > deadline:
                break

            # Generate neighbor with large step (scales with temperature)
            step_scale = 1.0 + 0.5 * (T / T0)
            delta = rng.normal(0, step_scale, self.dim)
            y = np.clip(x + delta, -5.0, 5.0)
            fy = safe_func(y)

            if fy >= np.inf:
                break

            # Accept if better, or with probability exp(-delta_f / T)
            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                delta_f = fy - fx
                # Geometric cooling
                T *= cooling_rate
                if T < T_end:
                    T = T_end
                accept_prob = np.exp(-delta_f / max(T, 1e-10))
                if rng.random() < accept_prob:
                    x = y
                    fx = fy

        # Phase 2: Coordinate descent refinement from best point
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            # Adaptive step sizes per dimension
            steps = np.ones(self.dim) * 0.5
            no_improve = 0
            max_no_improve = 12
            cd_budget = min(self.budget - calls, 250)

            for it in range(cd_budget):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break

                # Cycle through dimensions
                dim_idx = it % self.dim
                improved_this_pass = False

                # Try both directions
                for sign in [1.0, -1.0]:
                    if calls >= self.budget or time.time() > deadline:
                        break
                    y = x.copy()
                    y[dim_idx] = np.clip(y[dim_idx] + sign * steps[dim_idx], -5.0, 5.0)
                    fy = safe_func(y)

                    if fy >= np.inf:
                        break

                    if fy < fx:
                        x = y
                        fx = fy
                        steps[dim_idx] *= 1.2
                        improved_this_pass = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                    else:
                        steps[dim_idx] *= 0.75
                        if steps[dim_idx] < 0.005:
                            steps[dim_idx] = 0.005

                if improved_this_pass:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        # Small random perturbation to escape local minima
                        x = np.clip(x + rng.normal(0, 0.3, self.dim), -5.0, 5.0)
                        fx = safe_func(x)
                        if fx >= np.inf:
                            break
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                        steps = np.maximum(steps, 0.3)

        # Phase 3: Fine-tuning polish with very small steps
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            steps = np.ones(self.dim) * 0.1
            no_improve = 0

            while calls < self.budget and time.time() < deadline and no_improve < 20:
                dim_idx = rng.integers(self.dim)
                sign = rng.choice([-1.0, 1.0])
                y = x.copy()
                y[dim_idx] = np.clip(y[dim_idx] + sign * steps[dim_idx], -5.0, 5.0)
                fy = safe_func(y)

                if fy >= np.inf:
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    steps[dim_idx] *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                else:
                    steps[dim_idx] *= 0.85
                    if steps[dim_idx] < 0.001:
                        steps[dim_idx] = 0.001
                    no_improve += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
