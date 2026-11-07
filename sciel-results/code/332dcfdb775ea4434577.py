"""Search/portfolio optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma, stagnation restarts,
    and a final local-polish phase. Explicit wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin vs 300s wall-clock

        def remaining():
            return self.budget > 0 and time.time() < deadline

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            self.budget -= 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: diverse initial scan to find good basins.
        # Büche-Rastrigin has many local minima; the global one is shifted/rotated.
        # Use a mix of structured and random starts.
        starts = [np.zeros(self.dim)]
        # Axis-aligned points at moderate distance
        for axis in range(self.dim):
            for sign in (-1, 1):
                v = np.zeros(self.dim)
                v[axis] = sign * 2.5
                starts.append(v)
        # Diagonal-ish structured starts
        for scale in (1.0, 2.0, 3.0):
            for sign in (-1, 1):
                starts.append(sign * scale * np.ones(self.dim) / np.sqrt(self.dim) * 2)
        # Random starts
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        for s in starts:
            if not remaining():
                break
            evaluate(s)

        # Phase 2: multiple (1+1)-ES runs with different sigma schedules.
        # Each run uses stagnation-based restart: if no improvement for
        # 'stagnation_limit' steps, restart from a random point or the best.
        strategies = [
            # (sigma_init, sigma_up, sigma_down, stagnation_limit)
            (2.0, 1.4, 0.82, 30),
            (1.0, 1.3, 0.80, 25),
            (3.0, 1.5, 0.78, 35),
            (0.5, 1.2, 0.85, 20),
            (1.5, 1.35, 0.80, 28),
        ]

        for strat_i, (sig0, up, down, stag_lim) in enumerate(strategies):
            if not remaining():
                break
            # Alternate starting from best vs random
            if best_x is not None and strat_i % 2 == 0:
                x = best_x.copy()
                fx = best_f
            else:
                x = np.clip(rng.uniform(-4.5, 4.5, self.dim), -5, 5)
                fx = evaluate(x)
                if not remaining():
                    break

            sigma = sig0
            no_improve = 0
            while remaining():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                if not remaining():
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * up, 5.0)
                    no_improve = 0
                else:
                    sigma *= down
                    no_improve += 1
                    if no_improve >= stag_lim:
                        # Stagnation: restart from best or random
                        if best_x is not None and rng.random() < 0.5:
                            x = best_x.copy()
                            fx = best_f
                        else:
                            x = np.clip(rng.uniform(-4.5, 4.5, self.dim), -5, 5)
                            fx = evaluate(x)
                            if not remaining():
                                break
                        sigma = sig0
                        no_improve = 0

        # Phase 3: local fine-tuning from the global best with small sigma.
        if best_x is not None and remaining():
            x = best_x.copy()
            fx = best_f
            sigma = 0.15
            no_improve = 0
            while remaining():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                if not remaining():
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 0.5)
                    no_improve = 0
                else:
                    sigma *= 0.92
                    no_improve += 1
                    if sigma < 1e-5 or no_improve > 40:
                        break

        # Phase 4: if budget remains, do a few more random-restart short runs
        # to explore other basins.
        if remaining():
            for _ in range(4):
                if not remaining():
                    break
                x = np.clip(rng.uniform(-4.5, 4.5, self.dim), -5, 5)
                fx = evaluate(x)
                if not remaining():
                    break
                sigma = 1.0
                no_improve = 0
                # Short burst: max 20 steps or until stagnation
                for _ in range(20):
                    if not remaining():
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = evaluate(y)
                    if not remaining():
                        break
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.3, 3.0)
                        no_improve = 0
                    else:
                        sigma *= 0.85
                        no_improve += 1
                        if no_improve > 12:
                            break

        if best_x is None:
            best_x = np.zeros(self.dim)
        return best_x, best_f
# EVOLVE-BLOCK-END
