import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with scheduled restarts and local polish for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 0: quick multi-start to seed a rough best (helps AOCC early progress)
        # Use a slightly tighter range since the minimizer is in [-4, 4]^5
        for _ in range(min(10, budget)):
            if not time_ok() or calls >= budget:
                break
            x0 = rng.uniform(-4.5, 4.5, d)
            fx0 = float(func(x0))
            calls += 1
            record(x0, fx0)

        # Reserve budget for final polish
        polish_budget = max(50, int(0.15 * budget))
        es_limit = budget - polish_budget

        # Phase 1: adaptive (1+1)-ES with scheduled restarts
        # Weierstrass is highly multimodal; restarts with increasing perturbation help escape local minima
        # Schedule: (initial_sigma, spread_for_restart)
        # spread_for_restart controls how far from best_x to start a new restart
        schedule = [
            (1.5, 2.0),
            (1.2, 1.8),
            (0.9, 1.5),
            (0.6, 1.2),
            (0.4, 0.9),
            (0.25, 0.6),
            (0.15, 0.4),
            (0.08, 0.25),
            (0.04, 0.15),
            (0.02, 0.08),
            (0.01, 0.05),
        ]

        c_sigma = 0.5
        zeta = 0.33

        for sigma0, spread in schedule:
            if not time_ok() or calls >= es_limit:
                break

            # Initialize: first restart at best, subsequent restarts perturbed from best
            if best_x is None:
                x = rng.uniform(-4.5, 4.5, d)
            else:
                # Perturb around best with the scheduled spread
                x = np.clip(best_x + spread * rng.standard_normal(d), -5.0, 5.0)

            fx = float(func(x))
            calls += 1
            record(x, fx)
            sigma = sigma0

            ew_succ = 0.0
            ew_total = 0.0
            no_improve = 0
            stag_limit = 35  # Slightly tighter to trigger restarts more often

            while time_ok() and calls < es_limit:
                z = rng.standard_normal(d)
                y = np.clip(x + sigma * z, -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                success = fy < fx
                ew_succ = 0.8 * ew_succ + 0.2 * (1.0 if success else 0.0)
                ew_total = 0.8 * ew_total + 0.2
                sr = ew_succ / max(ew_total, 1e-12)
                if success:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                sigma *= np.exp(c_sigma * (sr - zeta))
                sigma = float(np.clip(sigma, 1e-7, 5.0))
                if no_improve > stag_limit:
                    break

        # Phase 2: local polish
        if best_x is not None and time_ok() and calls < budget:
            x = best_x.copy()
            fx = best_fx

            # Coordinate descent with adaptive step
            for _ in range(3):
                if not time_ok() or calls >= budget:
                    break
                for coord in range(d):
                    if not time_ok() or calls >= budget:
                        break
                    for direction in (1, -1):
                        if not time_ok() or calls >= budget:
                            break
                        step = 0.3
                        for _ in range(10):
                            if not time_ok() or calls >= budget:
                                break
                            y = x.copy()
                            y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                            fy = float(func(y))
                            calls += 1
                            if fy < fx:
                                x = y
                                fx = fy
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                step *= 0.7
                            else:
                                break

            # Adaptive random fine steps with tighter bounds
            sigma = 0.05
            while time_ok() and calls < budget:
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.8

        # Safety: ensure a valid result
        if best_x is None:
            x = np.zeros(d)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
