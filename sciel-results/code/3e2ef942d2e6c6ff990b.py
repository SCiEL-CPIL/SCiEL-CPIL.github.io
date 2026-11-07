"""Black-box optimizer for BBOB-style Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: coarse grid + (1+1)-ES with adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        calls = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Coarse exploration to find a good starting region
        # Evaluate a grid of points in the interior
        best_x = None
        best_f = float('inf')

        # Start with center and some random points
        candidates = []
        # Center
        candidates.append(np.zeros(self.dim))
        # Random points
        for _ in range(20):
            candidates.append(rng.uniform(-4, 4, self.dim))
        # Axis-aligned points near center
        for d in range(self.dim):
            for s in [-2, -1, 1, 2]:
                pt = np.zeros(self.dim)
                pt[d] = s
                candidates.append(pt)

        for c in candidates:
            if not check_time():
                break
            if calls >= self.budget:
                break
            fc = evaluate(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)

        # Phase 2: (1+1)-ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        success_count = 0
        total_trials = 0
        window = 20

        while calls < self.budget and check_time():
            # Adaptive sigma: 1/5 rule
            if total_trials >= window:
                success_rate = success_count / total_trials
                if success_rate > 0.2:
                    sigma *= 1.3
                elif success_rate < 0.15:
                    sigma *= 0.7
                success_count = 0
                total_trials = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            fy = evaluate(y)
            total_trials += 1

            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                best_x = x.copy()
                best_f = fx
            else:
                # Occasionally try a larger jump to escape local minima
                if total_trials % 50 == 0:
                    jump = x + 2.0 * rng.standard_normal(self.dim)
                    jump = np.clip(jump, -5.0, 5.0)
                    fj = evaluate(jump)
                    if fj < fx:
                        x = jump
                        fx = fj
                        best_x = x.copy()
                        best_f = fx
                        sigma = 1.0

        # Phase 3: Fine polishing with small steps
        if calls < self.budget and check_time():
            sigma_fine = 0.1
            for _ in range(30):
                if calls >= self.budget or not check_time():
                    break
                y = x + sigma_fine * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                sigma_fine *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
