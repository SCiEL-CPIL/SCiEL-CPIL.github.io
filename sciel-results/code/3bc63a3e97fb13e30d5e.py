"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search: sample within shrinking hyper-sphere centered at origin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = np.zeros(self.dim)
        # Radius: WFG9 Pareto front is near origin in decision space.
        # Start wide enough to cover relevant region, shrink to focus.
        r_initial = 0.5  # half the unit hypercube radius
        r_final = 0.01   # very tight at end

        all_F = []
        used = 0

        # Initial batch: uniform random in full bounds to get a spread
        n_init = min(40, self.budget // 5)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = np.array(problem(x))
            if np.all(np.isfinite(f)):
                all_F.append(f)
            used += 1

        # Main loop: sample within shrinking hyper-sphere centered at origin
        remaining = self.budget - used
        for i in range(remaining):
            if time.time() > deadline:
                break
            # Linearly shrink radius from r_initial to r_final
            progress = i / max(1, remaining - 1)
            r = r_initial + (r_final - r_initial) * progress

            # Sample uniformly within hyper-sphere of radius r centered at origin
            # Generate direction on unit sphere, then radius^(1/dim) scaling
            direction = rng.standard_normal(self.dim)
            norm_d = np.linalg.norm(direction)
            if norm_d > 0:
                direction /= norm_d
            # Uniform in ball: radius * u^(1/dim)
            u = rng.random()
            actual_r = r * (u ** (1.0 / self.dim))
            x = center + actual_r * direction

            # Clip to bounds (safety)
            x = np.clip(x, lo, hi)

            f = np.array(problem(x))
            if np.all(np.isfinite(f)):
                all_F.append(f)
            used += 1

        # Final polish: very tight radius around best-known region
        if used < self.budget and time.time() < deadline:
            n_polish = min(20, self.budget - used)
            for i in range(n_polish):
                if time.time() > deadline:
                    break
                direction = rng.standard_normal(self.dim)
                norm_d = np.linalg.norm(direction)
                if norm_d > 0:
                    direction /= norm_d
                u = rng.random()
                actual_r = r_final * (u ** (1.0 / self.dim))
                x = center + actual_r * direction
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                if np.all(np.isfinite(f)):
                    all_F.append(f)
                used += 1

        return all_F
# EVOLVE-BLOCK-END
