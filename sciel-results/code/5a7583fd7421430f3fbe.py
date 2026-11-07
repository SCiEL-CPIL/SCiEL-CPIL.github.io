"""Combined optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: coarse random sampling + CMA-ES-like covariance-adapted local search + adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s kill

        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_func(x):
            nonlocal best_f, best_x, call_count
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64).clip(-5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return best_f if best_f != float('inf') else 1e10
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Coarse random sampling to find good starting points
        # f21 has 101 peaks, so we need decent exploration
        n_random = min(80, max(20, budget // 6))
        for i in range(n_random):
            if call_count >= budget or time.time() > deadline - 60:
                break
            x = rng.uniform(-4, 4, dim)
            safe_func(x)

        # Also try a few structured points
        special_starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
        ]
        for s in special_starts:
            if call_count >= budget or time.time() > deadline - 60:
                break
            safe_func(s)

        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        # Phase 2: CMA-ES-like local search from the best point
        # The covariance matrix adapts to the hidden rotation, which is critical
        # for f21's rotated peaks.
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        C = np.eye(dim)
        no_improve = 0
        max_no_improve = 40
        # Reserve budget for Phase 3
        phase2_budget = budget - call_count - 50
        if phase2_budget < 30:
            phase2_budget = max(0, budget - call_count)

        for step in range(phase2_budget):
            if time.time() > deadline - 30:
                break
            if call_count >= budget:
                break

            # Generate candidate using adapted covariance
            z = rng.standard_normal(dim)
            try:
                L = np.linalg.cholesky(C)
                y = x + sigma * (L @ z)
            except np.linalg.LinAlgError:
                y = x + sigma * z

            y = np.clip(y, -5, 5)
            fy = safe_func(y)

            if fy < fx:
                # Success: update position and covariance
                direction = (y - x) / max(sigma, 1e-10)
                direction = np.clip(direction, -10, 10)
                # Rank-1 covariance update with decay
                C = 0.85 * C + 0.15 * np.outer(direction, direction)
                # Normalize to keep trace ~ dim
                tr = np.trace(C)
                if tr > 1e-10:
                    C = C / tr * dim
                x, fx = y, fy
                sigma *= 1.2
                no_improve = 0
            else:
                no_improve += 1
                sigma *= 0.8
                # Occasionally reset covariance to avoid getting stuck
                if no_improve % 15 == 0:
                    C = 0.5 * C + 0.5 * np.eye(dim)
                    tr = np.trace(C)
                    if tr > 1e-10:
                        C = C / tr * dim
                if no_improve >= max_no_improve:
                    break

        # Phase 3: Adaptive local refinement around the best point
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            sigma_ref = 0.3
            max_ref_steps = min(150, budget - call_count)
            for step in range(max_ref_steps):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break
                y = np.clip(best_x + sigma_ref * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < best_f:
                    sigma_ref *= 1.3
                else:
                    sigma_ref *= 0.7
                if sigma_ref < 0.005:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
