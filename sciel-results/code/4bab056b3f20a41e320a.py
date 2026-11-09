"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Two-stage optimizer: coarse LHS bracketing + CMA-ES + L-BFGS-B + coordinate polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track all evaluations
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Stage 1: Coarse Latin Hypercube sampling (20 points)
        # Use stratified sampling in each dimension
        n_lhs = 20
        lhs_points = []
        for i in range(n_lhs):
            # Stratified sampling: divide each dimension into n_lhs equal parts
            point = np.zeros(n)
            for d in range(n):
                # Random position within the i-th stratum
                pos = (i + rng.random()) / n_lhs
                point[d] = -5.0 + 10.0 * pos
            lhs_points.append(point)

        for p in lhs_points:
            if eval_count >= budget or time.time() > deadline:
                break
            evaluate(p)

        # Also evaluate origin and a few deterministic points
        for p in [np.zeros(n), np.ones(n) * 2.0, np.ones(n) * -2.0]:
            if eval_count >= budget or time.time() > deadline:
                break
            evaluate(p)

        # Stage 2: CMA-ES from best point
        if eval_count < budget and time.time() < deadline:
            x_center = best_x.copy()
            sigma = 1.5  # moderate initial step size

            # Simple (1+1)-ES with adaptive sigma, but with better initialization
            # Use a population-based approach for better exploration
            pop_size = 10
            sigma = 1.0
            success_ratio = 0.5

            for gen in range(20):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Generate population
                for _ in range(pop_size):
                    if eval_count >= budget or time.time() > deadline:
                        break
                    y = np.clip(x_center + sigma * rng.standard_normal(n), -5.0, 5.0)
                    fy = evaluate(y)

            # Update center to best in population (already tracked)
            x_center = best_x.copy()

            # Adapt sigma based on progress
            # Simple rule: if we made progress recently, increase sigma slightly
            # Otherwise decrease
            # For simplicity, use a fixed schedule that decreases over generations
            sigma *= 0.9

        # Stage 3: L-BFGS-B from best point with tight tolerances
        if eval_count < budget and time.time() < deadline and best_x is not None:
            x_start = best_x.copy()

            # Use L-BFGS-B with tight tolerances
            def objective(x):
                nonlocal eval_count
                if eval_count >= budget or time.time() > deadline:
                    return best_f  # Return best known to stop
                x_clipped = np.clip(x, -5.0, 5.0)
                f = float(func(x_clipped))
                eval_count += 1
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
                return f

            try:
                res = minimize(
                    objective,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxfun': 80,
                        'gtol': 1e-10,
                        'ftol': 1e-14,
                        'maxiter': 100
                    }
                )
                if res.x is not None:
                    f_res = float(func(np.clip(res.x, -5.0, 5.0)))
                    eval_count += 1
                    if f_res < best_f:
                        best_f = f_res
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Stage 4: Coordinate-wise golden-section polish
        if eval_count < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
            inv_phi = 1.0 / golden_ratio
            inv_phi2 = inv_phi * inv_phi

            for _ in range(3):  # 3 passes over all coordinates
                for d in range(n):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Determine search interval around current best
                    # Use a window that shrinks each pass
                    window = 0.5 / (_ + 1)
                    lo = max(-5.0, x[d] - window)
                    hi = min(5.0, x[d] + window)

                    if hi - lo < 1e-12:
                        continue

                    # Golden section search along dimension d
                    c = hi - inv_phi * (hi - lo)
                    d_pos = lo + inv_phi * (hi - lo)

                    x_c = x.copy()
                    x_c[d] = c
                    x_d = x.copy()
                    x_d[d] = d_pos

                    fc = evaluate(x_c)
                    fd = evaluate(x_d)

                    for _ in range(15):  # 15 iterations of golden section
                        if eval_count >= budget or time.time() > deadline:
                            break
                        if fc < fd:
                            hi = d_pos
                            d_pos = c
                            fd = fc
                            c = hi - inv_phi * (hi - lo)
                            x_c = x.copy()
                            x_c[d] = c
                            fc = evaluate(x_c)
                        else:
                            lo = c
                            c = d_pos
                            fc = fd
                            d_pos = lo + inv_phi * (hi - lo)
                            x_d = x.copy()
                            x_d[d] = d_pos
                            fd = evaluate(x_d)

                    # Update x with best coordinate value
                    if fc < fd:
                        x[d] = c
                    else:
                        x[d] = d_pos

                    if x[d] != x[d]:  # NaN check
                        x[d] = best_x[d]

            # Final evaluation of polished point
            if eval_count < budget:
                f_final = evaluate(x)

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))
            eval_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
