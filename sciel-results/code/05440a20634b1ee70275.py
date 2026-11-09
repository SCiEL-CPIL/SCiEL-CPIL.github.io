"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: estimate gradient, jump to optimal vertex, polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Estimate the effective gradient in original coordinates
        # For linear f(x) = c + w·x, we estimate w_i via finite differences
        h = 1.0  # step size for gradient estimation

        # Evaluate at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1

        # Estimate gradient components
        w = np.zeros(dim)
        for i in range(dim):
            if calls >= budget:
                break
            xp = np.zeros(dim)
            xp[i] = h
            xm = np.zeros(dim)
            xm[i] = -h
            fp = func(xp)
            fm = func(xm)
            calls += 2
            w[i] = (fp - fm) / (2.0 * h)

        # Phase 2: Compute optimal vertex for linear function
        # Minimum of c + w·x on [-5,5]^d is at x_i = -5*sign(w_i)
        x_best = np.zeros(dim)
        for i in range(dim):
            if w[i] > 0:
                x_best[i] = -5.0
            elif w[i] < 0:
                x_best[i] = 5.0
            else:
                x_best[i] = 0.0

        f_best = func(x_best)
        calls += 1

        # Phase 3: Verify and polish with a few local steps
        # Try perturbing each coordinate slightly to see if we can improve
        # For a truly linear function, the vertex is optimal, but numerical
        # issues or near-zero gradient components might benefit from refinement
        for _ in range(min(20, budget - calls)):
            if time.time() > deadline:
                break
            improved = False
            for i in range(dim):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                # Try moving coordinate i toward the boundary in the sign direction
                # or try a small step in the negative gradient direction
                step = 0.1
                for delta in [step, -step, 2*step, -2*step]:
                    if calls >= budget:
                        break
                    y = x_best.copy()
                    y[i] = np.clip(x_best[i] + delta, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    if fy < f_best:
                        x_best = y
                        f_best = fy
                        improved = True
                        break
            if not improved:
                # Try a small random perturbation
                if calls >= budget:
                    break
                rng = np.random.default_rng(42)
                y = np.clip(x_best + 0.05 * rng.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < f_best:
                    x_best = y
                    f_best = fy

        # Phase 4: Final high-precision polish - check all vertices near the best
        # For linear functions, the true optimum is exactly at a vertex
        # Let's verify by checking the 2^dim vertices if budget allows (dim=5, so 32 vertices)
        if calls < budget and time.time() < deadline:
            # Generate all 2^dim vertices
            vertices = np.array([[(-1 if (j >> i) & 1 else 1) * 5.0 for i in range(dim)] for j in range(2**dim)])
            # Sort by estimated distance from x_best (closer first)
            dists = np.array([np.sum((v - x_best)**2) for v in vertices])
            order = np.argsort(dists)
            for idx in order:
                if calls >= budget or time.time() > deadline:
                    break
                v = vertices[idx]
                fv = func(v)
                calls += 1
                if fv < f_best:
                    x_best = v
                    f_best = fv

        return x_best, f_best
# EVOLVE-BLOCK-END
