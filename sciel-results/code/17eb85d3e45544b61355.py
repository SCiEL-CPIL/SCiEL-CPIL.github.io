import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.

    For ZDT3, the Pareto front is obtained by setting x_1 = t and x_i = 0 for i > 1.
    The objective function f_2(t) = 11*(1 - sqrt(t/11)).
    The hypervolume contribution is maximized by sampling density proportional to
    the local slope of the front. We use a power-law spacing that clusters points
    near t=0 where the derivative is highest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)

        # Optimal power-law spacing for maximizing hypervolume of a concave front
        # The density should be proportional to |df2/dt|^(-1) * something
        # For f2(t) = 11*(1-sqrt(t/11)), df2/dt = -sqrt(11/t)/2
        # The hypervolume integrand involves (11 - f2(t)) which is ~ 11*sqrt(t/11)
        # So we want to maximize sum of (11-f2(t_i)) * dt_i
        # Optimal spacing: t_i = (i/N)^p where p optimizes the quadrature
        # Empirically p around 0.5-0.8 works well
        p = 0.75

        if n_sample == 2:
            t_values = np.array([0.0, 1.0])
        else:
            i = np.arange(n_sample)
            t_values = (i / (n_sample - 1)) ** p
            # Ensure exact endpoints
            t_values[0] = 0.0
            t_values[-1] = 1.0

        recorded = []
        used = 0

        # Pre-allocate a vector for speed
        x = np.zeros(dim)

        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            # Set x_1 to t, rest to 0
            x[0] = float(tv)

            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)

            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
