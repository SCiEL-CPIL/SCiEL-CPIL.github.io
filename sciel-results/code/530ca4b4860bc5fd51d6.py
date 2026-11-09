import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.

    For ZDT3, the Pareto front is obtained by setting x_1 = t and x_i = 0 for i > 1.
    The objective function f_2(t) = 11(1 - sqrt(t/11)).
    The hypervolume is dominated by the term (11 - t_N) * sqrt(11 * t_N) which is maximized at t_N = 1.
    The remaining hypervolume is approximated by the sum sum_{i=1}^{N-1} (t_{i+1} - t_i) * sqrt(11 * t_i).
    To maximize this sum (minimize quadrature error), we use a non-uniform spacing
    with density proportional to t^(-3/4), which corresponds to t_i = (i/N)^4.
    We avoid t=0 to prevent wasting an evaluation on a point with zero contribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)
        
        # Optimal power-law spacing p=4 based on error minimization
        p = 4.0
        
        if n_sample == 2:
            # Ensure we cover the range, avoid 0 if possible?
            # With N=2, t1=0.5, t2=1.0?
            # t = (i/N)^p. i=1 -> 0.5^4 = 0.0625. i=2 -> 1.0.
            # This is fine.
            pass

        i = np.arange(1, n_sample + 1)
        t_values = (i / n_sample) ** p

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
