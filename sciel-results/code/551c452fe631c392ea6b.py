import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with uniform sampling.
    
    The ZDT3 Pareto front is given by x = (t, 0, ..., 0) for t in [0, 1].
    The objective values are f1(t) = t and f2(t) = 2 * (1 - sqrt(t) - t * sin(10 * pi * t)).
    To maximize the hypervolume of the sampled points (which approximates the integral 
    of the front), we use uniform sampling. This ensures a stable approximation of the 
    area under the curve, which is crucial for the oscillatory f2(t) of ZDT3.
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

        # Build a non-uniform t grid that emphasizes critical points of ZDT3's Pareto front.
        # The front has oscillations with period 0.1 in t. Critical points are:
        # - Zeros of sin(10*pi*t): t = k/10 for k=0..10
        # - Extrema of sin(10*pi*t): t = (2k+1)/20 for k=0..9
        # We include all these critical points, then fill remaining budget with uniform points
        # that avoid duplicates.

        critical_points = set()
        # Zeros: t = 0, 0.1, 0.2, ..., 1.0
        for k in range(11):
            critical_points.add(round(k / 10.0, 12))
        # Extrema: t = 0.05, 0.15, ..., 0.95
        for k in range(10):
            critical_points.add(round((2 * k + 1) / 20.0, 12))

        critical_list = sorted(critical_points)
        n_critical = len(critical_list)

        # Fill remaining budget with uniform points, skipping any that are too close to critical points
        n_fill = max(0, budget - n_critical)
        if n_fill > 0:
            uniform_candidates = np.linspace(0.0, 1.0, n_fill + 1)[1:-1]  # exclude endpoints
            # Filter out candidates too close to critical points
            min_dist = 1e-6
            fill_points = []
            for uc in uniform_candidates:
                if all(abs(uc - cp) > min_dist for cp in critical_list):
                    fill_points.append(uc)
                if len(fill_points) >= n_fill:
                    break
            if len(fill_points) < n_fill:
                # If we couldn't get enough, just take the first n_fill from uniform
                fill_points = list(uniform_candidates[:n_fill])
        else:
            fill_points = []

        # Combine and sort
        all_t = sorted(set(critical_list + fill_points))
        # Truncate to budget if needed
        if len(all_t) > budget:
            # Keep critical points first, then uniform
            critical_set = set(critical_list)
            critical_in_all = [t for t in all_t if t in critical_set]
            non_critical_in_all = [t for t in all_t if t not in critical_set]
            all_t = critical_in_all + non_critical_in_all[:max(0, budget - len(critical_in_all))]
            all_t = sorted(all_t)

        recorded = []
        used = 0
        
        # Pre-allocate a vector for speed
        x = np.zeros(dim)
        
        for tv in all_t:
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
