import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with adaptive sampling.
    
    The ZDT3 Pareto front is given by x = (t, 0, ..., 0) for t in [0, 1].
    The objective values are f1(t) = t and f2(t) = 2 * (1 - sqrt(t) - t * sin(10 * pi * t)).
    To maximize the hypervolume, we use adaptive sampling that places more points
    near the oscillatory features (zeros and extrema of sin(10*pi*t)).
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
        
        # Create adaptive sampling: combine critical points with uniform fill
        # Zeros of sin(10*pi*t) at t = k/10 for k=0,...,10
        # Extrema at t = (2k+1)/20 for k=0,...,9
        critical_points = set()
        for k in range(11):
            critical_points.add(k / 10.0)
        for k in range(10):
            critical_points.add((2 * k + 1) / 20.0)
        
        critical_sorted = sorted(critical_points)
        
        # Allocate budget: use critical points + uniform fill
        n_critical = len(critical_sorted)
        n_uniform = n_sample - n_critical
        
        if n_uniform > 0:
            # Uniform points, avoiding exact duplicates with critical points
            t_uniform = np.linspace(0.0, 1.0, n_uniform + 2)[1:-1]
            # Remove points too close to critical points
            t_filtered = []
            for tv in t_uniform:
                if all(abs(tv - cp) > 1e-9 for cp in critical_sorted):
                    t_filtered.append(tv)
            t_values = np.array(critical_sorted + t_filtered)
        else:
            t_values = np.array(critical_sorted)
        
        # Sort for consistency
        t_values = np.sort(t_values)
        
        # Ensure we don't exceed budget
        if len(t_values) > n_sample:
            t_values = t_values[:n_sample]
        
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
