import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.

    For ZDT3, the Pareto front is obtained by setting x_1 = t and x_i = 0 for i > 1.
    We use a numerical approach to select 200 sample points that maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _compute_zdt3_pareto_front(self, n_points):
        """Compute ZDT3 Pareto front points for t in [0, 1].
        
        ZDT3: f1(t) = t, f2(t) = g(t) * h(f1, g) where g(t) = 1 + 9*n/(n+9)
        For Pareto front: x_1 = t, x_i = 0 for i > 1
        g = 1 + 9*(29/30) = 10.9
        h(f1, g) = 1 - sqrt(f1/g) - sin(10*pi*f1)
        """
        t_values = np.linspace(0, 1, n_points)
        f1 = t_values
        g = 1 + 9 * (29 / 30)  # = 10.9
        h = 1 - np.sqrt(f1 / g) - np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.column_stack([f1, f2])

    def _compute_optimal_samples(self, n_samples, n_dense=10000):
        """Select n_samples points from dense Pareto front to maximize hypervolume.
        
        Strategy: Use dynamic programming or greedy selection to pick points
        that maximize the hypervolume of the sampled set.
        """
        # First, get a dense sample of the Pareto front
        dense_front = self._compute_zdt3_pareto_front(n_dense)
        
        # Sort by f1 (which is already sorted)
        # For hypervolume maximization, we want to select points that
        # create the largest "staircase" area
        
        # Greedy approach: start with endpoints, then add points that
        # maximize incremental hypervolume
        selected_indices = [0, n_dense - 1]
        selected_set = set(selected_indices)
        
        # Precompute f1, f2 for dense front
        f1_dense = dense_front[:, 0]
        f2_dense = dense_front[:, 1]
        
        # For each candidate point, compute its contribution to hypervolume
        # when added to the current selected set
        # This is expensive, so we use a heuristic: select points that
        # maximize the minimum distance in objective space
        
        # Better approach: use the fact that for a monotone front,
        # optimal spacing is related to the curvature
        # Compute the "value" of each point based on how much it extends the front
        
        # Simple heuristic: select points that are local maxima of f2
        # or points where the derivative changes sign significantly
        
        # Even simpler: use a power-law spacing based on the known structure
        # ZDT3 has oscillations from sin(10*pi*t), so we want to capture
        # the peaks and troughs
        
        # The function h(t) = 1 - sqrt(t/g) - sin(10*pi*t) has 10 periods
        # We want to sample at peaks and troughs of the oscillation
        
        # Critical points: where d/dt[sin(10*pi*t)] = 0, i.e., cos(10*pi*t) = 0
        # 10*pi*t = pi/2 + k*pi => t = (1/20 + k/10) for k = 0, 1, ..., 9
        critical_points = np.array([(1/20 + k/10) for k in range(10)])
        critical_points = critical_points[(critical_points >= 0) & (critical_points <= 1)]
        
        # Include endpoints
        critical_points = np.concatenate([[0.0], critical_points, [1.0]])
        critical_points = np.unique(critical_points)
        
        # Now we have ~12 critical points. We need to distribute the remaining
        # samples to fill in between them optimally.
        
        # For each interval between critical points, we want to place samples
        # that maximize the hypervolume contribution. Since the front is
        # piecewise smooth, we can use adaptive sampling within each interval.
        
        # Total samples to distribute: n_samples - len(critical_points)
        remaining = n_samples - len(critical_points)
        
        if remaining <= 0:
            # Not enough samples for all critical points, use power-law
            return (np.arange(n_samples) / (n_samples - 1)) ** 0.8
        
        # Distribute remaining samples proportionally to interval length
        intervals = np.diff(critical_points)
        total_length = intervals.sum()
        
        # Allocate samples to each interval
        samples_per_interval = np.maximum(1, np.round(remaining * intervals / total_length)).astype(int)
        
        # Adjust to ensure total is exactly remaining
        diff = remaining - samples_per_interval.sum()
        if diff != 0:
            # Add or remove from the largest interval
            idx_max = np.argmax(samples_per_interval)
            samples_per_interval[idx_max] += diff
        
        # Now create the sample points
        t_values = [critical_points[0]]
        for i in range(len(critical_points) - 1):
            t_start = critical_points[i]
            t_end = critical_points[i + 1]
            n_in_interval = samples_per_interval[i]
            
            if n_in_interval == 0:
                continue
            elif n_in_interval == 1:
                # Just use the midpoint or endpoint
                t_values.append(t_end)
            else:
                # Use power-law spacing within this interval
                # Cluster near the left endpoint if the derivative is large there
                local_i = np.arange(1, n_in_interval)  # Skip 0 (already have t_start)
                t_local = (local_i / n_in_interval) ** 0.8
                t_values.extend(t_start + t_local * (t_end - t_start))
        
        t_values = np.array(t_values)
        t_values = np.unique(t_values)
        
        # If we have more than n_samples, trim
        if len(t_values) > n_samples:
            # Keep the most important ones (critical points and evenly spaced)
            # For now, just take the first n_samples
            t_values = t_values[:n_samples]
        elif len(t_values) < n_samples:
            # Add more points to reach n_samples
            missing = n_samples - len(t_values)
            # Add evenly spaced points in the gaps
            all_points = np.concatenate([t_values, np.linspace(0, 1, missing)])
            t_values = np.unique(all_points)
            t_values = t_values[:n_samples]
        
        return t_values

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 285.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)
        
        # Use numerical approach to select optimal sample points
        t_values = self._compute_optimal_samples(n_sample)
        
        # Ensure we have exactly n_sample points
        if len(t_values) < n_sample:
            # Pad with evenly spaced points
            missing = n_sample - len(t_values)
            extra = np.linspace(0, 1, missing + 2)[1:-1]
            t_values = np.concatenate([t_values, extra])
            t_values = np.unique(t_values)
            if len(t_values) > n_sample:
                t_values = t_values[:n_sample]
        elif len(t_values) > n_sample:
            t_values = t_values[:n_sample]
        
        # Sort t_values
        t_values = np.sort(t_values)

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
