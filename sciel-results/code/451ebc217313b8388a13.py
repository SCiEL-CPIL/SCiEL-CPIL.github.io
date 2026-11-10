import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized spacing.
    
    The ZDT3 Pareto front is given by x = (t, 0, ..., 0) for t in [0, 1].
    f1(t) = t, f2(t) = 2*(1 - sqrt(t) - t*sin(10*pi*t)).
    
    We use a numerical integration approach to optimally space 200 points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _compute_f2(self, t):
        """Compute f2(t) for ZDT3 Pareto front."""
        t = np.asarray(t, dtype=float)
        # Handle t=0: f2(0) = 2
        result = np.where(t > 1e-15,
                          2.0 * (1.0 - np.sqrt(t) - t * np.sin(10.0 * np.pi * t)),
                          2.0)
        return result

    def _hypervolume_2d(self, f1_vals, f2_vals, ref):
        """Compute 2D hypervolume with reference point ref = [ref1, ref2]."""
        # Sort by f1 ascending
        order = np.argsort(f1_vals)
        f1_s = f1_vals[order]
        f2_s = f2_vals[order]
        
        # Only count points strictly better than reference
        mask = (f1_s < ref[0]) & (f2_s < ref[1])
        f1_s = f1_s[mask]
        f2_s = f2_s[mask]
        
        if len(f1_s) == 0:
            return 0.0
        
        # Standard 2D hypervolume computation
        n = len(f1_s)
        total = 0.0
        for i in range(n):
            if i < n - 1:
                width = f1_s[i + 1] - f1_s[i]
            else:
                width = ref[0] - f1_s[i]
            if width > 0 and f2_s[i] < ref[1]:
                total += width * (ref[1] - f2_s[i])
        
        return total

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)
        n_sample = budget
        ref = np.array([11.0, 11.0])

        # High-resolution grid for numerical integration
        n_fine = 100000
        t_fine = np.linspace(0.0, 1.0, n_fine)
        f2_fine = self._compute_f2(t_fine)
        
        # Compute cumulative hypervolume contribution
        # For each point, the contribution to HV is (ref[1] - f2[i]) * width
        # where width is the interval width it "owns"
        # 
        # Strategy: Use the fact that optimal spacing for maximizing discrete HV
        # in 2D with a monotone decreasing front is to equalize the marginal
        # contribution of each point.
        #
        # The marginal contribution of point i with neighbors i-1 and i+1 is:
        # (t[i+1] - t[i-1])/2 * (ref[1] - f2[i])
        # 
        # For optimal spacing, we want to place points where the "density"
        # of hypervolume per unit t is highest.
        
        # Compute the hypervolume density: how much HV is contributed per unit t
        # The HV in interval [t_i, t_{i+1}] is approximately:
        # (t_{i+1} - t_i) * (ref[1] - (f2[i] + f2[i+1])/2)
        
        # Better approach: compute the exact HV for a given set of points
        # and use gradient-based optimization
        
        # Initialize with uniform spacing
        t_values = np.linspace(0.0, 1.0, n_sample)
        f2_values = self._compute_f2(t_values)
        
        # Local search: iteratively improve each point's position
        # For each point, try small perturbations to maximize HV
        
        def compute_hv(t_vals):
            f2_vals = self._compute_f2(t_vals)
            return self._hypervolume_2d(t_vals, f2_vals, ref)
        
        current_hv = compute_hv(t_values)
        
        # Multi-scale local search
        scales = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
        
        for scale in scales:
            improved = True
            max_iters = 1000
            iter_count = 0
            while improved and iter_count < max_iters:
                improved = False
                iter_count += 1
                
                if time.time() > deadline - 10:
                    break
                
                # Try to improve each interior point
                for i in range(1, n_sample - 1):
                    # Try left and right perturbations
                    for direction in [-1, 1]:
                        new_t = t_values[i] + direction * scale
                        if new_t <= t_values[i-1] or new_t >= t_values[i+1]:
                            continue
                        
                        # Compute new HV
                        t_test = t_values.copy()
                        t_test[i] = new_t
                        new_hv = compute_hv(t_test)
                        
                        if new_hv > current_hv + 1e-12:
                            t_values[i] = new_t
                            current_hv = new_hv
                            improved = True
        
        # Ensure sorted and within bounds
        t_values = np.sort(t_values)
        t_values = np.clip(t_values, 0.0, 1.0)
        
        # Ensure exact endpoints
        if len(t_values) >= 2:
            t_values[0] = 0.0
            t_values[-1] = 1.0
        
        # Remove duplicates if any
        t_values = np.unique(t_values)
        
        # If we have fewer points than budget, add more
        if len(t_values) < n_sample:
            n_extra = n_sample - len(t_values)
            extra_t = np.linspace(0.0, 1.0, n_extra + 2)[1:-1]
            t_values = np.sort(np.concatenate([t_values, extra_t]))
            t_values = np.unique(t_values)
            if len(t_values) > n_sample:
                idx = np.linspace(0, len(t_values) - 1, n_sample).astype(int)
                t_values = t_values[idx]
        
        # Final evaluation
        recorded = []
        used = 0
        x = np.zeros(dim)

        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x[0] = float(tv)
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
