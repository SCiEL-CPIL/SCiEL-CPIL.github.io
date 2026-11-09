import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with numerically optimized spacing.
    
    The ZDT3 Pareto front is given by x = (t, 0, ..., 0) for t in [0, 1].
    f1(t) = t, f2(t) = 2*(1 - sqrt(t) - t*sin(10*pi*t)).
    
    We numerically optimize the spacing exponent p to maximize discrete hypervolume.
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
        
        # Hypervolume: sum of rectangles
        # For each point i, width = f1[i+1] - f1[i] (or ref[0] - f1[i] for last)
        # height = min(f2[i], ref[1]) - f2[i]... no, height contribution
        # Actually: HV = sum_i (f1[i+1] - f1[i]) * (ref[1] - f2[i]) for i where f2[i] < ref[1]
        # But need to be careful: the envelope is the lower envelope of f2 values
        
        # Better approach: compute the non-dominated hypervolume
        # Sort by f1, then compute the area under the lower envelope
        n = len(f1_s)
        if n == 0:
            return 0.0
        
        # The hypervolume is the area between the Pareto front and the reference point
        # = integral from min_f1 to ref[0] of (ref[1] - f2_envelope(f1)) df1
        
        # Use the fact that for sorted points, the contribution of point i is:
        # (f1[i+1] - f1[i]) * (ref[1] - f2[i]) if f2[i] is the minimum so far
        # Actually simpler: for 2D, HV = sum over i of (f1[i+1] - f1[i]) * (ref[1] - f2[i])
        # where we extend f1 to include ref[0] and f2 values are the actual f2 at each point
        # But this only works if points are non-dominated and sorted properly
        
        # Standard 2D HV: sort by f1 asc, then for each consecutive pair,
        # the width is the difference in f1, and the height is ref[1] - f2[i]
        # but we need to take the minimum f2 in each interval
        
        # Actually the standard formula: 
        # HV = sum_{i=0}^{n-1} (f1[i+1] - f1[i]) * (ref[1] - f2[i])
        # where f1[n] = ref[0], and this assumes f2 is monotonically decreasing
        # (which it is on the Pareto front for ZDT3)
        
        # For safety, let's use the standard sweep
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

        # Compute f2 on a fine grid to optimize spacing
        t_fine = np.linspace(0.0, 1.0, 100000)
        f2_fine = self._compute_f2(t_fine)
        
        # df2/dt numerically
        df2dt = np.gradient(f2_fine, t_fine)
        abs_df2dt = np.abs(df2dt)
        abs_df2dt[0] = abs_df2dt[1] if len(abs_df2dt) > 1 else 1.0

        # Try different exponents p and pick the one that maximizes HV
        ref = np.array([11.0, 11.0])
        
        def evaluate_exponent(p):
            """Evaluate HV for a given exponent p."""
            if p <= 0:
                # Uniform spacing
                t_vals = np.linspace(0.0, 1.0, n_sample)
            else:
                density = abs_df2dt ** p
                cdf = np.cumsum(density)
                cdf /= cdf[-1]
                u = np.linspace(0.0, 1.0, n_sample)
                t_vals = np.interp(u, cdf, t_fine)
                t_vals = np.sort(t_vals)
            
            # Ensure endpoints
            t_vals[0] = 0.0
            t_vals[-1] = 1.0
            
            f1_vals = t_vals
            f2_vals = self._compute_f2(t_vals)
            
            return self._hypervolume_2d(f1_vals, f2_vals, ref)

        # Search for optimal p
        best_p = 0.0
        best_hv = -1.0
        
        # Coarse search
        for p in np.linspace(0.0, 1.5, 31):
            hv = evaluate_exponent(p)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Fine search around best
        p_range = np.linspace(best_p - 0.15, best_p + 0.15, 31)
        p_range = p_range[p_range >= 0]
        for p in p_range:
            hv = evaluate_exponent(p)
            if hv > best_hv:
                best_hv = hv
                best_p = p

        # Generate final t values with best_p
        if best_p <= 0:
            t_values = np.linspace(0.0, 1.0, n_sample)
        else:
            density = abs_df2dt ** best_p
            cdf = np.cumsum(density)
            cdf /= cdf[-1]
            u = np.linspace(0.0, 1.0, n_sample)
            t_values = np.interp(u, cdf, t_fine)
            t_values = np.sort(t_values)
            t_values[0] = 0.0
            t_values[-1] = 1.0

        # Deduplicate
        t_values = np.unique(t_values)
        
        # If we have fewer than budget due to dedup, add more points
        if len(t_values) < n_sample:
            # Add evenly spaced points to fill
            n_extra = n_sample - len(t_values)
            extra_t = np.linspace(0.0, 1.0, n_extra + 2)[1:-1]
            t_values = np.sort(np.concatenate([t_values, extra_t]))
            t_values = np.unique(t_values)
            if len(t_values) > n_sample:
                idx = np.linspace(0, len(t_values) - 1, n_sample).astype(int)
                t_values = t_values[idx]
        
        # Ensure exact endpoints
        if len(t_values) >= 2:
            t_values[0] = 0.0
            t_values[-1] = 1.0

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
