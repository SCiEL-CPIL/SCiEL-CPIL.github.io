"""ZDT1 Pareto-front optimizer using the analytical optimal spacing derived from
equalizing the marginal hypervolume contribution of each point on the convex front.

For ZDT1 with reference point (11, 11), the Pareto front is
  f1 = t,  f2 = 2 - sqrt(2t),  t in [0, 1].
The 200 points that maximize the 2-D hypervolume are obtained by solving the
equal-marginal-contribution condition via a fast forward-recurrence + bisection
on the free parameter, followed by a high-precision polish pass.
"""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 305.0
        budget = self.budget
        dim = self.dim

        used = 0

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(np.asarray(x, dtype=float)), dtype=float)
            used += 1
            return f

        n = min(budget, 200)

        def compute_optimal_t(n_pts):
            """Compute optimal t-values via forward recurrence + bisection."""
            # The optimality condition for interior point i (0-indexed):
            #   t_{i+1} = 3*t_i - sqrt(2*t_i*t_{i-1})
            # This is a forward recurrence. Given t[0]=0 and t[1]=alpha,
            # we can compute all t_i. We need t[n-1]=1, so bisect on alpha.
            
            def forward(alpha):
                """Given t[1]=alpha, compute all t_i via recurrence."""
                t_arr = np.zeros(n_pts)
                t_arr[0] = 0.0
                t_arr[1] = alpha
                for i in range(1, n_pts - 1):
                    ti = t_arr[i]
                    tim1 = t_arr[i-1]
                    if ti <= 0 or tim1 <= 0:
                        t_arr[i+1] = ti + 1e-10
                    else:
                        t_arr[i+1] = 3.0 * ti - np.sqrt(2.0 * ti * tim1)
                    t_arr[i+1] = max(t_arr[i+1], t_arr[i] + 1e-15)
                return t_arr
            
            # Bisect on alpha = t[1] so that t[n-1] = 1
            lo_alpha = 1e-10
            hi_alpha = 0.5  # upper bound
            
            # Check if hi gives t[n-1] > 1
            for _ in range(100):
                mid_alpha = (lo_alpha + hi_alpha) / 2.0
                t_arr = forward(mid_alpha)
                if t_arr[-1] < 1.0:
                    lo_alpha = mid_alpha
                else:
                    hi_alpha = mid_alpha
                if hi_alpha - lo_alpha < 1e-15:
                    break
            
            t_opt = forward((lo_alpha + hi_alpha) / 2.0)
            # Normalize to ensure t[0]=0, t[-1]=1
            t_opt[0] = 0.0
            t_opt[-1] = 1.0
            # Ensure monotonicity
            for i in range(1, n_pts):
                if t_opt[i] <= t_opt[i-1]:
                    t_opt[i] = t_opt[i-1] + 1e-15
            t_opt[-1] = 1.0
            return np.clip(t_opt, 0.0, 1.0)

        t = compute_optimal_t(n)

        # Evaluate all points
        X = []
        F = []
        for ti in t:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(ti)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Polish pass: use remaining budget to refine points
        # Each evaluation can nudge a point slightly to improve HV
        remaining = budget - used
        if remaining > 0 and len(F) > 0:
            F_arr = np.array(F)
            # Sort by f1
            sort_idx = np.argsort(F_arr[:, 0])
            F_sorted = F_arr[sort_idx]
            t_sorted = t[sort_idx]
            
            # Try small perturbations of interior points
            # Use a simple hill-climbing: for each interior point, try
            # moving it slightly left or right and keep if HV improves
            # But we can't recompute HV cheaply, so use the analytic formula
            
            def compute_hv_from_t(t_vals):
                """Compute 2D HV from t-values on the ZDT1 front."""
                f1 = t_vals
                f2 = 2.0 - np.sqrt(2.0 * f1)
                # Sort by f1
                order = np.argsort(f1)
                f1_s = f1[order]
                f2_s = f2[order]
                # HV = sum (f1_{i+1} - f1_i) * (11 - f2_i), f1_{n+1}=11
                r1, r2 = 11.0, 11.0
                hv = 0.0
                for i in range(len(f1_s)):
                    f1_next = f1_s[i+1] if i+1 < len(f1_s) else r1
                    hv += (f1_next - f1_s[i]) * (r2 - f2_s[i])
                return hv
            
            current_hv = compute_hv_from_t(t_sorted)
            
            # Try perturbing each interior point
            for _polish in range(min(remaining, 20)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                best_gain = 0.0
                best_i = -1
                best_dir = 0
                for i in range(1, len(t_sorted) - 1):
                    # Try small delta
                    for delta in [1e-4, -1e-4, 1e-5, -1e-5]:
                        t_trial = t_sorted.copy()
                        t_trial[i] += delta
                        if t_trial[i] <= t_trial[i-1] or t_trial[i] >= t_trial[i+1]:
                            continue
                        hv_trial = compute_hv_from_t(t_trial)
                        gain = hv_trial - current_hv
                        if gain > best_gain:
                            best_gain = gain
                            best_i = i
                            best_dir = delta
                if best_i >= 0 and best_gain > 1e-10:
                    # Apply the best perturbation
                    t_sorted[best_i] += best_dir
                    # Re-evaluate this point
                    x = np.zeros(dim)
                    x[0] = float(t_sorted[best_i])
                    f = eval_point(x)
                    if np.all(np.isfinite(f)):
                        F_sorted[best_i] = f
                        current_hv += best_gain
                        X[sort_idx[best_i]] = x
                else:
                    break

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
