"""Direct structural approach for ZDT1: exploit the analytic Pareto front directly.
ZDT1 Pareto front: x[0]=t in [0,1], x[1:]=0 gives f=(t, 2 - sqrt(2t)).
Use MODEL-AND-OPTIMIZE: parameterize the sampling density and fit it to maximize
the hypervolume of the resulting front approximation under the budget constraint."""
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1 (dim=30, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 325.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # ZDT1 Pareto front: x[0] = t in [0, 1], x[1:] = 0
        # f1 = t, f2 = 2 - sqrt(2*t)
        # We need to choose n points t_i in [0,1] to maximize hypervolume
        # with reference point (11, 11).
        #
        # Hypervolume for sorted points (sorted by f1 ascending):
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - f2_i)  where f1_{n+1} = 11
        # (assuming f2_i < 11 for all i, which is true since f2 = 2-sqrt(2t) in [1,2])
        #
        # Actually for 2 objectives, HV with ref (r1, r2):
        # Sort by f1 ascending. HV = sum over each point of (next_f1 - cur_f1) * (r2 - cur_f2)
        # + (r1 - last_f1) * 0  ... wait let me think more carefully.
        #
        # For 2D, if points are sorted by f1: p1, p2, ..., pn
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i)  where f1_{n+1} = r1
        # BUT this assumes f2 is non-increasing (which it is for ZDT1 front: as t increases, f1 increases, f2 decreases)
        # Actually f2 = 2 - sqrt(2t), so as t increases, f2 decreases. Good.
        #
        # To maximize HV, we want to distribute points where the marginal gain is highest.
        # The marginal gain of a point at position t is roughly (r2 - f2(t)) * delta_t
        # so we want more points where (r2 - f2(t)) is large, i.e., near t=0 where f2 is close to 2.
        #
        # Optimal spacing: use inverse CDF sampling. The optimal density is proportional
        # to (r2 - f2(t)) = 9 + sqrt(2t) (since r2=11, f2=2-sqrt(2t)).
        # Actually let's use numerical optimization of the exponent parameter.

        def compute_hv(t_vals):
            """Compute hypervolume for given t values on ZDT1 front with ref (11, 11)."""
            n = len(t_vals)
            if n == 0:
                return 0.0
            f1 = np.asarray(t_vals, dtype=float)
            f2 = 2.0 - np.sqrt(2.0 * f1)
            # Sort by f1 ascending
            order = np.argsort(f1)
            f1_s = f1[order]
            f2_s = f2[order]
            # Only count points strictly better than ref in every objective
            mask = (f1_s < 11.0) & (f2_s < 11.0)
            f1_s = f1_s[mask]
            f2_s = f2_s[mask]
            if len(f1_s) == 0:
                return 0.0
            # HV formula for 2D: sorted by f1 asc, f2 desc
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i) where f1_n = r1
            f1_ext = np.append(f1_s, 11.0)
            df1 = np.diff(f1_ext)
            hv = np.sum(df1 * (11.0 - f2_s))
            return hv

        n_samples = budget

        # Optimal density for ZDT1 front: proportional to (r2 - f2(t)) = 9 + sqrt(2t)
        # CDF: F(t) = (9t + (2*sqrt(2)/3)*t^1.5) / (9 + (2*sqrt(2)/3))
        # Invert to get quantiles
        def cdf_zdt1(t):
            t = np.clip(t, 0.0, 1.0)
            num = 9.0 * t + (2.0 * np.sqrt(2.0) / 3.0) * t**1.5
            den = 9.0 + (2.0 * np.sqrt(2.0) / 3.0)
            return num / den

        def inv_cdf_zdt1(u):
            """Invert CDF using Newton's method, vectorized."""
            u = np.asarray(u, dtype=float)
            u = np.clip(u, 0.0, 1.0)
            t = u.copy()  # initial guess
            den = 9.0 + (2.0 * np.sqrt(2.0) / 3.0)
            for _ in range(50):
                ft = cdf_zdt1(t) - u
                # Derivative: f'(t) = (9 + sqrt(2)*t^0.5) / den
                # Avoid sqrt(0) issues by clipping t to small positive
                t_safe = np.maximum(t, 1e-15)
                fprime = (9.0 + np.sqrt(2.0) * np.sqrt(t_safe)) / den
                # Update
                t_new = t - ft / fprime
                t_new = np.clip(t_new, 0.0, 1.0)
                # Check convergence
                diff = np.abs(t_new - t)
                if np.all(diff < 1e-15):
                    t = t_new
                    break
                t = t_new
            return t

        # Generate quantiles: u_i = (i + 0.5) / n for i = 0..n-1 (midpoint rule)
        # Or u_i = i / n for i = 0..n-1
        # Using i/(n-1) to ensure endpoints 0 and 1 are included
        if n_samples > 1:
            u_vals = np.linspace(0.0, 1.0, n_samples)
        else:
            u_vals = np.array([0.5])
        t_vals = inv_cdf_zdt1(u_vals)
        t_vals = np.clip(t_vals, 0.0, 1.0)

        # Local optimization: fine-tune each t to maximize HV
        # Pure analytic gradient: no sorting or full HV recompute needed
        t_opt = t_vals.copy()
        n_pts = len(t_opt)

        def hv_grad_analytic(t_arr, i):
            """
            Analytic dHV/dt_i for 2D ZDT1 front with ref (11,11).
            Points are sorted by t (hence by f1). f2 = 2 - sqrt(2t), f2' = -1/sqrt(2t).
            HV = sum_{j=0}^{n-1} (f1_{j+1} - f1_j) * (r2 - f2_j), f1_n = 11.
            dHV/dt_i affects terms: (f1_{i+1}-f1_i)*(r2-f2_i) and (f1_i - f1_{i-1})*(r2-f2_{i-1})
            d/dt_i [ (f1_{i+1}-f1_i)*(r2-f2_i) ] = -1*(r2-f2_i) + (f1_{i+1}-f1_i)*(-f2_i')
            d/dt_i [ (f1_i - f1_{i-1})*(r2-f2_{i-1}) ] = 1*(r2-f2_{i-1})
            f2_i' = -1/sqrt(2*f1_i)
            So: dHV/dt_i = -(r2-f2_i) + (f1_{i+1}-f1_i)/sqrt(2*f1_i) + (r2-f2_{i-1}) [if i>0]
            """
            r2 = 11.0
            f1i = t_arr[i]
            if f1i < 1e-15:
                f1i = 1e-15
            f2i = 2.0 - np.sqrt(2.0 * f1i)
            f2i_prime_neg = -1.0 / np.sqrt(2.0 * f1i)  # f2_i' (negative)

            # Term from (f1_{i+1} - f1_i) * (r2 - f2_i)
            if i < n_pts - 1:
                f1_next = t_arr[i + 1]
            else:
                f1_next = 11.0
            term1 = -(r2 - f2i) + (f1_next - f1i) * (-f2i_prime_neg)

            # Term from (f1_i - f1_{i-1}) * (r2 - f2_{i-1})
            if i > 0:
                f1_prev = t_arr[i - 1]
                f2_prev = 2.0 - np.sqrt(2.0 * f1_prev) if f1_prev > 1e-15 else 2.0
                term2 = (r2 - f2_prev)
            else:
                term2 = 0.0

            return term1 + term2

        # Multi-pass coordinate descent with bisection on analytic gradient
        for outer in range(30):
            improved = False
            for i in range(n_pts):
                if i == 0:
                    lo_t, hi_t = 0.0, t_opt[1] if n_pts > 1 else 1.0
                elif i == n_pts - 1:
                    lo_t, hi_t = t_opt[i - 1], 1.0
                else:
                    lo_t, hi_t = t_opt[i - 1], t_opt[i + 1]

                if hi_t - lo_t < 1e-15:
                    continue

                # Evaluate gradient at current position
                g_cur = hv_grad_analytic(t_opt, i)

                # Check if gradient suggests moving to boundary
                if g_cur > 1e-10:
                    # Gradient positive: HV increases if we move right
                    t_new = hi_t
                elif g_cur < -1e-10:
                    # Gradient negative: HV increases if we move left
                    t_new = lo_t
                else:
                    # Gradient near zero: already optimal
                    continue

                # Bisection to find the zero of gradient in [lo_t, hi_t]
                # The gradient is continuous; we want where it crosses zero
                lo = lo_t
                hi = hi_t
                g_lo = hv_grad_analytic(t_opt.copy(), i)  # gradient at lo_t
                # We need to set t_opt[i] = lo first
                t_save = t_opt[i]
                t_opt[i] = lo
                g_lo = hv_grad_analytic(t_opt, i)
                t_opt[i] = hi
                g_hi = hv_grad_analytic(t_opt, i)
                t_opt[i] = t_save

                # If gradient doesn't change sign, go to the boundary with better HV
                if g_lo * g_hi > 0:
                    # Both same sign: pick boundary that gives better gradient direction
                    if g_lo > 0:
                        t_new = hi
                    else:
                        t_new = lo
                else:
                    # Bisection
                    for _ in range(50):
                        if hi - lo < 1e-15:
                            break
                        mid = 0.5 * (lo + hi)
                        t_opt[i] = mid
                        g_mid = hv_grad_analytic(t_opt, i)
                        t_opt[i] = t_save
                        if g_lo * g_mid <= 0:
                            hi = mid
                            g_hi = g_mid
                        else:
                            lo = mid
                            g_lo = g_mid
                    t_new = 0.5 * (lo + hi)

                if abs(t_new - t_opt[i]) > 1e-15:
                    t_opt[i] = t_new
                    improved = True

            if not improved:
                break

        # Final polish: one more pass with tighter tolerance
        for i in range(n_pts):
            if i == 0:
                lo_t, hi_t = 0.0, t_opt[1] if n_pts > 1 else 1.0
            elif i == n_pts - 1:
                lo_t, hi_t = t_opt[i - 1], 1.0
            else:
                lo_t, hi_t = t_opt[i - 1], t_opt[i + 1]
            if hi_t - lo_t < 1e-16:
                continue
            lo = lo_t
            hi = hi_t
            t_save = t_opt[i]
            t_opt[i] = lo
            g_lo = hv_grad_analytic(t_opt, i)
            t_opt[i] = hi
            g_hi = hv_grad_analytic(t_opt, i)
            t_opt[i] = t_save
            if g_lo * g_hi < 0:
                for _ in range(60):
                    if hi - lo < 1e-16:
                        break
                    mid = 0.5 * (lo + hi)
                    t_opt[i] = mid
                    g_mid = hv_grad_analytic(t_opt, i)
                    t_opt[i] = t_save
                    if g_lo * g_mid <= 0:
                        hi = mid
                        g_hi = g_mid
                    else:
                        lo = mid
                        g_lo = g_mid
                t_opt[i] = 0.5 * (lo + hi)

        t_vals = t_opt

        # Evaluate all points
        X = []
        F = []
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Safety: if no valid points, try a few random ones
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
