<<<<<<< SEARCH
        # Try different power-law spacings and pick the best
        # t_i = (i/(n-1))^p for p > 0, which concentrates near 0 for p > 1

        def compute_hv(t_vals):
            """Compute hypervolume for given t values on ZDT1 front with ref (11, 11)."""
            n = len(t_vals)
            if n == 0:
                return 0.0
            f1 = t_vals.copy()
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
            # Extend with reference point
            f1_ext = np.append(f1_s, 11.0)
            f2_ext = f2_s  # doesn't need extension for the formula
            # HV = sum_{i=0}^{n-1} (f1_ext[i+1] - f1_ext[i]) * (11 - f2_ext[i])
            df1 = np.diff(f1_ext)
            hv = np.sum(df1 * (11.0 - f2_ext))
            return hv

        # Also compute HV for a dense reference sample to get the normalization
        n_dense = 10000
        t_dense = np.linspace(0, 1, n_dense)
        # Add t=0 point
        t_dense = np.concatenate([[0.0], t_dense])
        hv_ref = compute_hv(t_dense)

        # Now optimize the power parameter p for the spacing
        # Try a range of p values
        best_p = 1.0
        best_hv = 0.0
        n_samples = budget

        # Use finer grid for p optimization
        for p in np.linspace(0.5, 3.0, 100):
            idx = np.arange(n_samples, dtype=float)
            if n_samples > 1:
                t_vals = (idx / (n_samples - 1)) ** p
            else:
                t_vals = np.array([0.0])
            # Ensure endpoints
            t_vals[0] = 0.0
            t_vals[-1] = 1.0
            hv = compute_hv(t_vals)
            if hv > best_hv:
                best_hv = hv
                best_p = p

        # Refine around best_p
        p_range = np.linspace(max(0.1, best_p - 0.1), best_p + 0.1, 50)
        for p in p_range:
            idx = np.arange(n_samples, dtype=float)
            if n_samples > 1:
                t_vals = (idx / (n_samples - 1)) ** p
            else:
                t_vals = np.array([0.0])
            t_vals[0] = 0.0
            t_vals[-1] = 1.0
            hv = compute_hv(t_vals)
            if hv > best_hv:
                best_hv = hv
                best_p = p

        # Generate the final sample using optimized p
        idx = np.arange(n_samples, dtype=float)
        if n_samples > 1:
            t_vals = (idx / (n_samples - 1)) ** best_p
        else:
            t_vals = np.array([0.0])
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
=======
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
            """Invert CDF using Newton's method."""
            u = np.clip(u, 0.0, 1.0)
            t = u  # initial guess
            for _ in range(50):
                ft = cdf_zdt1(t) - u
                # Derivative: f'(t) = (9 + sqrt(2)*t^0.5) / den
                den = 9.0 + (2.0 * np.sqrt(2.0) / 3.0)
                fprime = (9.0 + np.sqrt(2.0) * np.sqrt(t)) / den
                if fprime < 1e-12:
                    break
                t_new = t - ft / fprime
                t_new = np.clip(t_new, 0.0, 1.0)
                if np.abs(t_new - t) < 1e-15:
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
        # Use a simple coordinate descent / local search
        t_opt = t_vals.copy()
        for _ in range(5):
            improved = False
            for i in range(len(t_opt)):
                if i == 0:
                    lo_t, hi_t = 0.0, t_opt[1] if len(t_opt) > 1 else 1.0
                elif i == len(t_opt) - 1:
                    lo_t, hi_t = t_opt[i-1], 1.0
                else:
                    lo_t, hi_t = t_opt[i-1], t_opt[i+1]
                # Try a few perturbations
                best_local = t_opt[i]
                best_hv_local = compute_hv(t_opt)
                for frac in [0.25, 0.5, 0.75]:
                    t_try = lo_t + frac * (hi_t - lo_t)
                    t_try = np.clip(t_try, 0.0, 1.0)
                    t_test = t_opt.copy()
                    t_test[i] = t_try
                    hv_test = compute_hv(t_test)
                    if hv_test > best_hv_local:
                        best_hv_local = hv_test
                        best_local = t_try
                if best_local != t_opt[i]:
                    t_opt[i] = best_local
                    improved = True
            if not improved:
                break
        t_vals = t_opt
>>>>>>> REPLACE

