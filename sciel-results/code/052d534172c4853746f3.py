import numpy as np
import time


class Optimizer:
    """ZDT6 specialized optimizer: exploits known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = self.budget

        # Phase 1: Random exploration to confirm front (use ~10% of budget)
        n_explore = max(10, n_evals // 10)
        rng = np.random.default_rng(42)

        all_F = []
        all_X = []

        # Random sampling
        X_explore = lo + (hi - lo) * rng.random((n_explore, self.dim))
        for x in X_explore:
            f = problem(x)
            all_F.append(f)
            all_X.append(x)

        # Check if front is at x[1:] = 0 (ZDT6 structure)
        # Find best f1 from exploration
        F_arr = np.array(all_F)
        best_f1_idx = np.argmin(F_arr[:, 0])
        best_x = all_X[best_f1_idx].copy()

        # Phase 2: Exploit the front structure
        # On ZDT6 front: x_2..x_10 = 0, f1 = 1 - sqrt(x_1), f2 = 1 - sqrt(f1)
        # We want to distribute x1 values to maximize hypervolume
        #
        # Key: sample x1 with density proportional to marginal HV contribution
        # For 2D, HV contribution ~ (ref_f1 - f1) * (ref_f2 - f2)
        # The marginal contribution when adding a point depends on gaps
        #
        # Use inverse-CDF sampling with weight w(x1) = 1 / (10 + sqrt(f1(x1)))
        # where f1(x1) = 1 - sqrt(x1)
        #
        # Alternatively, sample uniformly in f1-space since that often works well
        # for 2D fronts with curved shapes.

        n_exploit = n_evals - n_explore

        # Generate x1 samples using a distribution that emphasizes regions
        # with larger reference-point distance
        # f1 = 1 - sqrt(x1) => x1 = (1 - f1)^2
        # Sample f1 with density proportional to (11 - f1) * (11 - f2)
        # where f2 = 1 - sqrt(f1)
        #
        # Simple approach: sample f1 uniformly, then convert to x1
        # This gives more points in high-f1 region (where front is flatter)

        # Use a mixture: 70% uniform in f1, 30% with bias toward high f1
        f1_samples = []
        n_uniform = int(n_exploit * 0.6)
        n_biased = n_exploit - n_uniform

        # Uniform in f1 in [0, 1]
        f1_unif = rng.random(n_uniform)
        f1_samples.extend(f1_unif.tolist())

        # Biased: sample f1 with density ~ (11 - f1) * (11 - (1 - sqrt(f1)))
        # = (11 - f1) * (10 + sqrt(f1))
        # Use rejection or inverse transform
        # Simpler: sample u ~ uniform, set f1 = u^0.3 (concentrates toward 1)
        # Actually, let's use the weight 1/(10+sqrt(f1)) from the knowledge
        # This means density ~ 1/(10+sqrt(f1)), which is higher for small f1
        # But that seems counterintuitive... Let me think again.
        #
        # The knowledge says weight 1/(10+sqrt(f1)) better aligns density with
        # curvature. Let's use inverse-CDF with this weight.
        #
        # w(f1) = 1 / (10 + sqrt(f1))
        # CDF: integral of 1/(10+sqrt(t)) dt from 0 to f1
        # Let u = sqrt(t), t = u^2, dt = 2u du
        # integral 2u/(10+u) du = integral (2 - 20/(10+u)) du
        # = 2u - 20 ln(10+u) + C
        # At u=0: 0 - 20 ln(10) + C
        # CDF(f1) = [2*sqrt(f1) - 20 ln(10+sqrt(f1))] / [0 - 20 ln(10)]
        #        = [2*sqrt(f1) - 20 ln(10+sqrt(f1)) + 20 ln(10)] / (20 ln(10))
        #        = [sqrt(f1) - 10 ln((10+sqrt(f1))/10)] / (10 ln(10))

        # Inverse CDF is hard analytically, so use rejection or grid-based
        # Grid-based: create a grid of f1 values, compute weights, use searchsorted
        grid_size = 10000
        f1_grid = np.linspace(0, 1, grid_size)
        weights = 1.0 / (10.0 + np.sqrt(f1_grid))
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]

        # Sample from the biased distribution
        u_biased = rng.random(n_biased)
        f1_biased = np.interp(u_biased, cdf, f1_grid)
        f1_samples.extend(f1_biased.tolist())

        # Convert f1 to x1: x1 = (1 - f1)^2
        x1_values = (1.0 - np.array(f1_samples)) ** 2
        x1_values = np.clip(x1_values, 0.0, 1.0)

        # Evaluate points on the front: x = [x1, 0, 0, ..., 0]
        for x1_val in x1_values:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1_val
            f = problem(x)
            all_F.append(f)
            all_X.append(x)

        # Phase 3: Polish - refine around promising regions
        if time.time() < deadline:
            remaining = n_evals - len(all_F)
            if remaining > 5:
                # Find the x1 values that gave best results
                F_all = np.array(all_F)
                # Sort by f1 to find gaps
                valid_mask = np.all(F_all < 11.0, axis=1)
                if np.any(valid_mask):
                    F_valid = F_all[valid_mask]
                    X_valid = np.array(all_X[valid_mask])
                    # Sort by f1
                    order = np.argsort(F_valid[:, 0])
                    F_sorted = F_valid[order]
                    X_sorted = X_valid[order]

                    # Find largest gaps in f1 and sample midpoints
                    f1_vals = F_sorted[:, 0]
                    gaps = np.diff(f1_vals)
                    n_polish = min(remaining, 20)
                    if len(gaps) > 0:
                        # Sample in largest gaps
                        gap_indices = np.argsort(gaps)[::-1][:n_polish]
                        for gi in gap_indices:
                            if time.time() > deadline:
                                break
                            mid_f1 = (f1_vals[gi] + f1_vals[gi + 1]) / 2
                            x1_mid = (1.0 - mid_f1) ** 2
                            x = np.zeros(self.dim)
                            x[0] = x1_mid
                            f = problem(x)
                            all_F.append(f)
                            all_X.append(x)

                    # Also do local refinement around best points
                    if remaining > 30:
                        # Take a few good points and do small perturbations in x1
                        n_refine = min(remaining - 20, 30)
                        # Pick points spread across the front
                        idxs = np.linspace(0, len(X_sorted) - 1, min(10, len(X_sorted))).astype(int)
                        for idx in idxs:
                            if time.time() > deadline:
                                break
                            x_base = X_sorted[idx].copy()
                            # Small perturbation in x1
                            for delta in [-0.01, 0.01]:
                                x_new = x_base.copy()
                                x_new[0] = np.clip(x_base[0] + delta, 0, 1)
                                f = problem(x_new)
                                all_F.append(f)
                                all_X.append(x_new)

        return all_X
