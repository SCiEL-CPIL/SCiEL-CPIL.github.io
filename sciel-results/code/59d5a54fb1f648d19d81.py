"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """
    Hybrid optimizer combining (1+1)-ES with periodic Gaussian Process (GP) surrogate refinement.
    - Uses (1+1)-ES with adaptive step size for robust global exploration.
    - Periodically fits a GP model (Matérn 5/2 kernel) to recent history.
    - Maximizes an Expected Improvement (EI) acquisition function to propose a new candidate.
    - If the GP-proposed point improves the best, adopt it; otherwise continue with ES.
    - Time-guarded to stay well within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize at origin
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # ES parameters
        sigma = 2.5
        min_sigma = 1e-6
        max_sigma = 10.0
        growth = 1.15
        decay = 0.92
        no_improve = 0

        # GP History
        hist_X = [x.copy()]
        hist_f = [fx]
        max_hist = 30
        min_hist_for_gp = 12  # Minimum points for a reasonable GP fit

        calls = 1

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # 1. Try GP refinement periodically
            # We try GP every 10 steps, but only if we have enough history and haven't just done it
            if i >= 10 and (i % 10 == 0) and len(hist_X) >= min_hist_for_gp:
                gp_x, gp_f = self._gp_refinement(func, best_x, best_f, hist_X, hist_f, d, rng)
                if gp_x is not None:
                    calls += 1
                    if gp_f < best_f:
                        best_f = gp_f
                        best_x = gp_x.copy()
                        # Update current position to the new best to continue from there
                        x = best_x.copy()
                        fx = best_f
                        no_improve = 0
                    # Add to history
                    hist_X.append(gp_x)
                    hist_f.append(gp_f)
                    if len(hist_X) > max_hist:
                        hist_X.pop(0)
                        hist_f.pop(0)
                    continue

            # 2. Standard (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1

            # Update history
            hist_X.append(y)
            hist_f.append(fy)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * growth, max_sigma)
                no_improve = 0
            else:
                sigma = max(sigma * decay, min_sigma)
                no_improve += 1
                # Stagnation: occasionally try a larger jump
                if no_improve >= 25:
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0

            # Update best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f

    def _gp_refinement(self, func, best_x, best_f, hist_X, hist_f, d, rng):
        """
        Fit a GP model to recent history and find a point maximizing Expected Improvement.
        Returns (x_candidate, f_candidate) if successful, else (None, None).
        """
        try:
            X = np.array(hist_X)
            y = np.array(hist_f)
            n = len(X)

            # Normalize y for numerical stability in GP
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std < 1e-10:
                y_std = 1.0
            y_norm = (y - y_mean) / y_std

            # GP Parameters
            # Matérn 5/2 kernel: k(r) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
            # We use a fixed lengthscale for simplicity, or could adapt it.
            # For d=5, a lengthscale of ~1-2 is reasonable given box size 10.
            lengthscale = 1.5
            sigma_f = 1.0  # Signal variance
            noise = 1e-6   # Small noise for numerical stability

            # Build Kernel Matrix K
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    r = np.linalg.norm(X[i] - X[j]) / lengthscale
                    K[i, j] = sigma_f**2 * (1 + np.sqrt(5) * r + (5.0 / 3.0) * r**2) * np.exp(-np.sqrt(5) * r)
            K[np.diag_indices(n)] += noise

            # Cholesky decomposition for stable inversion
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add more jitter
                K[np.diag_indices(n)] += 1e-4
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return None, None

            # Solve K alpha = y_norm
            # alpha = K^{-1} y
            # K = L L^T => L L^T alpha = y => L^T v = y => L v = y? No.
            # L w = y => w = L^{-1} y
            # L^T alpha = w => alpha = (L^T)^{-1} w
            w = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, w)

            # Define EI acquisition function
            # EI(x) = (mu(x) - best_f_norm - xi) * Phi(Z) + sigma(x) * phi(Z)
            # where Z = (mu(x) - best_f_norm - xi) / sigma(x)
            # We work in normalized space for the GP, but compare to best_f in original space?
            # It's cleaner to work in original space for EI calculation if we transform mean/std back.
            # mu_orig(x) = y_mean + y_std * mu_norm(x)
            # sigma_orig(x) = y_std * sigma_norm(x)

            best_f_norm = (best_f - y_mean) / y_std
            xi = 0.01  # Exploration parameter

            def ei(x_new):
                # Compute K(x_new, X)
                k_star = np.zeros(n)
                for i in range(n):
                    r = np.linalg.norm(x_new - X[i]) / lengthscale
                    k_star[i] = sigma_f**2 * (1 + np.sqrt(5) * r + (5.0 / 3.0) * r**2) * np.exp(-np.sqrt(5) * r)

                # Mean and Variance
                # mu_norm = k_star^T alpha
                mu_norm = np.dot(k_star, alpha)

                # var_norm = k(x_new, x_new) - k_star^T K^{-1} k_star
                # k(x_new, x_new) = sigma_f^2 (since r=0)
                k_xx = sigma_f**2

                # v = L^{-1} k_star
                v = np.linalg.solve(L, k_star)
                var_norm = k_xx - np.dot(k_star, v)
                var_norm = max(var_norm, 1e-10)
                std_norm = np.sqrt(var_norm)

                # Transform to original scale
                mu_orig = y_mean + y_std * mu_norm
                std_orig = y_std * std_norm

                if std_orig < 1e-8:
                    # If variance is tiny, EI is just improvement in mean
                    improvement = mu_orig - best_f - xi
                    return max(improvement, 0.0)

                Z = (mu_orig - best_f - xi) / std_orig
                EI = (mu_orig - best_f - xi) * norm.cdf(Z) + std_orig * norm.pdf(Z)
                return EI

            # Maximize EI using a local search from a few starting points
            # Starting points: current best, random points in box, points from history
            candidates = [best_x.copy()]
            # Add some random points
            for _ in range(5):
                candidates.append(rng.uniform(-5, 5, d))
            # Add some points from history (maybe the ones with low f)
            # Sort history by f
            sorted_indices = np.argsort(hist_f)
            for idx in sorted_indices[:5]:
                candidates.append(hist_X[idx].copy())

            best_ei = -np.inf
            best_x_ei = None

            # Simple local search (Nelder-Mead or just random perturbation around candidates)
            # Given time constraints, a simple random perturbation around candidates is safer/faster than full NM.
            # Let's try a few steps of random walk around each candidate.
            for c in candidates:
                c = np.clip(c, -5, 5)
                current_ei = ei(c)
                if current_ei > best_ei:
                    best_ei = current_ei
                    best_x_ei = c.copy()

                # Local refinement around c
                for _ in range(20):
                    step = 0.5 * rng.standard_normal(d)
                    new_c = np.clip(c + step, -5, 5)
                    new_ei = ei(new_c)
                    if new_ei > current_ei:
                        c = new_c
                        current_ei = new_ei
                        if current_ei > best_ei:
                            best_ei = current_ei
                            best_x_ei = c.copy()

            if best_x_ei is not None:
                # Evaluate the best EI point
                f_candidate = func(best_x_ei)
                # Only accept if it's actually better than the current best
                if f_candidate < best_f:
                    return best_x_ei, f_candidate
                else:
                    # Even if not better, we might want to return it to update history?
                    # The logic in __call__ handles updating history.
                    # But if it's not better, we might just return None to let ES continue.
                    # However, adding non-improving points to GP history is good for exploration.
                    # Let's return it so it gets added to history.
                    return best_x_ei, f_candidate

            return None, None

        except Exception:
            return None, None
# EVOLVE-BLOCK-END
