"""Rewritten optimizer: CMA-ES + quadratic surrogate refinement."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with quadratic surrogate refinement for bent-cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        # Initial population: center + random points
        rng = np.random.default_rng(42)
        center = np.zeros(dim)
        fc = func(center)
        best_x = center.copy()
        best_f = fc

        # Try a few random starting points to find a better initial center
        for _ in range(10):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        # CMA-ES parameters
        lam = 2 * dim  # population size ~10 for dim=5
        sigma = 1.0
        mean = best_x.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_lam = np.sqrt(lam) * (1 + 1 / (4 * dim) - 1 / (2 * lam))

        # Adaptation rates
        c_sigma = (lam + 2) / (dim + lam + 5)
        mu_sigma = 0.5 * c_sigma * (2 - c_sigma)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) ** 2 - 1))
        mu_eff = lam / 2  # effective selection mass

        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam / 2 + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights[:lam // 2] ** 2)

        # Track evaluations
        n_evals = 11  # 1 center + 10 random
        # Store recent evaluations for surrogate
        recent_x = []
        recent_f = []
        recent_x.append(center.copy())
        recent_f.append(fc)
        for _ in range(10):
            pass  # we didn't store them, let's redo

        # Actually, let's be more careful. Let's restart with proper tracking.
        # Re-initialize with proper bookkeeping
        n_evals = 0
        recent_x = []
        recent_f = []

        def eval_and_track(x):
            nonlocal n_evals, best_x, best_f
            fx = func(x)
            n_evals += 1
            recent_x.append(x.copy())
            recent_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            # Keep only last 50 for surrogate
            if len(recent_x) > 50:
                recent_x.pop(0)
                recent_f.pop(0)
            return fx

        # Initial evaluations
        fc = eval_and_track(center)
        for _ in range(10):
            x = rng.uniform(-5, 5, dim)
            eval_and_track(x)

        # Re-initialize CMA state from best point
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES loop
        while n_evals < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = 0.5 * (C + C.T)
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

            # Sample
            ys = []
            xs = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = sigma * (L @ z)
                x = np.clip(mean + y, -5, 5)
                xs.append(x)
                ys.append(y)

            # Evaluate
            fs = []
            for x in xs:
                fx = eval_and_track(x)
                fs.append(fx)
                if n_evals >= budget or time.time() > deadline:
                    break

            if n_evals >= budget or time.time() > deadline:
                break

            fs = np.array(fs)
            xs = np.array(xs)

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            ys_sorted = [ys[i] for i in order]

            # Update mean
            for i in range(lam // 2):
                mean += weights[i] * xs_sorted[i]

            # Update sigma (CSA)
            h_sigma = (np.sqrt((mu_eff + 2) / (dim + 2)) - 1) / (1 + c_sigma)
            # Approximate delta
            C_sqrt = np.sqrt(np.diag(C))
            y_mean = np.mean(ys_sorted[:lam // 2], axis=0)
            delta = np.linalg.norm(y_mean) / (sigma * chi_lam) if sigma > 0 else 0
            h_sigma = 1.0 if delta < (1 + 2 / (dim + 1)) * h_sigma else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_mean / chi_lam)
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_lam - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update C
            y_c = np.mean(ys_sorted[:lam // 2], axis=0)
            h_c = (np.sqrt((mu_eff + 2) / (dim + 2)) - 1 + 2 * (1 - c_c)) / (1 + c_c)
            delta_c = np.linalg.norm(y_c) / (sigma * chi_lam) if sigma > 0 else 0
            h_c = 1.0 if delta_c < (1 + 2 / (dim + 1)) * h_c else 0.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_c / chi_lam)

            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(lam // 2):
                rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])

            C = (1 + c_1 + c_mu - c_1 - mu_eff * c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            C = 0.5 * (C + C.T)

            # Update C_inv
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)

            # Surrogate refinement every 20 evaluations
            if n_evals % 20 == 0 and len(recent_x) >= dim + 1:
                surrogate_x, surrogate_f = self._try_surrogate_refinement(
                    func, recent_x[-(dim + 1):], recent_f[-(dim + 1):],
                    best_x, best_f, deadline
                )
                if surrogate_f is not None and surrogate_f < best_f:
                    best_x = surrogate_x
                    best_f = surrogate_f
                    mean = surrogate_x.copy()
                    sigma *= 0.5

        return best_x, best_f

    def _try_surrogate_refinement(self, func, xs, fs, best_x, best_f, deadline):
        """Fit a quadratic surrogate and find its minimum."""
        dim = len(best_x)
        xs = np.array(xs)
        fs = np.array(fs)

        # Build design matrix for quadratic: [1, x, x_i*x_j for i<=j]
        n = len(xs)
        n_features = 1 + dim + dim * (dim + 1) // 2

        if n < n_features:
            return best_x, None

        A = np.ones((n, n_features))
        A[:, 1:1 + dim] = xs
        idx = 1 + dim
        for i in range(dim):
            for j in range(i, dim):
                A[:, idx] = xs[:, i] * xs[:, j]
                idx += 1

        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, fs, rcond=None)
        except np.linalg.LinAlgError:
            return best_x, None

        # Find minimum of the quadratic: solve grad = 0
        # f(x) = c0 + c1'x + x'Qx where Q is symmetric
        # grad = c1 + 2Qx = 0 => x = -0.5 Q^{-1} c1

        c0 = coeffs[0]
        c1 = coeffs[1:1 + dim]

        # Build Q from the quadratic coefficients
        Q = np.zeros((dim, dim))
        idx = 1 + dim
        for i in range(dim):
            for j in range(i, dim):
                val = coeffs[idx]
                Q[i, j] = val
                Q[j, i] = val
                idx += 1

        # Check if Q is positive definite enough
        try:
            eigvals = np.linalg.eigvalsh(Q)
            if np.min(eigvals) < -1e-6:
                # Not convex, skip
                return best_x, None
            Q_reg = Q + 1e-8 * np.eye(dim)
            x_min = -0.5 * np.linalg.solve(Q_reg, c1)
        except np.linalg.LinAlgError:
            return best_x, None

        # Clip to box
        x_min = np.clip(x_min, -5, 5)

        # Evaluate
        if time.time() > deadline:
            return best_x, None

        f_min = func(x_min)

        # Only accept if better than current best
        if f_min < best_f:
            return x_min, f_min
        else:
            return best_x, None
