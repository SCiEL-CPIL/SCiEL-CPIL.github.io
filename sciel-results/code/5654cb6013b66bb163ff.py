"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead polish for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s kill

        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize center in [-4, 4]^5 (minimizer lies here)
        x = rng.uniform(-4.0, 4.0, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # CMA-ES parameters
        sigma = 2.0
        lambda_ = int(4 + 3 * d)  # population size
        weights = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights /= weights.sum()
        mu = int(np.floor(lambda_ / 2))
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)

        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate population
            ys = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                y = x + sigma * (C @ rng.standard_normal(d))
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                ys.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(ys) == 0:
                break

            # Sort by fitness
            ys.sort(key=lambda t: t[0])
            selected = ys[:mu]
            mean_new = np.zeros(d)
            for w_i, (f_i, y_i) in zip(weights[:mu], selected):
                mean_new += w_i * y_i

            # Update path
            diff = (mean_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            h_sigma = np.dot(p_sigma, p_sigma) / (1 - (1 - c_sigma) ** (2 * (gen + 1))) \
                      < (1 + 2 / (d + 1)) * chi_n ** 2

            if h_sigma:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * (mu_eff - 1) + c_mu - c_1 + delta_h) * C \
                + c_1 * (mu_eff - 1 + 1 / mu_eff) * np.outer(diff, diff) \
                + c_mu * np.sum([w_i * np.outer(y_i - x, y_i - x) / sigma ** 2
                                  for w_i, (f_i, y_i) in zip(weights[:mu], selected)], axis=0) \
                - delta_h * outer

            # Eigenvalue regularization for positive-definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(d)

            x = mean_new
            fx = best_f  # use best found as reference

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Periodic Nelder-Mead polish if we have budget remaining
            if gen % 10 == 9 and calls < budget - 20 and time.time() < deadline - 5:
                nm_x, nm_f = self._nelder_mead(func, best_x, budget - calls, d)
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x
                calls = budget  # NM consumed remaining budget, exit CMA-ES
                break

        # Final Nelder-Mead polish if budget remains
        if calls < budget and time.time() < deadline - 2:
            nm_x, nm_f = self._nelder_mead(func, best_x, budget - calls, d)
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x

        return best_x, best_f

    def _nelder_mead(self, func, x0, remaining_calls, d):
        """Nelder-Mead simplex method with call budget."""
        if remaining_calls < 2 * d + 1:
            return x0, func(x0)

        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        calls = 1

        for i in range(d):
            xi = x0.copy()
            xi[i] += 0.05 if x0[i] < 4.95 else -0.05
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            fvals.append(func(xi))
            calls += 1
            if calls >= remaining_calls:
                break

        if calls >= remaining_calls:
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Nelder-Mead iterations
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        best_x = x0.copy()
        best_f = fvals[0]

        while calls < remaining_calls:
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                calls += 1
                if fc < fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= remaining_calls:
                            break
                    if calls >= remaining_calls:
                        break

            if calls >= remaining_calls:
                break

        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]]
# EVOLVE-BLOCK-END
