"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES followed by Nelder-Mead polish for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, calls_remaining, deadline):
        """Nelder-Mead simplex method with adaptive step size."""
        dim = self.dim
        if dim <= 0 or calls_remaining <= 0:
            return x0.copy(), func(x0), 0

        # Initialize simplex
        simplex = [x0.copy()]
        # Initial step: small relative to domain
        initial_step = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += initial_step
            simplex.append(np.clip(xi, -5, 5))

        fvals = np.array([func(x) for x in simplex])
        calls_used = dim + 1

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 500
        best_x = x0.copy()
        best_f = min(fvals)
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        for _ in range(max_iter):
            if calls_used >= calls_remaining or time.time() >= deadline:
                break

            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = fvals[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_used += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_used += 1
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= calls_remaining or time.time() >= deadline:
                            break

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        deadline = time.time() + 310.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: CMA-ES for global search
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x.copy()
        best_f = func(x)
        calls_used = 1
        gen = 0
        no_improve = 0
        stagnation_threshold = 20

        # Reserve budget for Nelder-Mead polish
        nm_budget = min(50, budget // 10)
        cma_budget = budget - nm_budget

        while calls_used < cma_budget and time.time() < deadline:
            if calls_used + lam > cma_budget:
                break

            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            pop = np.array([x + sigma * (z @ L) for z in rng.standard_normal((lam, dim))])
            pop = np.clip(pop, -5, 5)
            fits = np.array([func(p) for p in pop])
            calls_used += lam
            gen += 1

            # Update best
            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f:
                best_f = fits[idx_min]
                best_x = pop[idx_min].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Sort by fitness
            sorted_idx = np.argsort(fits)
            y_sorted = pop[sorted_idx]

            # Weighted mean
            m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update path states
            yw = (m_new - x) / sigma
            pc = (1 - cs) * pc + cs * yw
            ps = (1 - cs) * ps + cs * np.sign(yw)

            # Sigma update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen))) < (1.4 + 2 / (dim + 1)) * chi
            sigma *= np.exp(cs / chi * (np.linalg.norm(pc) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Covariance update
            delta_hsig = (1.0 if hsig else 0.0) * (1 - cs) * cs
            C = (1 + c1 + cmu - c1 - cmu) * C
            C += c1 * (np.outer(yw, yw) + (1.0 if hsig else 0.0) * cs * (1 - cs) - C)
            C += cmu * np.sum(weights[:, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x) / (sigma ** 2), axis=0)
            C = (C + C.T) / 2

            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals) + 1e-12) * np.eye(dim)

            x = m_new

            # Early stop if stagnated
            if no_improve >= stagnation_threshold:
                break

        # Phase 2: Nelder-Mead polish
        remaining = budget - calls_used
        if remaining > 0 and time.time() < deadline:
            nm_x, nm_f, nm_calls = self._nelder_mead(func, best_x, remaining, deadline)
            calls_used += nm_calls
            if nm_f < best_f:
                best_x = nm_x
                best_f = nm_f

        return best_x, best_f
# EVOLVE-BLOCK-END
