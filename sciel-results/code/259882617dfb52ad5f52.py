import numpy as np
import time


class Optimizer:
    """CMA-ES with PSO-inspired global exploration for BBOB-style f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES with the given initial state and budget."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, (mu_eff - 2) / (dim + 2) - 1) + cs

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        B = eigvecs
        D = np.sqrt(eigvals)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        no_improve = 0

        for gen in range(budget // lam):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            pop = []
            fvals = []
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)
                calls += 1

            if len(pop) == 0:
                break

            idx = np.argsort(fvals)
            pop = [pop[i] for i in idx]
            fvals = np.array(fvals)[idx]

            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 20 and sigma < 1e-5:
                break

            mu_use = min(mu, len(pop))
            m = np.sum(weights[:mu_use, None] * np.array(pop[:mu_use]), axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B.T @ (m - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (B.T @ (m - x))

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * outer
            for i in range(mu_use):
                zi = (np.array(pop[i]) - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = m

        return best_x, best_fx, calls

    def _pso_global(self, func, budget, rng, deadline, n_particles=20):
        """PSO for global exploration."""
        dim = self.dim
        calls = 0
        deadline = min(deadline, time.time() + 280.0)

        # Initialize swarm
        pos = rng.uniform(-5, 5, (n_particles, dim))
        vel = rng.uniform(-2, 2, (n_particles, dim))

        pbest_pos = pos.copy()
        pbest_val = np.full(n_particles, np.inf)
        gbest_pos = pos[0].copy()
        gbest_val = np.inf

        for i in range(n_particles):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            val = func(pos[i])
            calls += 1
            pbest_val[i] = val
            if val < gbest_val:
                gbest_val = val
                gbest_pos = pos[i].copy()

        w = 0.8
        c1 = 1.5
        c2 = 1.5

        while calls < budget:
            if time.time() > deadline:
                break

            for i in range(n_particles):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
                vel[i] = np.clip(vel[i], -2.0, 2.0)
                pos[i] = pos[i] + vel[i]
                pos[i] = np.clip(pos[i], -5, 5)

                val = func(pos[i])
                calls += 1

                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[i].copy()

            w = max(0.5, w * 0.97)

        return gbest_pos, gbest_val, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Phase 1: PSO global exploration to find a good basin
        pso_budget = min(100, budget // 5)
        pso_x, pso_fx, pso_calls = self._pso_global(func, pso_budget, rng, deadline, n_particles=20)

        # Also evaluate a few diverse points
        candidates = [
            np.zeros(dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        ]

        best_x = pso_x.copy()
        best_fx = pso_fx
        calls = pso_calls

        for c in candidates:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            fc = func(c)
            calls += 1
            if fc < best_fx:
                best_fx = fc
                best_x = c.copy()

        # Phase 2: CMA-ES local refinement
        if calls < budget - 10:
            sigma0 = 1.0
            cma_budget = budget - calls
            cma_x, cma_fx, cma_calls = self._cmaes(func, best_x, sigma0, cma_budget, rng, deadline)
            calls += cma_calls
            if cma_fx < best_fx:
                best_fx = cma_fx
                best_x = cma_x.copy()

        # Phase 3: If budget remains, try a second CMA-ES from a slightly perturbed best
        remaining = budget - calls
        if remaining > 20 and time.time() < deadline - 10:
            # Perturb best and run short CMA-ES
            perturbed = best_x + 0.5 * rng.standard_normal(dim)
            perturbed = np.clip(perturbed, -5, 5)
            fp = func(perturbed)
            calls += 1
            if fp < best_fx:
                best_fx = fp
                best_x = perturbed.copy()

            if remaining > 15 and time.time() < deadline - 5:
                cma_x2, cma_fx2, cma_calls2 = self._cmaes(func, best_x, 0.5, remaining, rng, deadline)
                calls += cma_calls2
                if cma_fx2 < best_fx:
                    best_fx = cma_fx2
                    best_x = cma_x2.copy()

        return best_x, best_fx
