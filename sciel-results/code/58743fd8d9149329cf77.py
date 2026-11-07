"""Combined PSO-CMA-ES hybrid optimizer with adaptive strategy switching and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid PSO-CMA-ES optimizer with adaptive strategy switching and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        self._call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            val = func(np.clip(x, -5, 5))
            if val < best_f:
                best_f = val
                best_x = np.clip(x, -5, 5).copy()
            return val

        def nelder_mead_local(start_x, max_evals):
            """Quick local NM polish around start_x."""
            nonlocal best_x, best_f, deadline
            if self._call_count >= self.budget or time.time() > deadline:
                return
            x0 = np.clip(start_x, -5, 5)
            f0 = counted_func(x0)

            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                # Use a smaller initial step for better precision
                step = 0.1
                xi[i] = x0[i] + step if x0[i] < 4.9 else x0[i] - step
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, self.dim + 1):
                simplex_f.append(counted_func(simplex[i]))

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            iters = 0
            while iters < max_evals and self._call_count < self.budget and time.time() < deadline:
                iters += 1
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = counted_func(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = counted_func(xe)
                    simplex[-1] = xe if fe < fr else xr
                    simplex_f[-1] = fe if fe < fr else fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = counted_func(xc)
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Strategy 1: PSO for global exploration
        n_particles = 25
        w_max = 0.9
        w_min = 0.3
        c1 = 1.49445
        c2 = 1.49445

        positions = rng.uniform(-5, 5, (n_particles, dim))
        velocities = rng.uniform(-1, 1, (n_particles, dim))
        personal_bests = np.copy(positions)
        personal_best_f = np.full(n_particles, np.inf)

        for i in range(n_particles):
            if self._call_count >= budget or time.time() > deadline:
                break
            f = counted_func(positions[i])
            personal_best_f[i] = f
            personal_bests[i] = positions[i].copy()

        global_best = personal_bests[np.argmin(personal_best_f)].copy()
        global_best_f = np.min(personal_best_f)

        # PSO Phase: 60% of budget
        pso_budget = int(budget * 0.6)
        pso_calls_used = 0
        max_pso_iters = pso_budget // n_particles

        stagnation_counter = 0
        last_best_f = global_best_f

        for it in range(max_pso_iters):
            if self._call_count >= budget or time.time() > deadline:
                break
            pso_calls_used += 1

            progress = min(1.0, (it + 1) / max_pso_iters)
            w = w_max - (w_max - w_min) * progress

            for i in range(n_particles):
                if self._call_count >= budget or time.time() > deadline:
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (personal_bests[i] - positions[i])
                                 + c2 * r2 * (global_best - positions[i]))

                v_max = 2.0
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                f = counted_func(positions[i])
                if f < personal_best_f[i]:
                    personal_best_f[i] = f
                    personal_bests[i] = positions[i].copy()
                    if f < global_best_f:
                        global_best_f = f
                        global_best = positions[i].copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # Check for stagnation
            if global_best_f < last_best_f:
                last_best_f = global_best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # If stagnation is high, break early to save budget for CMA-ES
            if stagnation_counter > 3 * n_particles:
                break

        # Strategy 2: CMA-ES for refined exploitation
        # Initialize CMA-ES around the current global best
        if self._call_count < budget and time.time() < deadline:
            m = global_best.copy()
            sigma = 0.5  # Start with a moderate step size
            C = np.eye(dim)
            B = np.eye(dim)
            D = 1.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            # CMA-ES parameters
            lam = 10
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            cma_budget = int((budget - self._call_count) * 0.7)
            cma_calls_used = 0
            max_cma_iters = cma_budget // lam

            for it in range(max_cma_iters):
                if self._call_count >= budget or time.time() > deadline:
                    break
                cma_calls_used += 1

                # Sample population
                pop = np.zeros((lam, dim))
                pop_f = np.zeros(lam)

                for i in range(lam):
                    if self._call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    # Transform to normal with covariance C
                    # Use Cholesky or eigendecomposition
                    x = m + sigma * (B @ (D * (B.T @ z)))
                    x = np.clip(x, -5, 5)
                    pop[i] = x
                    pop_f[i] = counted_func(x)

                if self._call_count >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update mean
                m_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                m_new = np.clip(m_new, -5, 5)

                # Update covariance
                diff = (pop[:mu] - m) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - (1 - c_c) * C) + \
                    c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * diff[:, np.newaxis, :] * diff[:, :, np.newaxis], axis=0)

                # Symmetrize C
                C = (C + C.T) / 2

                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (B.T @ (m_new - m))))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (B @ (D * (B.T @ (m_new - m))))

                # Update B and D via eigendecomposition
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-10, None)
                B = evecs
                D = 1.0 / np.sqrt(evals)

                m = m_new

                # Track best
                if pop_f[0] < global_best_f:
                    global_best_f = pop_f[0]
                    global_best = pop[0].copy()

        # Strategy 3: Final Local Polish
        polish_budget = budget - self._call_count
        if polish_budget > 10 and best_x is not None:
            # Run NM from the global best
            nelder_mead_local(global_best, polish_budget // 2)

            # Also try from top PSO particles if budget remains
            if self._call_count < budget:
                order = np.argsort(personal_best_f)
                top_k = min(2, n_particles)
                for k in range(top_k):
                    if self._call_count >= budget or time.time() > deadline:
                        break
                    alloc = max(5, (budget - self._call_count) // (top_k - k))
                    nelder_mead_local(personal_bests[order[k]], alloc)

        return best_x, best_f
# EVOLVE-BLOCK-END
