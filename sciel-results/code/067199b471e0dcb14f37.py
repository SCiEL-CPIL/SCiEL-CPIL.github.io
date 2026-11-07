"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with derivative-free Broyden refinement for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_cmaes(self, func, x0, sigma0, budget, dim, rng, deadline):
        """Run CMA-ES from a given starting point."""
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        calls = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls += 1

        if calls >= budget or time.time() > deadline:
            return best_x, best_f, calls

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        L = np.eye(dim)
        stagnation_counter = 0
        best_f_prev = best_f
        stagnation_limit = 25

        while calls < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)

            # Evaluate population
            fvals = []
            for x in pop:
                fx = _eval(x)
                if fx is None:
                    fvals.append(np.inf)
                else:
                    fvals.append(fx)

            if calls >= budget or time.time() > deadline:
                break

            fvals = np.array(fvals)
            order = np.argsort(fvals)
            sorted_pop = np.array([pop[i] for i in order])
            sorted_fvals = fvals[order]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

            # Compute C^{-1/2} via eigendecomposition
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-12, None)
            inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

            diff = (new_mean - mean) / sigma

            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            delta_sigma = (norm_ps / chi_n - 1) * d_sigma
            sigma *= np.exp(c_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update C
            y = (sorted_pop[:mu] - mean) / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            decay = 1 - c_1 - c_mu
            rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
            C = decay * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            mean = new_mean

            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            # Convergence check
            if sigma < 1e-12:
                break

            # Stagnation detection
            if best_f < best_f_prev - 1e-10:
                stagnation_counter = 0
                best_f_prev = best_f
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit:
                break

        return best_x, best_f, calls

    def _dfb_refine(self, func, x0, budget, dim, deadline):
        """Derivative-free Broyden refinement around x0."""
        if budget < 10 or time.time() > deadline:
            return x0, func(x0), 1

        x = x0.copy()
        fx = func(x)
        calls = 1
        if calls >= budget or time.time() > deadline:
            return x, fx, calls

        # Initialize Broyden matrix
        B = np.eye(dim)
        H = np.eye(dim)  # Hessian approximation (inverse of B)

        # Finite difference step
        delta = 1e-4

        def _eval(x):
            nonlocal calls, fx
            if calls >= budget or time.time() > deadline:
                return None
            f = func(x)
            calls += 1
            return f

        for iteration in range(budget - 1):
            if time.time() > deadline or calls >= budget:
                break

            # Compute numerical gradient
            grad = np.zeros(dim)
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                x_plus = x.copy()
                x_plus[i] += delta
                x_minus = x.copy()
                x_minus[i] -= delta
                f_plus = _eval(x_plus)
                if f_plus is None:
                    break
                f_minus = _eval(x_minus)
                if f_minus is None:
                    break
                grad[i] = (f_plus - f_minus) / (2 * delta)

            if calls >= budget or time.time() > deadline:
                break

            # Check convergence
            if np.linalg.norm(grad) < 1e-8:
                break

            # Broyden update: solve B @ s = -grad
            try:
                s = np.linalg.solve(B, -grad)
            except np.linalg.LinAlgError:
                s = -grad

            # Line search with backtracking
            step_size = 1.0
            for _ in range(20):
                x_new = x + step_size * s
                x_new = np.clip(x_new, -5, 5)
                if calls >= budget or time.time() > deadline:
                    break
                f_new = _eval(x_new)
                if f_new is None:
                    break
                if f_new < fx - 1e-10 * step_size * np.linalg.norm(grad) ** 2:
                    # Broyden rank-1 update
                    y_vec = (f_new - fx) / step_size  # approximate gradient change
                    # Proper Broyden update
                    s_vec = x_new - x
                    if np.linalg.norm(s_vec) > 1e-12:
                        grad_new = grad  # approximate; we don't have true grad_new cheaply
                        # Use simplified update
                        B = B + np.outer(s_vec, s_vec) / (np.dot(s_vec, s_vec) + 1e-12)
                        # Regularize
                        B = (B + B.T) / 2
                        B += 1e-8 * np.eye(dim)

                    x = x_new
                    fx = f_new
                    break
                else:
                    step_size *= 0.5

            if time.time() > deadline or calls >= budget:
                break

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Phase 1: CMA-ES with multiple restarts
        n_restarts = 4
        budget_per_restart = max(50, budget // n_restarts)

        # Initial points
        x0_list = [np.zeros(dim)]
        for i in range(n_restarts - 1):
            x0 = rng.uniform(-4, 4, size=dim)
            x0_list.append(x0)

        global_best_x = None
        global_best_f = np.inf
        total_calls = 0

        for i, x0 in enumerate(x0_list):
            if total_calls >= budget or time.time() > deadline:
                break

            remaining = budget - total_calls
            alloc = min(budget_per_restart, remaining)
            if alloc < 20:
                break

            sigma0 = 2.0 if i == 0 else 1.0

            best_x, best_f, calls = self._run_cmaes(
                func, x0, sigma0, alloc, dim, rng, deadline
            )

            total_calls += calls

            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        # Phase 2: DFB refinement around the best point
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 5 and global_best_x is not None:
            best_x, best_f, calls = self._dfb_refine(
                func, global_best_x, min(remaining, 80), dim, deadline
            )
            total_calls += calls
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        # Phase 3: If we still have budget, do another round of CMA-ES from best point
        remaining = budget - total_calls
        if remaining > 30 and time.time() < deadline - 10 and global_best_x is not None:
            best_x, best_f, calls = self._run_cmaes(
                func, global_best_x, 0.5, min(remaining, 60), dim, rng, deadline
            )
            total_calls += calls
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

            # Another DFB pass
            remaining = budget - total_calls
            if remaining > 10 and time.time() < deadline - 5:
                best_x, best_f, calls = self._dfb_refine(
                    func, global_best_x, min(remaining, 40), dim, deadline
                )
                total_calls += calls
                if best_f < global_best_f:
                    global_best_f = best_f
                    global_best_x = best_x.copy()

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
