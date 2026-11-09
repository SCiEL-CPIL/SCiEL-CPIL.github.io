"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local polish for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_polish(self, func, x0, budget_left, deadline, rng):
        """Nelder-Mead style local refinement with budget limit."""
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        calls = 1
        
        # Build initial simplex
        simplex = [x.copy()]
        for i in range(dim):
            xi = x.copy()
            step = 0.1 if abs(x[i]) < 4.9 else 0.05
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            calls += 1
            if calls > budget_left or time.time() > deadline:
                break
        
        if len(simplex) < 2:
            return x, fx, calls
        
        # Evaluate simplex
        fvals = np.empty(len(simplex))
        for i, xi in enumerate(simplex):
            if i == 0:
                fvals[i] = fx
            else:
                fvals[i] = func(xi)
                calls += 1
                if calls > budget_left or time.time() > deadline:
                    break
        
        if calls > budget_left:
            best_idx = np.argmin(fvals[:len(simplex)])
            return simplex[best_idx], fvals[best_idx], calls
        
        # Nelder-Mead iterations
        alpha, gamma, rho = 1.0, 2.0, 0.5
        max_iter = min(50, (budget_left - calls) // (dim + 1))
        
        for _ in range(max_iter):
            if time.time() > deadline or calls >= budget_left:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if calls > budget_left or time.time() > deadline:
                if fr < fvals[-1]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                break
            
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if calls > budget_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if calls > budget_left or time.time() > deadline:
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls > budget_left or time.time() > deadline:
                            break
                    if calls > budget_left or time.time() > deadline:
                        break
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], calls

    def _cma_es_run(self, func, x_start, sigma_start, budget_left, deadline, rng):
        """Single CMA-ES run."""
        dim = self.dim
        
        # CMA-ES parameters - larger population for multimodal Rastrigin
        lam = max(12, int(4 + 3 * np.log(dim)))  # population size ~12 for dim=5
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        cc = (4 + mu_eff / (dim + 4)) / (dim + 4 + 2 * mu_eff / (dim + 4))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - cc + cs)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        x = x_start.copy()
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        sigma = sigma_start
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        gen = 0

        no_improve = 0
        threshold = 1e-6

        while calls < budget_left:
            if time.time() > deadline:
                break
            
            gen += 1
            
            # Generate population
            B = rng.standard_normal((lam, dim))
            
            # Sample from N(x, sigma^2 * C)
            try:
                L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = (C + C.T) / 2 + 1e-4 * np.eye(dim)
                L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
            
            pop = x[None, :] + sigma * (B @ L.T)
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if calls >= budget_left:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = pop[i].copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if calls >= budget_left:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update mean
            x_old = x.copy()
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_sorted[i]
            x = np.clip(x_new, -5, 5)

            # Update evolution paths
            diff = (x - x_old) / max(sigma, 1e-10)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ diff)
            ps_norm = np.linalg.norm(ps)
            hsig = (ps_norm / np.sqrt(1 - (1 - cs) ** (gen + 1))
                    / np.sqrt(dim) < 1.4 + 2 / (dim + 1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff

            # Update covariance
            delta = (1 + 2 * (hsig - 1) * c1 / cmu) if cmu > 0 else 1
            outer = np.zeros((dim, dim))
            for i in range(mu):
                outer += weights[i] * np.outer(pop_sorted[i] - x_old, pop_sorted[i] - x_old)
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / max(sigma**2, 1e-10) * outer

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C + 1e-8 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Regularize
                C = (C + C.T) / 2 + 1e-3 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / damps * (ps_norm / np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Restart if stuck
            if no_improve > lam * 3 and sigma < 0.01:
                break
            if sigma < 1e-4:
                break

        return best_x, best_fx, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_fx = np.inf
        total_calls = 0

        # Strategy: Try multiple starting points with different sigmas
        # Rastrigin is multimodal, so we need good exploration
        
        # Phase 1: Broad exploration with several CMA-ES runs from different starts
        start_points = [
            np.zeros(dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([3.0, -3.0, 2.0, -2.0, 3.0]),
            np.array([-3.0, 3.0, -2.0, 2.0, -3.0]),
        ]
        
        # Allocate budget: ~60% for CMA-ES exploration, ~25% for more CMA-ES, ~15% for polish
        n_starts = min(len(start_points), 3)
        cma_budget_each = max(50, int(budget * 0.25))
        
        for s in range(n_starts):
            if total_calls >= budget or time.time() > deadline:
                break
            
            x_start = start_points[s].copy()
            sigma_start = 2.0 if s == 0 else 1.5
            
            remaining = budget - total_calls
            if remaining < cma_budget_each:
                cma_budget_each = remaining
            
            if cma_budget_each < 20:
                break
            
            x_best, fx_best, calls_made = self._cma_es_run(
                func, x_start, sigma_start, cma_budget_each, deadline, rng
            )
            
            total_calls += calls_made
            if fx_best < best_fx:
                best_fx = fx_best
                best_x = x_best.copy()
        
        # Phase 2: Local polish on the best point found
        if total_calls < budget and time.time() < deadline:
            polish_budget = min(80, budget - total_calls)
            if polish_budget >= 10:
                x_polished, fx_polished, calls_polish = self._local_polish(
                    func, best_x, polish_budget, deadline, rng
                )
                total_calls += calls_polish
                if fx_polished < best_fx:
                    best_fx = fx_polished
                    best_x = x_polished.copy()
        
        # Phase 3: If budget remains, do another CMA-ES from the best point with small sigma
        if total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining >= 30:
                x_refined, fx_refined, calls_refined = self._cma_es_run(
                    func, best_x, 0.3, remaining, deadline, rng
                )
                total_calls += calls_refined
                if fx_refined < best_fx:
                    best_fx = fx_refined
                    best_x = x_refined.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
