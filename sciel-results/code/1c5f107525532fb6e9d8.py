"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (mu+lambda)-ES with multi-restart Nelder-Mead for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, start_x, budget_remaining, max_time, rng, dim):
        """Nelder-Mead with tight convergence. Returns (best_x, best_f, calls_made)."""
        calls_made = 0
        simplex = [start_x.copy()]
        base_sigma = 0.3
        for i in range(dim):
            pt = start_x.copy()
            pt[i] += base_sigma
            pt = np.clip(pt, -5, 5)
            simplex.append(pt)

        simplex_f = []
        for pt in simplex:
            if calls_made >= budget_remaining or time.time() > max_time:
                break
            f = func(pt)
            calls_made += 1
            simplex_f.append(f)

        if len(simplex_f) < dim + 1:
            return start_x, min(simplex_f) if simplex_f else float('inf'), calls_made

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        shrink = 0.5

        best_f = min(simplex_f)
        best_x = simplex[np.argmin(simplex_f)].copy()

        max_iter = min(budget_remaining - calls_made, 150)
        for _ in range(max_iter):
            if calls_made >= budget_remaining or time.time() > max_time:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if simplex_f[-1] - simplex_f[0] < 1e-12:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            worst_pt = simplex[-1]
            worst_f = simplex_f[-1]

            x_ref = centroid + alpha * (centroid - worst_pt)
            x_ref = np.clip(x_ref, -5, 5)
            if calls_made >= budget_remaining or time.time() > max_time:
                break
            f_ref = func(x_ref)
            calls_made += 1

            if f_ref < simplex_f[0]:
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                if calls_made >= budget_remaining or time.time() > max_time:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                    break
                f_exp = func(x_exp)
                calls_made += 1
                if f_exp < f_ref:
                    x_new, f_new = x_exp, f_exp
                else:
                    x_new, f_new = x_ref, f_ref
                simplex[-1] = x_new
                simplex_f[-1] = f_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            elif f_ref < simplex_f[-2]:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
            else:
                x_con = centroid + rho * (worst_pt - centroid)
                x_con = np.clip(x_con, -5, 5)
                if calls_made >= budget_remaining or time.time() > max_time:
                    break
                f_con = func(x_con)
                calls_made += 1
                if f_con < worst_f:
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made >= budget_remaining or time.time() > max_time:
                            break
                        f = func(simplex[i])
                        calls_made += 1
                        simplex_f[i] = f
                        if f < best_f:
                            best_f = f
                            best_x = simplex[i].copy()

        return best_x, best_f, calls_made

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = start_time + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Phase 1: (mu+lambda)-ES for exploration
        mu = 4
        lam = 12
        sigma = 1.5
        sigma_min = 1e-6
        sigma_max = 3.5

        # Initial population: stratified + center
        population = []
        # Center point
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls += 1
        population.append((fx0, x0.copy()))
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()

        # Stratified initial points
        for i in range(1, lam):
            if calls >= budget or time.time() > max_time:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            calls += 1
            population.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        population.sort(key=lambda t: t[0])

        # ES main loop with CMA-ES-like sigma adaptation
        es_phase_budget = int(budget * 0.65)
        success_history = []
        stagnation_limit = 35
        iters_no_improve = 0

        while calls < es_phase_budget and time.time() < max_time:
            parents = [x for _, x in population[:mu]]
            mean_x = np.mean(parents, axis=0)

            offspring = []
            for j in range(lam):
                if calls >= es_phase_budget or time.time() > max_time:
                    break
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                candidate = mean_x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                offspring.append((fc, candidate.copy()))
                if fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()

            if not offspring:
                break

            combined = population + offspring
            combined.sort(key=lambda t: t[0])
            population = combined[:lam]

            if population[0][0] < best_f - 1e-12:
                iters_no_improve = 0
            else:
                iters_no_improve += 1

            # 1/5 success rule for sigma adaptation
            parent_best = parents[0][0] if parents else best_f
            successes = sum(1 for f, _ in offspring if f < parent_best)
            success_rate = successes / len(offspring) if offspring else 0.0

            if success_rate > 0.35:
                sigma = min(sigma * 1.15, sigma_max)
            elif success_rate < 0.15:
                sigma = max(sigma * 0.85, sigma_min)
            else:
                sigma = min(sigma * 1.02, sigma_max)

            # Stagnation: restart around best with wider spread
            if iters_no_improve >= stagnation_limit:
                population = []
                for i in range(lam):
                    if calls >= es_phase_budget or time.time() > max_time:
                        break
                    spread = 1.0 + 0.5 * (i / lam)
                    x = best_x + rng.normal(0, spread, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    population.append((fx, x.copy()))
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                population.sort(key=lambda t: t[0])
                sigma = 0.8
                iters_no_improve = 0

        # Phase 2: Multi-restart Nelder-Mead from diverse archive points
        nm_budget = budget - calls
        nm_max_time = max_time

        # Select diverse starting points from ES population
        archive = [best_x.copy()]
        # Add some diverse points from the ES population
        if len(population) > mu:
            for idx in [mu, mu + 2, mu + 4]:
                if idx < len(population):
                    archive.append(population[idx][1].copy())

        # Deduplicate and limit to 3-4 restarts
        unique_starts = []
        for pt in archive:
            if all(np.linalg.norm(pt - u) > 0.5 for u in unique_starts):
                unique_starts.append(pt)
            if len(unique_starts) >= 4:
                break

        nm_per_restart = max(10, nm_budget // len(unique_starts))

        for start_pt in unique_starts:
            if calls >= budget or time.time() > nm_max_time:
                break
            remaining = budget - calls
            nm_budget_this = min(nm_per_restart, remaining)
            if nm_budget_this < 10:
                break

            nx, nf, nc = self._nelder_mead(func, start_pt, nm_budget_this, nm_max_time, rng, dim)
            calls += nc
            if nf < best_f:
                best_f = nf
                best_x = nx

        return best_x, best_f
# EVOLVE-BLOCK-END
