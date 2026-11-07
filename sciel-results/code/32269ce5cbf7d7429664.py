import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Population-based differential evolution (DE/rand/1) with adaptive mutation,
    combined with targeted local exploitation via L-BFGS-B and Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        # GP data storage
        X_gp = []
        y_gp = []

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("time exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def gp_predict(X_test):
            """Gaussian Process prediction with RBF kernel and adaptive length-scale."""
            n = len(X_gp)
            if n == 0:
                return np.zeros(X_test.shape[0]), np.zeros(X_test.shape[0])
            if n == 1:
                return np.full(X_test.shape[0], y_gp[0]), np.full(X_test.shape[0], 1.0)

            y = np.array(y_gp)
            y_mean = y.mean()
            y_centered = y - y_mean

            # Adaptive length-scale from data spread
            X_arr = np.array(X_gp)
            if n > 1:
                diffs = np.linalg.norm(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :], axis=-1)
                upper = np.triu(diffs, k=1)
                if upper.size > 0:
                    ls = np.median(upper)
                    if ls < 1e-6:
                        ls = 1.0
                else:
                    ls = 1.0
            else:
                ls = 1.0

            inv_2ls2 = 0.5 / (ls ** 2)

            # RBF kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X_gp[i] - X_gp[j]
                    K[i, j] = np.exp(-inv_2ls2 * np.sum(diff ** 2))
            K += 1e-8 * np.eye(n)

            # Kernel between test and training
            K_s = np.zeros((X_test.shape[0], n))
            for i in range(X_test.shape[0]):
                for j in range(n):
                    diff = X_test[i] - X_gp[j]
                    K_s[i, j] = np.exp(-inv_2ls2 * np.sum(diff ** 2))

            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            mu = y_mean + K_s @ K_inv @ y_centered
            var = 1.0 - K_s @ K_inv @ K_s.T
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)

            return mu, sigma

        def expected_improvement(mu, sigma, f_best):
            """EI acquisition function for minimization."""
            improvement = f_best - mu
            z = improvement / (sigma + 1e-10)
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial diverse sampling ---
        n_initial = min(15, budget // 4)
        starts = [np.zeros(dim)]
        for _ in range(n_initial - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                update_best(s, f)
                X_gp.append(s.copy())
                y_gp.append(f)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)
            X_gp.append(best_x.copy())
            y_gp.append(best_f)

        # --- Phase 2: Differential Evolution with adaptive mutation ---
        pop_size = min(15, max(10, budget // 20))
        # Initialize population from GP data if enough, else random
        if len(X_gp) >= pop_size:
            # Use diverse points from GP data
            X_arr = np.array(X_gp)
            # Sort by fitness, take spread
            sorted_idx = np.argsort(y_gp)
            # Take a mix: top few + random
            top_k = min(5, pop_size)
            top_indices = sorted_idx[:top_k]
            remaining = np.setdiff1d(np.arange(len(X_gp)), top_indices)
            if len(remaining) > 0:
                rand_indices = rng.choice(remaining, size=pop_size - top_k, replace=False)
                pop_indices = np.concatenate([top_indices, rand_indices])
            else:
                pop_indices = np.random.choice(len(X_gp), size=pop_size, replace=False)
            population = [X_gp[i].copy() for i in pop_indices]
            pop_fitness = [y_gp[i] for i in pop_indices]
        else:
            population = []
            pop_fitness = []
            for x in X_gp:
                population.append(x.copy())
                pop_fitness.append(float(np.inf) if np.isnan(x).any() else 0.0)
            # Fill remaining with random
            while len(population) < pop_size:
                x = rng.uniform(-5, 5, dim)
                population.append(x)
                pop_fitness.append(float('inf'))
            # Evaluate un-evaluated population members
            for i in range(len(population)):
                if pop_fitness[i] == float('inf'):
                    try:
                        f = wrapped(population[i])
                        pop_fitness[i] = f
                        X_gp.append(population[i].copy())
                        y_gp.append(f)
                        update_best(population[i], f)
                    except Exception:
                        pass

        # Adaptive mutation factor
        F = 0.8  # Start with standard DE/rand/1 mutation factor

        de_iterations = 0
        max_de_iterations = 30

        while calls[0] < budget and time.time() < deadline and de_iterations < max_de_iterations:
            de_iterations += 1

            # Adaptive mutation: reduce F as we converge
            if best_f < np.inf:
                # Estimate convergence rate
                if len(y_gp) > 5:
                    recent_f = y_gp[-min(10, len(y_gp)):]
                    f_range = max(recent_f) - min(recent_f)
                    if f_range > 1e-6:
                        F = np.clip(0.4 + 0.4 * (f_range / max(abs(best_f), 1e-6)), 0.3, 1.0)

            # Generate offspring via DE/rand/1
            offspring = []
            offspring_fitness = []

            for i in range(len(population)):
                if calls[0] >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(len(population), size=3, replace=False)

                # DE/rand/1 mutation
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Crossover
                crossover_rate = 0.7
                u = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < crossover_rate or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = population[i][j]
                u = np.clip(u, -5, 5)

                # Evaluate offspring
                try:
                    f_u = wrapped(u)
                    offspring.append(u.copy())
                    offspring_fitness.append(f_u)
                    X_gp.append(u.copy())
                    y_gp.append(f_u)
                    update_best(u, f_u)
                except Exception:
                    offspring.append(population[i].copy())
                    offspring_fitness.append(pop_fitness[i])

                # Selection
                if f_u < pop_fitness[i]:
                    population[i] = u.copy()
                    pop_fitness[i] = f_u

            # Remove empty offspring
            if len(offspring) < 2:
                break

            # Occasionally run local exploitation on the best population member
            if calls[0] < budget - 10 and time.time() < deadline and rng.random() < 0.25:
                # Find best in current population
                best_pop_idx = np.argmin(pop_fitness)
                best_pop_x = population[best_pop_idx].copy()
                try:
                    res = minimize(
                        wrapped,
                        best_pop_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxls': 15, 'gtol': 1e-5}
                    )
                    update_best(res.x, res.fun)
                    X_gp.append(res.x.copy())
                    y_gp.append(res.fun)
                    # Update population
                    population[best_pop_idx] = res.x.copy()
                    pop_fitness[best_pop_idx] = res.fun
                except Exception:
                    pass

        # --- Phase 3: GP-guided exploitation for remaining budget ---
        while calls[0] < budget and time.time() < deadline:
            if len(X_gp) >= 3:
                # Generate candidate points
                n_candidates = 30
                candidates = []

                # Random candidates
                for _ in range(n_candidates // 3):
                    candidates.append(rng.uniform(-5, 5, dim))

                # Perturbations around best
                for _ in range(n_candidates // 3):
                    pert = rng.standard_normal(dim) * 0.6
                    candidates.append(np.clip(best_x + pert, -5, 5))

                # Perturbations around recent good points
                if len(X_gp) > 5:
                    sorted_idx = np.argsort(y_gp)
                    top_k = min(4, len(y_gp))
                    for idx in sorted_idx[:top_k]:
                        rp = X_gp[idx]
                        pert = rng.standard_normal(dim) * 0.3
                        candidates.append(np.clip(rp + pert, -5, 5))

                # Deduplicate and clip
                unique_candidates = []
                for c in candidates:
                    c = np.clip(c, -5, 5)
                    if not any(np.allclose(c, u, atol=0.05) for u in unique_candidates):
                        unique_candidates.append(c)

                if not unique_candidates:
                    unique_candidates = [rng.uniform(-5, 5, dim)]

                cand_arr = np.array(unique_candidates)

                # GP predictions
                try:
                    mu, sigma = gp_predict(cand_arr)
                    ei = expected_improvement(mu, sigma, best_f)

                    # Select top candidate by EI
                    best_idx = np.argmax(ei)
                    next_x = cand_arr[best_idx]
                except Exception:
                    next_x = np.clip(best_x + rng.standard_normal(dim) * 0.8, -5, 5)
            else:
                next_x = rng.uniform(-5, 5, dim)

            # Evaluate the selected point
            try:
                f_next = wrapped(next_x)
                X_gp.append(next_x.copy())
                y_gp.append(f_next)

                # Local exploitation if improvement
                if f_next < best_f:
                    update_best(next_x, f_next)
                    if calls[0] < budget - 10 and time.time() < deadline:
                        try:
                            res = minimize(
                                wrapped,
                                next_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * dim,
                                options={'maxiter': 20, 'maxls': 10, 'gtol': 1e-5}
                            )
                            update_best(res.x, res.fun)
                            X_gp.append(res.x.copy())
                            y_gp.append(res.fun)
                        except Exception:
                            pass
                else:
                    if calls[0] < budget - 10 and time.time() < deadline and rng.random() < 0.2:
                        try:
                            res = minimize(
                                wrapped,
                                best_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * dim,
                                options={'maxiter': 15, 'maxls': 8, 'gtol': 1e-4}
                            )
                            if res.fun < best_f:
                                update_best(res.x, res.fun)
                                X_gp.append(res.x.copy())
                                y_gp.append(res.fun)
                        except Exception:
                            pass

            except Exception:
                pass

        # --- Phase 4: Final high-precision polish ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            # Coordinate descent polish
            cd_x = best_x.copy()
            step = 0.3
            for _ in range(15):
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        trial = cd_x.copy()
                        trial[d] = np.clip(trial[d] + direction * step, -5, 5)
                        try:
                            f_trial = wrapped(trial)
                        except Exception:
                            break
                        if f_trial < best_f:
                            update_best(trial, f_trial)
                            cd_x = trial.copy()
                            improved = True
                            break
                    else:
                        continue
                    break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break

            # L-BFGS-B final with tight tolerance
            if calls[0] < budget and time.time() < deadline:
                try:
                    res_final = minimize(
                        wrapped,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 80, 'maxls': 25, 'gtol': 1e-8}
                    )
                    update_best(res_final.x, res_final.fun)
                except Exception:
                    pass

            # Nelder-Mead final
            if calls[0] < budget and time.time() < deadline:
                try:
                    res_nm = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 80, 'xatol': 1e-10,
                                 'fatol': 1e-14, 'adaptive': True}
                    )
                    update_best(res_nm.x, res_nm.fun)
                except Exception:
                    pass

        return best_x, best_f
