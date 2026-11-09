"""
Hybrid optimizer for Büche-Rastrigin f4:
- Initial dense random sampling for early progress and basin identification
- CMA-ES with adaptive restart for global exploration
- L-BFGS-B for fast local refinement from best points
"""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid CMA-ES + L-BFGS-B optimizer with early random sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        lambda_pop = 9
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

        cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
        cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

        def run_cmaes(x_init, sigma_init, max_calls):
            nonlocal calls_made, best_x, best_f

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_f = evaluate(x_mean)
            local_best_x = x_mean.copy()

            gen = 0
            no_improve_gen = 0
            last_improvement = local_best_f

            while not check_deadline() and calls_made < max_calls:
                # Sample lambda points
                offspring = []
                for i in range(lambda_pop):
                    if check_deadline() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                # Sort by fitness
                offspring.sort(key=lambda x: x[1])

                # Update mean
                x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))

                hsig = (np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))

                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))

                # Update covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_new_mean
                gen += 1

                # Update best
                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    last_improvement = local_best_f
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                # Update basis
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                # Check for stall - restart if no improvement for 8 generations
                if no_improve_gen >= 8 and calls_made < max_calls - 10:
                    break

                # Check convergence
                if np.linalg.norm(D) < 1e-12 or sigma < 1e-12:
                    break

            return local_best_f, local_best_x, calls_made

        def run_lbfgsb(x_init, max_calls):
            """Run L-BFGS-B from x_init with remaining budget."""
            nonlocal calls_made, best_x, best_f

            if check_deadline() or calls_made >= max_calls:
                return

            # L-BFGS-B wrapper that tracks calls
            lbfgsb_calls = [0]

            def lbfgsb_func(x):
                nonlocal calls_made, best_x, best_f
                if check_deadline() or calls_made >= max_calls:
                    return np.inf
                f = func(x)
                calls_made += 1
                lbfgsb_calls[0] += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f

            try:
                result = minimize(
                    lbfgsb_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, max_calls - calls_made),
                             'maxfun': max(1, max_calls - calls_made),
                             'gtol': 1e-6}
                )
                # Evaluate the final point if it's better
                if result.success or result.fun < best_f:
                    f_final = func(np.clip(result.x, -5, 5))
                    calls_made += 1
                    if f_final < best_f:
                        best_f = f_final
                        best_x = np.clip(result.x, -5, 5).copy()
            except:
                pass

        # Phase 0: Dense random sampling for early progress
        # This gives us good initial estimates and identifies promising basins
        if not check_deadline():
            n_initial = min(50, total_budget // 10)
            for i in range(n_initial):
                if check_deadline():
                    break
                x = rng.uniform(-4.5, 4.5, dim)
                evaluate(x)

        # Phase 1: CMA-ES broad exploration from best initial point
        if not check_deadline() and best_x is not None:
            x_init = best_x.copy()
            sigma_init = 3.0
            alloc1 = int(total_budget * 0.30)
            max_calls_1 = min(calls_made + alloc1, total_budget)
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls_1)

        # Phase 2: L-BFGS-B refinement from current best
        if not check_deadline() and best_x is not None:
            alloc2 = int(total_budget * 0.25)
            max_calls_2 = min(calls_made + alloc2, total_budget)
            run_lbfgsb(best_x.copy(), max_calls_2)

        # Phase 3: CMA-ES from a different region
        if not check_deadline():
            if best_x is not None:
                x_init = rng.uniform(-4, 4, dim)
                for _ in range(10):
                    if np.linalg.norm(x_init - best_x) > 2.0:
                        break
                    x_init = rng.uniform(-4, 4, dim)
            else:
                x_init = rng.uniform(-4, 4, dim)
            sigma_init = 2.5
            alloc3 = int(total_budget * 0.20)
            max_calls_3 = min(calls_made + alloc3, total_budget)
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls_3)

        # Phase 4: L-BFGS-B refinement again
        if not check_deadline() and best_x is not None:
            alloc4 = int(total_budget * 0.15)
            max_calls_4 = min(calls_made + alloc4, total_budget)
            run_lbfgsb(best_x.copy(), max_calls_4)

        # Phase 5: Final CMA-ES refinement with small sigma
        if not check_deadline() and best_x is not None:
            x_init = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
            sigma_init = 0.8
            max_calls_5 = total_budget
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls_5)

        # Phase 6: Final L-BFGS-B polish
        if not check_deadline() and best_x is not None:
            max_calls_6 = total_budget
            run_lbfgsb(best_x.copy(), max_calls_6)

        return best_x, best_f
