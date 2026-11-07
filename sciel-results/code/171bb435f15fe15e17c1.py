"""
Robust CMA-ES with integer-grid initialization for BBOB f3 (Rastrigin) on [-5, 5]^5.
Exploits Rastrigin's periodic structure via integer-grid sampling for initialization,
then refines with a well-tuned CMA-ES to handle hidden rotations and shifts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin for 330s wall-clock

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Integer-grid + perturbation initialization
        # Rastrigin has local minima at integer coordinates. After hidden rotation+shift,
        # these become a diverse set of points. Sample integer points and small perturbations.
        init_points = []

        # Integer points in [-4, 4]^5 (too many for full grid, so sample)
        # Use a structured sampling: sample 100 random integer points
        for _ in range(100):
            x_int = rng.integers(-4, 5, n)  # integers in [-4, 4]
            init_points.append(x_int.astype(float))

        # Add small perturbations around some integer points (near-integer)
        for _ in range(40):
            x_int = rng.integers(-4, 5, n)
            x_pert = x_int.astype(float) + 0.3 * rng.standard_normal(n)
            init_points.append(np.clip(x_pert, -5, 5))

        # Add origin and some structured points
        init_points.append(np.zeros(n))
        for scale in [1.0, 2.0, 3.0]:
            init_points.append(scale * rng.standard_normal(n))

        # Evaluate all init points
        for x in init_points:
            if check_deadline():
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)

        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        def run_cmaes(x_start, f_start, max_calls, pop_size=None, sigma_init=None):
            nonlocal calls_made, best_x, best_f

            if max_calls < 10:
                return

            # Population size: tuned for 5D
            if pop_size is None:
                pop_size = max(4, 4 + int(3 * np.log(n)))
                pop_size = min(pop_size, max_calls // 4)
            if pop_size < 2:
                return

            mu = max(1, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            # Initial sigma
            if sigma_init is None:
                sigma = 0.5
            else:
                sigma = sigma_init
            mean = x_start.copy().astype(float)

            C = np.eye(n)
            C_sqrt = np.eye(n)

            # CMA-ES parameters
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            no_improve = 0
            last_best = f_start
            generation = 0

            while not check_deadline() and calls_made < budget:
                # Check if we can afford a full generation
                if calls_made + pop_size > budget:
                    remaining = budget - calls_made
                    if remaining < mu:
                        break
                    # Do a partial generation
                    pop_size = remaining
                    mu = max(1, pop_size // 2)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= np.sum(weights)
                    mu_eff = 1.0 / np.sum(weights**2)

                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    offspring.append((fy, y))
                    if check_deadline():
                        break

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])
                gen_best = offspring[0][0]

                # Check improvement
                if gen_best < last_best - 1e-10:
                    last_best = gen_best
                    no_improve = 0
                else:
                    no_improve += 1

                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]

                old_mean = mean.copy()
                mean = y_w

                # Update paths
                ys = (mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n
                        < (1.4 + 2 / (n + 1)) * cc)

                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                else:
                    pc = (1 - cs) * pc

                # Update covariance
                rank_one = np.outer(ys, ys)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = ((1 - c1 - cmu) * C
                     + c1 * (np.outer(pc, pc) + (1 - hsig) * C)
                     + cmu * rank_mu)

                C = (C + C.T) / 2.0

                # Ensure positive definiteness
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-15)

                generation += 1

                # Convergence criteria
                if sigma < 1e-12:
                    break
                if no_improve > 30 and sigma < 0.05:
                    break
                if generation > 150:
                    break

            return best_x, best_f

        # Run initial CMA-ES from best point
        run_cmaes(best_x, best_f, budget - calls_made, sigma_init=0.5)

        # Phase 3: Adaptive restarts if budget remains
        remaining = budget - calls_made
        restart_attempts = 0
        max_restarts = 8

        while remaining > 30 and restart_attempts < max_restarts and not check_deadline():
            # Try restarting from different integer-grid points or with larger sigma
            found_improvement = False

            # Strategy 1: Restart from best_x with larger sigma
            if remaining > 50:
                x_restart = best_x.copy()
                run_cmaes(x_restart, best_f, remaining, sigma_init=1.0)
                remaining = budget - calls_made
                if remaining < 30:
                    break

            # Strategy 2: Try a few random integer-grid points
            for _ in range(3):
                if remaining < 40 or check_deadline():
                    break
                x_int = rng.integers(-4, 5, n).astype(float)
                f_int = func(x_int)
                calls_made += 1
                remaining = budget - calls_made
                if f_int < best_f:
                    best_f = f_int
                    best_x = x_int.copy()
                    found_improvement = True
                    if remaining > 40:
                        run_cmaes(best_x, best_f, remaining, sigma_init=0.3)
                        remaining = budget - calls_made
                if remaining < 30:
                    break

            restart_attempts += 1

            # If no improvement, reduce restart frequency
            if not found_improvement and restart_attempts >= 5:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
