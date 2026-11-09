import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB-style Lunacek bi-Rastrigin (f24).
    Uses CMA-ES with restarts to handle the deceptive bi-modal landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # 3 restarts with budgets that give CMA-ES enough evaluations to adapt
        # CMA-ES in 5D needs ~20-30 evaluations to start being effective
        num_restarts = 3
        budgets = [
            int(total_budget * 0.40),  # 200 calls
            int(total_budget * 0.35),  # 175 calls
            total_budget - int(total_budget * 0.40) - int(total_budget * 0.35)  # 125 calls
        ]

        # Initialization strategies for each restart
        def get_initial_x(restart_idx):
            if restart_idx == 0:
                return np.zeros(n)
            elif restart_idx == 1:
                return rng.uniform(-4, 4, n)
            else:
                # Perturb the best found so far to escape local minima
                if best_x is not None:
                    return np.clip(best_x + rng.standard_normal(n) * 2.5, -5, 5)
                else:
                    return rng.uniform(-5, 5, n)

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = min(budgets[restart_idx], total_budget - calls_made)
            if budget_this_restart < 10:
                break

            x0 = get_initial_x(restart_idx)
            fx0 = func(x0)
            calls_made += 1

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # CMA-ES parameters
            # In 5D, lambda (population size) of ~10 is reasonable
            # sigma0: initial step size, tuned for the [-5,5] box
            sigma0 = 1.5

            # Track calls within this restart
            restart_start_calls = calls_made

            # Use CMA-ES via scipy's minimize
            # We need to carefully count function calls
            def cma_func(x):
                nonlocal calls_made
                if calls_made >= total_budget:
                    return best_f  # Return best known to stop
                if time.time() > deadline:
                    return best_f
                val = func(np.clip(x, -5, 5))
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = np.clip(x, -5, 5).copy()
                return val

            # CMA-ES options
            # maxiter controls the number of iterations (each iteration uses lambda evaluations)
            # With lambda~10 and budget~200, we get ~20 iterations
            # maxfev is the total function evaluations for this call
            remaining = total_budget - calls_made
            maxfev = min(budget_this_restart - 1, remaining)  # -1 for the initial eval

            try:
                result = minimize(
                    cma_func,
                    x0,
                    method='Nelder-Mead',  # Fallback if CMA-ES not available
                    options={'maxiter': maxfev // 5, 'xatol': 1e-6, 'fatol': 1e-8}
                )
            except Exception:
                pass

            # Try CMA-ES specifically (scipy has it in newer versions)
            # Actually, scipy.optimize doesn't have CMA-ES built-in in all versions
            # Let's implement a simple CMA-ES ourselves for reliability

        # Actually, let me implement a proper CMA-ES manually since scipy's CMA-ES
        # may not be available. This is more reliable.

        # Reset and redo with manual CMA-ES
        best_x = None
        best_f = np.inf
        calls_made = 0

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = min(budgets[restart_idx], total_budget - calls_made)
            if budget_this_restart < 10:
                break

            x0 = get_initial_x(restart_idx)
            fx0 = func(x0)
            calls_made += 1

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Manual CMA-ES implementation
            # Population size lambda
            lam = max(4, int(4 + 3 * np.log(n)))  # ~11 for n=5
            # Initial step size
            sigma = 1.5
            # Mean
            m = x0.copy()
            # Covariance matrix
            C = np.eye(n)
            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # Learning rates
            cmu = 2 / ((n + 1.3) ** 2 + lam)
            ccs = (2 + cmu) / (n + 2)
            cs = (4 + cmu) / (n + 4)
            cc = (4 + cmu - cs) / 2
            # Damping
            damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cs
            dampc = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cc
            # Cholesky factor of C
            Chol = np.linalg.cholesky(C)
            # Inverse of Cholesky
            Chol_inv = np.linalg.inv(Chol)
            # 1/sqrt(diag(C))
            inv_sqrtdiagC = 1.0 / np.sqrt(np.diag(C))
            # hsig threshold
            hsig = (1.7 + 2.0 / (n + 1)) * (1 - np.exp(-1 / ccs))
            # Eigen decomposition for step size adaptation
            # We'll use a simpler adaptation

            # Evaluate initial population
            # Generate first population
            while calls_made < total_budget and time.time() < deadline:
                # Generate population
                pop = []
                for i in range(lam):
                    # Check budget
                    if calls_made >= total_budget:
                        break
                    if time.time() > deadline:
                        break
                    zi = rng.standard_normal(n)
                    xi = m + sigma * (Chol @ zi)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_made += 1
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                    pop.append((fi, xi))

                if len(pop) < 2:
                    break

                # Sort by fitness
                pop.sort(key=lambda x: x[0])
                f_sorted = [p[0] for p in pop]
                x_sorted = [p[1] for p in pop]

                # Update mean using weighted average of top mu points
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights ** 2)

                m_new = np.sum([weights[i] * x_sorted[i] for i in range(mu)], axis=0)

                # Update evolution paths
                # ps update
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) * (inv_sqrtdiagC @ (Chol_inv @ (m_new - m) / sigma))

                # hsig check
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made - restart_start_calls)) < hsig * np.sqrt(n + 1) - 1:
                    hsig_val = 1
                else:
                    hsig_val = 0

                # pc update
                pc = (1 - cc) * pc + hsig_val * cc * (m_new - m) / sigma

                # Update covariance
                delta_d = hsig_val * (1 + cc - (1 - cc)) - 1
                C = (1 + cmu * (mueff - 2 + delta_d) - cc * (1 - (1 - cc) ** 2)) * C
                for i in range(mu):
                    y = (x_sorted[i] - m) / sigma
                    C += cmu * weights[i] * np.outer(y, y)
                C += cmu * weights[0] * delta_d * np.outer(pc, pc)

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2

                # Update sigma (1/5 rule on success rate)
                # Count how many of the top mu are better than m
                success = sum(1 for i in range(mu) if f_sorted[i] < f_sorted[mu])
                rate = success / mu
                if rate > 0.4:
                    sigma *= 1.1
                elif rate < 0.2:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update Cholesky factor
                try:
                    Chol = np.linalg.cholesky(C)
                    Chol_inv = np.linalg.inv(Chol)
                    inv_sqrtdiagC = 1.0 / np.sqrt(np.diag(C))
                except np.linalg.LinAlgError:
                    # Reset if not positive definite
                    C = np.eye(n)
                    Chol = np.eye(n)
                    Chol_inv = np.eye(n)
                    inv_sqrtdiagC = np.ones(n)

                m = m_new

                # Check stagnation
                if calls_made - restart_start_calls > budget_this_restart:
                    break

        # Final coordinate descent polish
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(20):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    for direction in [-1, 1]:
                        y = x.copy()
                        y[i] = np.clip(y[i] + direction * step, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < f:
                            x, f = y, fy
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break

        if best_x is None:
            best_x = np.zeros(n)
            if calls_made < total_budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
