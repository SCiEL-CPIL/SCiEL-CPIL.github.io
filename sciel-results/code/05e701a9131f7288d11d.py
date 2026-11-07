import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Initial Sampling: Use a mix of random and structured points
        n_init = 20
        init_points = []
        
        # Random points
        for _ in range(n_init):
            if n_calls >= budget:
                break
            x = np.random.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
            init_points.append((f, x.copy()))

        # Sort by fitness
        init_points.sort(key=lambda x: x[0])
        
        # If we ran out of budget during init, return best
        if n_calls >= budget:
            return best_x, best_f

        x_best = init_points[0][1].copy()
        f_best = init_points[0][0]

        # 2. Multi-start Local Optimization (L-BFGS-B)
        # L-BFGS-B is very efficient for smooth functions and respects bounds.
        # We will run it from the top few starting points.
        
        # Helper to run L-BFGS-B with a call counter
        def run_lbfgsb(x0, max_calls_allowed):
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= budget or time.time() > deadline:
                return x0, best_f

            # Wrapper for scipy minimize
            def scipy_func(x):
                nonlocal n_calls
                if n_calls >= budget or time.time() > deadline:
                    return float('inf')
                x = np.clip(x, -5.0, 5.0)
                val = func(x)
                n_calls += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(
                    scipy_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Update best if scipy found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            return best_x, best_f

        # Run L-BFGS-B from the best initial point
        if n_calls < budget and time.time() < deadline:
            x_best, f_best = run_lbfgsb(x_best, 100)

        # Run L-BFGS-B from the second best initial point (diversity)
        if n_calls < budget and time.time() < deadline and len(init_points) > 1:
            x_start2 = init_points[1][1].copy()
            # Only if it's significantly different from best
            if np.linalg.norm(x_start2 - x_best) > 0.5:
                x_best2, f_best2 = run_lbfgsb(x_start2, 50)
                if f_best2 < f_best:
                    x_best, f_best = x_best2, f_best2

        # 3. CMA-ES for Global Exploration / Escape
        # If L-BFGS-B stagnates or if we have budget left, use CMA-ES to explore.
        # CMA-ES is robust to local minima and can find better basins.
        
        def run_cma_es(x_start, sigma_start, max_iter):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start, best_f

            N = 4 + int(np.floor(3 * np.log(dim)))
            mu = N // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1. / np.sum(weights ** 2)
            
            c1 = (4 + mu / (dim + 1.7)) / ((dim + 2) ** 2 + mu)
            cmu = min(1 - c1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))
            c_sigma = (mu + 2) / (dim + mu + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
            
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D, 1e-12, None))

            best_f_local = best_f
            best_x_local = x_start.copy()
            no_improvement = 0
            stagnation_limit = 10

            for it in range(max_iter):
                if n_calls >= budget or time.time() > deadline:
                    break

                y = np.zeros((N, dim))
                for i in range(N):
                    z = np.random.standard_normal(dim)
                    y[i] = B @ (D * z)

                x_pop = x_mean + sigma * y
                f_pop = []
                for i in range(N):
                    if n_calls >= budget or time.time() > deadline:
                        break
                    f_pop.append(timed_call(x_pop[i]))
                
                if len(f_pop) < N:
                    break

                order = np.argsort(f_pop)
                x_pop_sorted = x_pop[order]
                f_pop_sorted = np.array(f_pop)[order]

                if f_pop_sorted[0] < best_f_local:
                    best_f_local = f_pop_sorted[0]
                    best_x_local = x_pop_sorted[0].copy()
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement > stagnation_limit:
                    break

                x_mean_old = x_mean.copy()
                x_mean = x_mean_old + sigma * (weights[:mu] @ x_pop_sorted[:mu])

                y_old = y[order]
                y_w = y_old[:mu]
                delta_x = (x_mean - x_mean_old) / sigma
                pc = (1 - c1) * pc + c1 * delta_x
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c1) ** (it + 1)) < (1 + 2 / (dim + 1)) * chi_n)
                ps = (1 - c_sigma) * ps + float(hsig) * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_x
                
                C = (1 + c1 * cmu * hsig * (hsig - 1) - c1) * C + cmu * np.outer(y_w.T @ weights[:mu], y_w.T @ weights[:mu])
                
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D, 1e-12, None))

                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

            return best_x_local, best_f_local

        # Run CMA-ES from the current best with a moderate sigma
        if n_calls < budget and time.time() < deadline:
            x_best, f_best = run_cma_es(x_best, 0.5, 50)

        # 4. Final Polish with L-BFGS-B
        # After CMA-ES moves us to a better basin, use L-BFGS-B to converge precisely.
        if n_calls < budget and time.time() < deadline:
            x_best, f_best = run_lbfgsb(x_best, 50)

        # 5. One more CMA-ES restart with smaller sigma if budget remains
        if n_calls < budget and time.time() < deadline:
            x_best, f_best = run_cma_es(x_best, 0.1, 30)
            
        # 6. Final L-BFGS-B polish
        if n_calls < budget and time.time() < deadline:
            x_best, f_best = run_lbfgsb(x_best, 20)

        return best_x, best_f
